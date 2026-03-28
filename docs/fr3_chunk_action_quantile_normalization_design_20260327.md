# FR3 Chunk Action 分位数归一化设计（2026-03-27）

## 目的

本文只讨论 FR3 chunk-based policy 的动作归一化设计。

为避免把“当前代码事实”和“后续设计建议”混在一起，本文拆成两层：

- 第一层：已实现事实
- 第二层：未来建议 / TODO

如果两层内容冲突，以“已实现事实”和当前代码为准。

## 第一层：已实现事实

### 当前范围

当前仓库已经落地的范围是：

- 只支持 ACT policy
- 只作用于 action chunk
- observation / state 归一化路径不变
- 当前量化范围固定为按 offset 的 `p02/p98`
- 当前 stats 默认读取路径是 `dataset_root/meta/policy_action_chunk_stats.chunk{chunk_size}.json`

对应代码：

- 配置字段：[src/lerobot/policies/act/configuration_act.py](/home/hph/Code/lerobot-replay/src/lerobot/policies/act/configuration_act.py)
- processor 创建与 stats 加载：[src/lerobot/policies/act/processor_act.py](/home/hph/Code/lerobot-replay/src/lerobot/policies/act/processor_act.py)
- action chunk quantile step：[src/lerobot/processor/action_chunk_quantile_processor.py](/home/hph/Code/lerobot-replay/src/lerobot/processor/action_chunk_quantile_processor.py)

### 当前问题定义

当前通用 `NormalizerProcessorStep` / `UnnormalizerProcessorStep` 对 `action` 仍然默认使用“一套全局 stats”。

这在下面条件成立时才合理：

- 同一个 feature 下的动作维度整体共享一套稳定分布
- chunk 内所有 future offset 的动作分布近似一致
- 尾部不够严重，不会明显淹没常见动作主干

FR3 的 chunk 动作数据不满足这些假设。

在 `outputs/datasets/lerobotv3_0310_100ep_aligned_ts` 上，已经观测到下面这个例子：

- 全局 `action.x`：`mean=0.155`，`std=0.114`，`p02=-0.006`，`p98=0.306`
- chunk `offset=0` 的 `action.x`：`mean=0.172`，`std=0.114`，`p02=-0.004`，`p98=0.312`
- chunk `offset=40` 的 `action.x`：`mean=0.238`，`std=0.056`，`p02=0.082`，`p98=0.314`

这至少说明：

- 不同 future offset 上，动作分布确实会偏移
- 同一个原始动作值，在不同 offset 上的相对位置可能不同
- 用一套全局 stats 去解释整个 action chunk，存在掩盖结构的风险

这里的证据是“支持继续实验”，不是“已经完成最终证明”。

### 当前实现方式

当前实现没有硬改通用 `_NormalizationMixin`，而是给 ACT action chunk 单独加了两个 step：

- `ActionChunkQuantileNormalizerProcessorStep`
- `ActionChunkQuantileUnnormalizerProcessorStep`

这样做的实际含义是：

- 通用 normalizer 继续负责 observation 和其他非 chunk 行为
- ACT 的 action 在开启该模式时先把通用 `ACTION` 归一化切到 `IDENTITY`
- 然后由专用 quantile step 处理 action chunk

### 当前训练链路

当前训练入口已经支持在 ACT 打开该能力时额外加载 `action_chunk_stats`。

实际行为：

- `lerobot_train.py` 在创建 processor 时传入 `dataset.meta.stats`
- 如果 `cfg.policy.action_chunk_quantile_normalization=true`，还会额外调用 `load_action_chunk_stats(...)`
- `make_act_pre_post_processors(...)` 会把这份 `action_chunk_stats` 同时接进 preprocessor 和 postprocessor

对训练 supervision target 的实际效果是：

- 若 action 张量形状是 `(B, chunk_size, action_dim)`，每个 offset 都使用自己的 `q02/q98`

### 当前推理链路

当前推理侧已经同步支持该能力。

实际行为分两类：

- 如果 postprocessor 收到完整 chunk，形状满足 `(..., chunk_size, action_dim)`，就按每个 offset 分别做反归一化
- 如果 postprocessor 收到单步 action，当前实现会维护一个内部 offset 指针，按 `0 -> 1 -> ... -> n_action_steps-1` 循环推进

这与当前 ACT 的 action queue 语义是匹配的：

- ACT 在 `n_action_steps > 1` 时，会先预测一段 action，再按队列逐个消费
- postprocessor 的单步 offset 指针正是为这条路径服务

这里需要明确：

- 当前实现保证的是“ACT 当前队列语义下”的 offset 对齐
- 这还不是一个对所有 chunk runtime 都自动成立的通用合同

### 当前 reset 语义

当前 FR3 real runtime 在 episode/reset 边界会同步 reset：

- `policy`
- `preprocessor`
- `postprocessor`

这一步已经落地，目的就是避免 postprocessor 的单步 offset 指针漂移。

### 当前 artifact / checkpoint 语义

当前 artifact 语义需要精确定义：

- 训练前的原始 stats 来源仍是 dataset metadata 文件
- 进入 processor 后，实际用到的 quantile 参数和 `offset_stats` 会被写进 `policy_preprocessor.json` / `policy_postprocessor.json`
- 当前没有额外复制一份独立的 `policy_action_chunk_stats.json` 到 checkpoint 根目录

所以当前 checkpoint 的“自描述”含义是：

- processor config 内已经包含了这套按 offset 的归一化合同
- 但 checkpoint 根目录里不一定存在一份单独、可直接查看的 chunk stats artifact 副本

### 当前四元数处理

如果 action names 能解析出 `qx/qy/qz/qw`，当前 postprocessor 在反归一化之后会对四元数重新单位化。

这能保证：

- 反归一化后的四元数长度回到单位球面

这不能保证：

- 四元数表示本身就已经是最优回归表示
- 逐维缩放再 renorm 没有几何失真

### 当前配置接口

当前代码里的真实配置接口是：

```yaml
policy:
  action_chunk_quantile_normalization: true
  action_chunk_stats_path: null
  action_chunk_quantile_clip: false
```

不是独立的 `action_chunk_normalization` 配置块。

### 当前未实现项

下面这些在当前代码里还没有做：

- 非 ACT policy 支持
- `p01/p99` 开关
- clipping 默认开启
- dataset 侧自动构建 chunk stats 的通用 builder
- checkpoint 根目录额外复制独立 `policy_action_chunk_stats.json`

### 当前验证覆盖

本轮已经确认的验证项：

- `tests/processor/test_action_chunk_quantile_processor.py`
  - chunk round-trip
  - 单步 action offset 推进与 reset
  - pipeline save/load
  - 无 action 时 normalizer noop
- `tests/configs/test_fr3_train_config.py`
  - FR3 相关训练配置可解析
  - `mask2ee` / `qoff` 配置会显式打开该能力

本轮没有被同等强度验证到的内容：

- 通用 stats builder
- 更完整的 checkpoint 恢复集成测试
- 更广泛的 offline/real A/B 效果结论

## 第二层：未来建议 / TODO

本节内容不是“当前代码事实”，而是建议方向。

### 设计目标

建议继续坚持下面这个方向：

- 仅对 action chunk 引入 offset-aware 归一化
- observation 归一化保持不变
- state 归一化保持不变
- 每个 future offset 单独统计一套 stats
- 每个时间步只使用自己的那套 stats 做 normalize / denorm

### 为什么值得继续做

基于当前 FR3 数据，offset-aware 归一化至少有明确动机。

例如同一个原始值 `x=0.20`：

- 在全局 `mean/std` 下：约 `0.392`
- 在 `offset=0` 的 `mean/std` 下：约 `0.246`
- 在 `offset=40` 的 `mean/std` 下：约 `-0.667`
- 在 `offset=40` 的 `p02/p98` 下：约 `0.015`

这说明“同一个数值在不同 horizon 上代表的相对语义不同”这件事，值得继续做更系统的验证。

但建议补更完整证据：

- 覆盖全部 action 维度
- 覆盖更多 offset
- 给出超界比例
- 给出多 episode / 多阶段统计，而不是只举一个维度例子

### 推荐的后续配置形态

从可读性上看，未来更推荐引入一个显式配置块，而不是继续扩散平铺字段。

建议形态：

```yaml
policy:
  action_chunk_normalization:
    enable: true
    method: quantile_per_offset
    lower_quantile: 0.02
    upper_quantile: 0.98
    clip: false
    quaternion_postprocess: renorm
```

这样更容易审查 train / infer 合同。

但这是未来建议，不是当前实现。

### 推荐的 stats artifact 方向

未来仍建议保持 dataset `meta/stats.json` 和 policy-specific chunk stats 分离。

推荐保留独立文件：

- `meta/policy_action_chunk_stats.chunk{chunk_size}.json`

推荐 schema 至少包含：

```json
{
  "version": 1,
  "feature_key": "action",
  "method": "quantile_per_offset",
  "lower_quantile": 0.02,
  "upper_quantile": 0.98,
  "chunk_size": 100,
  "action_dim": 8,
  "action_names": {
    "motors": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]
  },
  "offset_stats": [
    {
      "offset": 0,
      "q02": [...],
      "q98": [...],
      "q01": [...],
      "q99": [...],
      "median": [...]
    }
  ],
  "quaternion_postprocess": "renorm"
}
```

这里也需要明确：

- 对当前实现真正必需的是 `lower_quantile` / `upper_quantile` 对应统计和 `offset_stats`
- `q01/q99/median` 更偏向分析辅助字段，未来如果保留，建议明确写成 optional

### 推荐新增的 Stats Builder

未来应补一条正式的离线统计构建链路，而不是长期依赖手工准备 metadata 文件。

builder 建议负责：

1. 读取训练实际使用的 episode 集合
2. 按 policy `chunk_size` 枚举有效 chunk window
3. 按 offset 收集动作向量
4. 对每个 offset、每个动作维度计算 quantiles
5. 写出稳定 schema 的 artifact

建议强制满足：

- 不混入 padding 或不完整 future chunk
- 如果训练只用部分 episode，统计也只用同一子集
- artifact 必须记录 `chunk_size`

### 推荐补充的训练指标

未来不要只看归一化空间的 loss。

建议额外监控物理单位指标：

- 每个 offset 的 position error，单位 mm
- 每个 offset 的 rotation error，单位 deg
- 每个 offset 的 gripper error，单位 mm

原因很简单：

- 否则容易出现“训练 loss 变好，但物理动作误差没改善”的错觉

### 推荐补充的测试

未来建议补齐下面几类测试：

- Stats Builder 测试
  - 每个 offset 都生成 stats
  - episode 尾部不完整 chunk 被排除
  - action 维度顺序稳定
- 训练集成测试
  - `lerobot_train.py` 在打开该模式时能创建正确 processor
  - checkpoint 保存和恢复后合同保持一致
  - resume 训练后合同不漂移
- 推理集成测试
  - `make_pre_post_processors(..., pretrained_path=...)` 恢复的 chunk-aware processor 行为正确
  - later action 被缓存再执行时，offset 语义不丢失
- 回归测试
  - 非 chunk policy 行为完全不变
  - 现有 `MEAN_STD`、`MIN_MAX`、`QUANTILES`、`QUANTILE10` 行为完全不变

### 推荐的 rollout 顺序

建议继续按小范围 rollout：

- Stage 1
  - 只支持 ACT
  - 只作用于 action output
  - 只做 `p02/p98`
  - 不默认开 clipping
- Stage 2
  - 加 `p01/p99` 对照开关
  - 补 clipping 和超界比例日志
- Stage 3
  - 如果 ACT 结果稳定，再考虑推广到其他 chunk-based policy
  - 如有必要，再讨论更复杂的条件化统计，比如 `offset + phase-aware`

### 推荐的实验矩阵

建议继续按下面矩阵做 A/B，不要只盯训练 loss：

1. baseline
   - 全局 `mean/std`
2. quantile-only
   - per-offset `p02/p98`
3. masking-only
   - partial proprio mask + 现有归一化
4. combined
   - partial proprio mask + per-offset `p02/p98`

建议重点看：

- 长 horizon 的 offline action error tail 是否下降
- dataset-fed validation 的 `frame 24/32/40` 是否改善
- preview 稳定性是否先于真机 rollout 获得提升

## 当前建议

对这条 FR3 线，当前最务实的下一步仍然是：

- 保持 ACT-only 首版，不急着外扩
- 用这套 processor 重新训练一轮 `mask2ee` 模型做对照
- 继续拿全局 `mean/std` baseline 做 A/B
- 重点看 dataset-fed validation 的长 horizon 尾部是否改善
- 在证据足够之前，不要把它写成“已证明的通用归一化合同”
