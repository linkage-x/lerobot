# FR3 Chunk Action 分位数归一化设计（2026-03-27）

## 范围

本文记录 FR3 chunk-based action 按 offset 分位数归一化方案的设计与首版落地结果。

当前状态：

- ACT 首版已实现
- 当前只支持 ACT policy
- 当前只支持按 offset 的 `p02/p98`
- 当前只作用于 action chunk，不改 observation / state 的归一化路径
- 当前 stats 读取入口是 dataset metadata 下的 `policy_action_chunk_stats.chunk{chunk_size}.json`

目标：

- 改进 FR3 chunk-based policy 的动作归一化方式
- 显式尊重“future chunk 的不同 offset 上，动作分布并不相同”这一事实
- 在动作数据存在偏态、长尾、阶段切换和离群点时，优先考虑比全局 `mean/std` 更稳健的分位数归一化

本文讨论的是动作归一化设计，不是更宽泛的 `mask2ee` 合同。

## 本轮落地结论

本轮已完成的实现范围：

- 新增 `ActionChunkQuantileNormalizerProcessorStep`
- 新增 `ActionChunkQuantileUnnormalizerProcessorStep`
- ACT 训练入口可显式加载 `action_chunk_stats`
- ACT 训练时对 supervision action chunk 按 offset 做 `p02/p98 -> [-1, 1]` 归一化
- ACT 推理时对输出 action 按队列 offset 做反归一化
- 反归一化后对 quaternion 分量重新单位化
- FR3 runtime 在 `policy.reset()` 时同步 reset pre/post processor，避免 offset 指针漂移

当前没有做的事：

- 非 ACT policy 支持
- `p01/p99` 开关
- clipping 默认开启
- dataset 侧自动构建 chunk stats 的通用 builder
- checkpoint 侧额外复制一份独立 `policy_action_chunk_stats.json`

本轮验证结果：

- `py_compile` 通过
- `tests/processor/test_action_chunk_quantile_processor.py` 通过
- `tests/configs/test_fr3_train_config.py` 通过
- 已确认 FR3 `mask2ee` 训练配置会默认启用该能力

## 问题定义

当前仓库中的归一化能力，本质上仍是“每个 feature 一个全局 stats bundle”。

相关代码入口：

- 归一化模式枚举：`src/lerobot/configs/types.py`
- 归一化数学实现：`src/lerobot/processor/normalize_processor.py`
- processor 创建入口：`src/lerobot/policies/factory.py`
- ACT 的 chunk horizon 定义：`src/lerobot/policies/act/configuration_act.py`

今天的 `NormalizerProcessorStep` 和 `UnnormalizerProcessorStep`，对 `action` 这种 feature 仍然默认使用“一套全局统计量”。

这种做法只有在下面条件成立时才合适：

- 同一个 feature 下的动作维度整体共享一套稳定分布
- chunk 内所有 future offset 的动作分布近似一致
- 尾部不够严重，不会淹没常见动作主干

FR3 的 chunk 动作数据并不满足这些假设。

在 `outputs/datasets/lerobotv3_0310_100ep_aligned_ts` 上，已经观测到下面这个真实现象：

- 全局 `action.x`：`mean=0.155`，`std=0.114`，`p02=-0.006`，`p98=0.306`
- chunk `offset=0` 的 `action.x`：`mean=0.172`，`std=0.114`，`p02=-0.004`，`p98=0.312`
- chunk `offset=40` 的 `action.x`：`mean=0.238`，`std=0.056`，`p02=0.082`，`p98=0.314`

含义很直接：

- 同一个原始动作值，在不同 future offset 上可能代表完全不同的物理语义
- 后期 horizon 的动作分布可能既更窄，又整体偏移
- 因此，用一套全局 stats 去归一化整个 action chunk，会掩盖真正的重要结构

## 设计目标

为 chunk-based policy 引入“按 offset 感知”的动作归一化路径。

推荐的第一版目标：

- observation 归一化不变
- state 归一化不变
- 仅对 action chunk 引入新的 offset-aware 归一化
- 针对 chunk 内每个 future offset 单独统计一套稳健 stats
- 每个时间步都只使用它自己的那套 stats 来做 normalize / denorm

推荐优先尝试的稳健范围：

- 首选：`p02/p98`
- 对照：`p01/p99`

原因：

- 比全局 `mean/std` 更不怕偏态、长尾和离群点
- 更能保留主干分布里细小动作的分辨率
- 更符合 chunk action 本身“不同时间步分布不同”的事实

## Processor 设计建议

不要另起一整套和现有 processor 平行的体系。

更稳妥的方案是：

1. 扩展现有 normalization 框架
2. 继续复用 `PolicyProcessorPipeline`
3. 继续通过 `policy_preprocessor.json` 和 `policy_postprocessor.json` 落盘
4. 针对 action chunk 增加专用 processor step，或者在不破坏兼容性的前提下扩展现有 step

推荐实现方式：

- 现有 `NormalizerProcessorStep` / `UnnormalizerProcessorStep` 继续负责 observation 和非 chunk 输出
- 为 chunk action 单独增加专用 step

建议命名：

- `ActionChunkQuantileNormalizerProcessorStep`
- `ActionChunkQuantileUnnormalizerProcessorStep`

推荐拆成单独 step，而不是硬塞进当前 `_NormalizationMixin`，原因是：

- 当前 `_NormalizationMixin` 默认假设“每个 feature key 一套 stats”
- chunk 动作归一化需要额外引入“当前是 chunk 内哪个 offset”这个索引语义
- 独立出来更容易保持现有非 chunk policy 的行为完全不变

## 配置层设计建议

### Normalization Mode

不要直接复用当前 `QUANTILES` 的语义。

当前 `QUANTILES` 的意思是：

- 针对单个 feature key，使用全局 `q01/q99` 做缩放

而本设计需要的是：

- 只对 action chunk 生效
- 并且按 chunk offset 使用不同 stats

推荐新增一个显式模式，专门描述这种行为。

可选 enum 值示意：

- `ACTION_CHUNK_QUANTILES_02_98`
- 后续如有需要再补：`ACTION_CHUNK_QUANTILES_01_99`

如果觉得 enum 扩张太快，也可以保留 enum 简洁，转而引入一个 action-normalization 专用配置块。

推荐的第一版配置形态：

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

这种写法的优点是：

- 保存到 checkpoint 后可读性更强
- 审查 train / infer 合同时不需要靠推断

### Policy Config 需要提供的元信息

做 chunk-aware normalization 的 policy，至少要暴露出下面这些信息：

- `chunk_size`
- `n_action_steps`
- `action_delta_indices`

ACT 当前已经满足这些条件。

## Stats Artifact 设计建议

不要把这套信息含糊地塞进现有 dataset `meta/stats.json`，否则很容易混淆“全局 dataset stats”和“policy 相关的 chunk stats”。

当前首版采用的是 dataset metadata 独立文件：

1. 训练按 `dataset_root/meta/policy_action_chunk_stats.chunk{chunk_size}.json` 读取
2. 该文件作为训练合同的一部分进入 processor config
3. checkpoint 通过 `policy_preprocessor.json` / `policy_postprocessor.json` 自描述恢复行为

这样做的现实收益是：

- 不破坏现有 dataset `stats.json`
- stats 文件显式带 `chunk_size`
- processor 落盘后 train / infer 可锁定同一套按 offset 归一化合同

当前仍未做但未来可以补的增强：

- checkpoint 目录里显式复制一份 `policy_action_chunk_stats.json`
- 通用 stats builder 工具链
- 对同一 dataset 支持多种 quantile 方案并行命名

### 当前文件名约定

- `meta/policy_action_chunk_stats.chunk100.json`

泛化约定：

- `meta/policy_action_chunk_stats.chunk{chunk_size}.json`

### 建议 schema

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
    },
    {
      "offset": 1,
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

说明：

- 当前训练真正消费的是 `lower_quantile` / `upper_quantile` 对应的 stats，首版即 `q02/q98`
- 即使首版只用 `q02/q98`，仍建议保留 `q01/q99` 和 `median`，便于后续分析与对照
- `median` 对归一化公式本身不是必须，但对排查和可视化很有用
- 当前实现里，这份 artifact 由 dataset metadata 提供；processor config 会把实际用到的 stats 内嵌进 checkpoint

## 对数据集构建的影响

原始 dataset frame schema 不需要改。

不需要修改：

- canonical frame record
- `action` 列布局
- 视频对齐逻辑
- timestamp 合同

真正需要新增的是一条“离线统计构建”能力。

### 需要新增的 Stats Builder

建议新增一个工具，负责：

1. 读取训练实际使用的 episode 集合
2. 按 policy 的 `chunk_size` 枚举所有有效 chunk window
3. 按 offset 收集动作向量
4. 对每个 offset、每个动作维度计算 quantiles
5. 写出前述 schema 的 artifact

关键规则：

- 不要把 padding 或不完整 future chunk 混进去
- 如果训练只用一部分 episode，统计也必须只用相同子集
- artifact 里必须记录生成它时使用的 `chunk_size`

### 为什么 FR3 这里值得做

这不是泛泛而谈，而是被当前 FR3 数据真实驱动出来的需求。

比如在这份 aligned dataset 上：

- `offset=0` 的 `x`：`mean=0.172`，`std=0.114`，`p02=-0.004`，`p98=0.312`
- `offset=40` 的 `x`：`mean=0.238`，`std=0.056`，`p02=0.082`，`p98=0.314`

同一个原始值 `x=0.20`：

- 在全局 `mean/std` 下：归一化后约 `0.392`
- 在 `offset=0` 的 `mean/std` 下：约 `0.246`
- 在 `offset=40` 的 `mean/std` 下：约 `-0.667`
- 在 `offset=40` 的 `p02/p98` 下：约 `0.015`

也就是说：

- 对早期 horizon，它像是偏正一点的动作
- 对后期 horizon，它其实接近该 horizon 的中值附近

这正是全局归一化会误导 chunk policy 的地方。

## 训练链路需要改什么

### Processor 创建入口

当前训练路径里：

- `src/lerobot/scripts/lerobot_train.py` 会把 `dataset.meta.stats` 传给 `make_pre_post_processors(...)`

改造后建议：

- 当 chunk-aware action normalization 开启时，训练还要额外把 `action_chunk_stats` 传进去

建议 API 形态：

```python
make_pre_post_processors(
    policy_cfg=cfg.policy,
    dataset_stats=dataset.meta.stats,
    action_chunk_stats=chunk_action_stats,
)
```

### 训练时的 normalize

observation 侧不需要变成 chunk-aware。

真正要变的是 supervision target：

- 训练 loss 用到的 action chunk，必须对每个 offset 使用它自己的 stats 做 normalize

如果 target chunk 形状是 `(B, chunk_size, action_dim)`：

- `[:, 0, :]` 用 offset `0` stats
- `[:, 1, :]` 用 offset `1` stats
- …
- `[:, k, :]` 用 offset `k` stats

### 训练指标

主 loss 仍建议在归一化空间里算，数值会更稳。

但要新增物理单位监控：

- 每个 offset 的 position error，单位 mm
- 每个 offset 的 rotation error，单位 deg
- 每个 offset 的 gripper error，单位 mm

原因：

- 否则很容易出现“归一化 loss 变好了，但物理动作误差没变好”的错觉

### 四元数后处理

如果 action 里继续使用 quaternion 分量：

- 反归一化之后必须重新单位化 quaternion

原因：

- 分维度标量归一化/反归一化后，四元数分量很容易偏离单位球面
- offset-aware 的 robust scaling 会让这个问题更明显

如果未来把 rotation target 改成更适合回归的表示，这一步再重新评估。

## 推理链路需要改什么

推理侧必须同步支持，不能只改训练。

### Chunk 反归一化

模型输出的 action chunk，必须按 offset 分别做 denorm。

如果输出形状是 `(B, chunk_size, action_dim)`：

- `chunk[:, 0, :]` 用 offset `0` stats
- `chunk[:, 1, :]` 用 offset `1` stats
- …

这条是整个方案里最关键的实现点。

如果这里做错了，checkpoint 看起来可能还能正常输出数字，但实际执行到机器人上的物理动作会失真。

### Action Queue / Chunk 复用

如果 runtime 一次预测长 chunk，然后后续逐步消费这些动作：

- 某个动作一旦是以 `offset=40` 的身份被预测出来，就必须一直保留它原来的 offset 语义
- 不能因为“现在轮到它执行了”，就临时拿 `offset=0` 的 stats 再解释一遍

这点对下面场景尤其重要：

- ACT 且 `n_action_steps > 1`
- 任何会缓存并逐步消耗 chunk 的 runtime

### Checkpoint 落盘要求

checkpoint 里必须保存：

- processor config，明确指出 action chunk normalization 方式
- chunk stats artifact 本身

推理必须从 checkpoint 恢复它们，而不是运行时再从 dataset root 重新构造。

原因：

- checkpoint 必须是自描述的
- train / infer 必须锁死在同一份归一化合同上

## 建议的功能 rollout 顺序

建议分阶段做，不要一次铺太开。

### Stage 1

- 只支持 ACT
- 只作用于 action output
- 只做 `p02/p98`
- 不做 clipping
- denorm 后统一 renorm quaternion

### Stage 2

- 加入 `p01/p99` 作为显式对照开关
- 再考虑是否支持 clipping 和超界比例日志

### Stage 3

- 如果 ACT 验证有效，再推广到其他 chunk-based policy
- 如有必要，再讨论更复杂的统计条件化：
  - `offset + phase-aware`
  - `offset + contact-aware`

## 必须补的测试

首版实现已经覆盖了其中最关键的一部分，剩余项仍应视为后续 TODO。

### Stats Builder 测试

- 每个 offset 都生成了一套 stats
- episode 尾部不完整 chunk 被正确排除
- action 维度顺序保持不变
- artifact schema 落盘稳定、可重复

### Processor 单测

- normalize 再 unnormalize 后，action chunk 能回到原值附近
- offset `k` 的数据确实用了 offset `k` 的 stats，而不是全局 stats
- 推理单步 action 时，offset 指针会按 `n_action_steps` 正确推进并在 reset 后归零
- quaternion 在 inverse transform 之后确实被重新单位化

### 训练集成测试

- `lerobot_train.py` 在开启该模式时，能正确创建带 chunk stats 的 processor
- `TrainConfig` 能显式表达该能力已开启
- checkpoint 保存时 processor config 能恢复同一套 action-chunk 合同
- resume 训练时能恢复同一套 processor 合同

### 推理集成测试

- `make_pre_post_processors(..., pretrained_path=...)` 能从 checkpoint 恢复 chunk-aware action processor
- ACT 推理时对每个 predicted offset 使用了正确的 stats
- 缓存 later action 再执行时，offset 语义不丢失
- FR3 runtime reset 时，processor offset 状态也会同步 reset

### 回归测试

- 现有非 chunk policy 的归一化行为完全不变
- 现有 `MEAN_STD`、`MIN_MAX`、`QUANTILES`、`QUANTILE10` 行为完全不变

## 当前仍未决的设计问题

- 是不是应该新增一个显式 `NormalizationMode`，还是引入 action-normalization 专用 config block
- chunk stats 是否还要在 checkpoint 目录里显式复制一份可读 artifact
- `2/98` 是否应该成为默认 robust 范围，还是先作为显式实验开关
- quaternion action 表示本身是否也应该在未来一起重新设计

## 建议的首轮实验矩阵

不要只盯训练 loss。

建议实验矩阵：

1. baseline
   - 全局 `mean/std`
2. quantile-only
   - per-offset `p02/p98`
3. masking-only
   - partial proprio mask + 现有归一化
4. combined
   - partial proprio mask + per-offset `p02/p98`

主要验收信号：

- 长 horizon 的 offline action error tail 是否下降
- dataset-fed validation 里的 `frame 24/32/40` 是否明显改善
- preview 稳定性是否先于真机 rollout 获得提升

## 当前建议

对这条 FR3 线，最务实的下一步实现目标是：

- 保持当前 ACT-only 首版不扩散
- 用新 processor 重新训练一轮 `mask2ee` 模型做对照
- 继续拿当前全局 `mean/std` baseline 做 A/B
- 重点看 dataset-fed validation 的 `frame 24/32/40` 尾部是否改善
- 在确认有效之前，不要急着推广到其他 policy
