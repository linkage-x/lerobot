# FR3 ACT Tactile Encoder 决策与实现说明

## Context

- Dataset: `outputs/datasets/lerobotv3_0310_100ep`
- Policy target: ACT training on FR3 EE-to-EE pick-place data
- Tactile features exported in dataset:
  - `observation.tactile.left_clean`
  - `observation.tactile.right_clean`
  - `observation.tactile.left_raw`
  - `observation.tactile.right_raw`
  - `observation.tactile.valid_mask`

## Dataset Facts

- Single-side tactile shape: `[50, 10]`
- Layout: `row-major`
- Valid cells: `448`
- Invalid cells: `52`
- Clean tactile rule per side:

```text
clean = (raw - baseline) * valid_mask
```

- `valid_mask` is fixed sensor geometry, not frame-varying missingness
- Invalid cells in `*_clean` are already forced to `0.0`

---

## 从 first-pass 到当前实现

最初的 first-pass 决策是：

- 只使用 `left_clean` 和 `right_clean`
- 不显式消费 `valid_mask`
- 用共享 lightweight CNN 把触觉图编码成 tactile tokens

这样做的好处是：

- 结构简单
- 先把 tactile 接入 ACT 跑通
- 不过早扩大输入契约

但在进一步评估后，当前实现已经升级为三步增强：

1. `clean + valid_mask` 两通道输入
2. CNN stem 后增加轻量 residual block 和 SE/gating
3. 在 tactile tokens 进入主 ACT encoder 之前，增加 1 层 tactile-only Transformer

---

## 当前实现摘要

当前 FR3 DAS 配置默认启用了以下 tactile 结构：

- `tactile_use_valid_mask: true`
- `tactile_valid_mask_feature_key: observation.tactile.valid_mask`
- `tactile_encoder_residual_blocks: 1`
- `tactile_encoder_use_se: true`
- `tactile_transformer_layers: 1`

整体结构可以概括为：

```text
left_clean/right_clean + valid_mask
-> shared CNN stem
-> residual block + SE/gating
-> side embedding
-> flatten to tactile tokens
-> tactile-only Transformer
-> append before image tokens in ACT encoder
```

---

## 为什么从 `clean-only` 升级到 `clean + valid_mask`

这个问题的关键不在于：

> `clean` 里有没有包含 `valid_mask` 的影响

而在于：

> 模型能不能明确区分“无效单元的 0”和“有效单元但当前没接触的 0”

虽然：

- `clean = (raw - baseline) * valid_mask`

所以无效区域一定是 `0`。

但如果只给 `clean`，模型看到的 `0` 仍然存在歧义：

- 可能是无效区域
- 也可能是有效区域但当前没有触觉变化

把 `valid_mask` 显式作为第二通道输入的价值在于：

- 去掉这种歧义
- 显式提供传感器几何先验
- 降低模型自己去反推出固定无效区域的学习负担

为什么 first-pass 当时没有直接上它？

- 因为 `valid_mask` 是静态的
- 用 `clean-only` 也确实能先跑通训练

为什么现在值得加？

- 因为这是小改动
- 风险明显低于大规模重构
- 且比继续停留在 `clean-only` 更有信息表达优势

---

## 为什么在现有 CNN 后面加 residual block / SE / gating

### Residual block 的作用

Residual block 的核心不是“让网络盲目更深”，而是：

- 让网络更容易学到对已有特征的增量修正
- 让梯度传播更稳
- 降低深一点以后训练退化的风险

对 tactile 来说，这很合适，因为触觉图常常已经有较强的局部结构：

- 接触边缘
- 压力热点
- 条状或块状接触模式

Residual 更像是在已有触觉表示上做修正，而不是每一层都重写它。

### SE / gating 的作用

SE/gating 的核心是：

- 按当前输入动态调整不同通道的重要性

也就是说，模型可以根据当前这帧触觉：

- 强调更像真实接触模式的通道
- 压低更像噪声或无关模式的通道

它解决的是“提出来的特征里，当前到底该信谁”的问题。

### 为什么它们比直接换主干更优先

因为它们：

- 改动小
- 风险低
- 不改变整个 ACT 的融合方式
- 和当前小尺寸触觉图的 CNN 主干非常兼容

---

## 为什么试 `CNN stem + tactile-only Transformer`

这个设计的原理是：

- CNN 负责局部空间模式
- Transformer 负责 token 之间的全局关系

先用 CNN stem 的原因：

- `50x10` 是一个很小、很规整的 2D 网格
- CNN 对这种局部空间结构有天然归纳偏置
- 可以先把原始图压缩成少量 token，降低后续 attention 成本

再加 1 层 tactile-only Transformer 的原因：

- 让 tactile token 先在模态内部做一轮关系建模
- 学习非局部接触组合关系
- 再去和图像、状态做多模态融合

它比 pure CNN 多了一层全局建模能力，也比 pure Transformer 更节省 token 和数据。

为什么只先放 1 层：

- 当前 tactile token 数量本来就不大
- 1 层足够验证收益
- 更深很容易提高复杂度但不一定带来同等收益

---

## 是否要用 cross-attention

当前判断是：

- 先不把 cross-attention 当优先项

原因是现在的 ACT 已经把 tactile tokens 和 image tokens 拼进同一个 encoder 序列里，经过 encoder self-attention 之后，它们已经能相互交互。

所以如果现在再单独引入 tactile-image cross-attention，会带来这些额外成本：

- 需要拆出更明确的模态内编码器与模态间融合器
- 结构更复杂
- 调参面更大
- 短期收益不一定明显

只有在下面这类信号出现时，cross-attention 才会更值得考虑：

- tactile token 明显被 image token 淹没
- 希望做明确的非对称融合
- 需要更强的“tactile 查询 image”或“image 查询 tactile”机制

所以当前顺序仍然是：

1. `clean + valid_mask`
2. residual / SE
3. tactile-only Transformer
4. 之后再考虑显式 cross-attention

---

## 是否应该直接用 Transformer 替代 CNN

当前判断是：

- 不建议直接替代

原因有三点：

1. 当前触觉图很小，且局部结构明显，CNN 是合理默认选项
2. pure Transformer 会更依赖 token 化设计和数据量
3. 先保留 CNN，再在 token 级别加小型 Transformer，是更稳的折中路线

所以当前不是：

- `CNN vs Transformer`

而是：

- `CNN as stem + tiny Transformer as refinement`

---

## 当前落地顺序

按照实现优先级，当前代码采用的顺序是：

1. `clean + valid_mask` 两通道
2. CNN 后加 residual block 与 SE/gating
3. 加 1 层 tactile-only Transformer

这个顺序的原则是：

- 先补显式先验
- 再增强局部特征提取
- 最后再补全局关系建模

---

## 后续可继续做的 ablation

- compare `clean-only` vs `clean+mask`
- compare `clean+mask` vs `clean+raw+mask`
- compare `CNN-only` vs `CNN+residual/SE`
- compare `CNN+residual/SE` vs `CNN+residual/SE+1-layer tactile Transformer`
- only after the above: compare encoder self-attention fusion vs explicit cross-attention fusion

---

## 相关文件

- `src/lerobot/policies/act/configuration_act.py`
- `src/lerobot/policies/act/modeling_act.py`
- `src/lerobot/policies/act/processor_act.py`
- `src/lerobot/configs/franka_research3_ee2ee_act_das.yaml`
- `docs/act_policy_mermaid.md`
