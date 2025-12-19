# 通用 OT Loss 设计方案（LeRobot）

本文档描述如何将目前 ACT policy 内部的 OT 计算逻辑抽象为通用、可配置的 OT 模块，使其：

- 不再限定于 7 关节 + 1 夹爪状态；
- 支持多种特征（state/eef pose/image embedding 等）组合构建 OT 代价；
- 支持可学习 embedding 与直接 label cost 的灵活加权；
- 可被不同 policy（不仅是 ACT）复用。

---

## 1. 设计目标

1. **通用性**  
   - OT loss 不绑定特定 policy（如 ACT），而是一个独立模块；  
   - 支持任意张量特征对作为 OT 输入（例如 joint-state、eef pose、latent embedding 等）。

2. **可配置性**  
   - 通过配置指定：
     - 使用哪些特征；
     - 每个特征取哪一段维度；
     - 是否使用可学习 embedding；
     - 每个 cost term 的权重、正则参数等。

3. **梯度行为可控**  
   - cost 可以分为：
     - **embedding cost**：通过可学习 encoder 得到的表示，带梯度；
     - **label cost**：在原始空间上的距离（可选 detach），只作为 OT ground cost，不回传到 encoder。

4. **保持与 ot-sim2real 思路一致**  
   - 仍然使用 POT 的 Sinkhorn（或 unbalanced Sinkhorn）求解 OT 计划 π；  
   - OT loss 定义为 `Σ πᵢⱼ Mᵢⱼ`。

---

## 2. 模块拆分

建议新建模块：`src/lerobot/policies/ot_train/ot_loss.py`，负责：

1. 建立代价矩阵 `M`：
   - 来自多个 cost term（每个 term 对应一个特征或特征组合）；  
   - 每个 term 可有自己的权重。

2. 调用 POT 求 OT plan π：
   - 用 `ot.sinkhorn`（balanced）或后续扩展 `ot.unbalanced.sinkhorn_knopp_unbalanced`；  
   - 只在 **detached cost** 上运行 OT 算法（避免把 Sinkhorn 的内部细节纳入反向图）。

3. 根据 `M` 和 π 计算 OT loss 与统计指标。

整体调用关系建议：

- **policy**：
  - 定义并持有 OT embedding heads（`nn.ModuleDict`），参与训练。
- **训练脚本**：
  - 从 OT dataloader 拿到 `ot_pair`（src/tgt obs）；  
  - 按配置抽取特征张量，调用 OT loss 模块；  
  - 将返回的 `ot_loss` 与主 BC loss 组合。
- **OT loss 模块**：
  - 纯函数式逻辑：从特征张量与配置构造 cost，求 OT，返回 loss + metrics；
  - Learned embedding 仅来自各 policy 的观测编码（encode_feature_for_ot），不再支持独立 OTEmbeddingHead 回退。

---

## 3. OT 代价配置：OTTerm / OTCostConfig

### 3.1 term 概念

一个 OT 的总代价矩阵 `M` 可以看作多个子 cost term 的加权和：

- 例如：  
  - term1：关节空间 embedding cost  
  - term2：关节空间 label cost  
  - term3：eef pose cost  

为此定义：

```python
@dataclass
class OTTermConfig:
    name: str              # term 名称，用于日志和调试
    weight: float = 1.0    # 该 term 在总 cost 中的权重
```

### 3.2 总 OT cost 配置

```python
@dataclass
class OTCostConfig:
    terms: list[OTTermConfig]
    reg: float = 0.01      # Sinkhorn 正则系数
    # 可扩展:
    # reg_m: tuple[float, float] | None = None  # unbalanced 情况下的 tau1, tau2
```

OT 模块负责：

1. 根据不同 term 配置，构造同形状的 cost 矩阵（`M_term`）；  
2. 按 `term.weight` 线性组合得到总 `M`：
   ```python
   M = Σ (term.weight * M_term)
   ```  
3. 调用 POT：
   ```python
   pi = ot.sinkhorn(a, b, M.detach().cpu().numpy(), reg)
   ```  
4. 在 torch 张量上计算：
   ```python
   ot_loss = (pi_tensor * M).sum()
   ```

---

## 4. 特征配置：OTFeatureSpec / OTLossConfig

为了使 OT cost 适用于各种“特征对”，定义一个更高层的配置，描述每个特征如何从 `ot_pair` 中取出、如何构造 cost。

### 4.1 OTFeatureSpec

```python
@dataclass
class OTFeatureSpec:
    # 从哪取特征（通常来自 ot_pair["src"]["obs"] / ot_pair["tgt"]["obs"]）
    src_key: str                  # e.g. "observation.state"
    tgt_key: str                  # 通常与 src_key 相同

    # 在该特征维度上取哪一段（用于 joints+gripper 这种子空间）
    dim_slice: slice | None       # e.g. slice(0, 8) -> 前 8 维

    # 是否使用可学习 embedding（否则只用 label cost）
    use_learned_embed: bool
    embed_name: str | None        # 在 policy.ot_heads 里查哪一个 encoder

    # 每种 cost 的权重
    weight_embed: float = 1.0     # embedding cost 权重
    weight_label: float = 1.0     # label cost 权重
```

### 4.2 OTLossConfig

综合多个特征：

```python
@dataclass
class OTLossConfig:
    features: list[OTFeatureSpec]
    reg: float = 0.01
```

OT loss 模块的高层函数可以是：

```python
def compute_ot_loss_for_policy(
    policy: nn.Module,
    src_obs: dict[str, Tensor],
    tgt_obs: dict[str, Tensor],
    cfg: OTLossConfig,
) -> tuple[Tensor, dict]:
    ...
```

行为：

1. 对每个 `OTFeatureSpec`：
   - 从 `src_obs[spec.src_key]`、`tgt_obs[spec.tgt_key]` 中切出 `[ :, dim_slice ]`；  
   - 若 `use_learned_embed=True`：
     - 调 `policy.ot_heads[spec.embed_name]` 得到 embedding；  
     - `M_embed = cdist(src_embed, tgt_embed)^2`  
   - label cost：
     - 一般使用 `cdist(src_slice, tgt_slice)^2`，可选择 `detach()`。  
   - 按 `weight_embed` / `weight_label` 组合为一个 term cost：
     ```python
     M_term = weight_embed * M_embed + weight_label * M_label
     ```
2. 将所有 feature term 放入 term dict（`{spec.name: M_term}`），再交给 `compute_ot_loss_from_terms` 按 `OTCostConfig` 组合、求 PI 和最终 `ot_loss`。

特性：

- joints+gripper 是一个特征（`dim_slice=slice(0, 8)`）；  
- eef pose 可以是另一个特征（`src_key="observation.eef_pose", dim_slice=None`）；  
- 可以配置多个 feature term，一起构成最终 cost。

---

## 5. 可学习 OT Embedding Head

为了与 robomimic 行为一致，embedding cost 应直接来源于策略自身的观测编码。各策略应实现：

```python
def encode_feature_for_ot(self, key: str, x: Tensor) -> Tensor | None
```

若未实现该接口，OT 模块会对部分内置策略采用轻量启发式复用其内部 encoder（例如 ACT 的输入投影，Diffusion 的 rgb_encoder）。若仍无法得到编码，将直接报错。

---

## 6. 训练循环改造思路

在训练脚本（`lerobot_train.py`）中，OT 相关部分建议进行如下重构：

1. **保持数据层不变**  
   - `LeRobotOTPairDataset` 继续负责从 src/tgt LeRobotDataset + DTW pair_info 生成：
     ```python
     {
       "src": {"obs": {...}, "actions": ...},
       "tgt": {"obs": {...}, "actions": ...},
     }
     ```

2. **构建正统 OT 所需特征对**  
   - 在训练循环中，从 `ot_pair["src"]["obs"]` / `ot_pair["tgt"]["obs"]` 取出原始特征（未经归一化）：
     ```python
     src_obs_raw = ot_pair["src"]["obs"]
     tgt_obs_raw = ot_pair["tgt"]["obs"]
     ```
   - 按 `OTLossConfig.features` 提取各个 feature 的 `(src_tensor, tgt_tensor)`。

3. **调用通用 OT loss 模块**  
   - 如果 `cfg.ot.lambda_ot != 0` 且特征满足要求，则调用：
     ```python
     ot_loss, ot_metrics = compute_ot_loss_for_policy(
         policy=policy,
         src_obs=src_obs_raw,
         tgt_obs=tgt_obs_raw,
         cfg=ot_loss_cfg,
     )
     total_loss = bc_loss + lambda_ot * ot_loss
     ```

4. **日志与监控**  
   - 将 `ot_loss`、`pi_sum`、`pi_diag` 以及各 term 的 cost 统计追加到训练输出中，以便在 TensorBoard/W&B / log 中监控 OT 对齐情况。

---

## 7. ACT 示例：joints+gripper 为一个特征

下面用 ACT 做示例，说明如何在这个框架下实现与当前“7 关节 + 1 夹爪”一致的 OT 训练。

### 7.1 在 ACTPolicy 中提供 OT 编码接口

```python
class ACTPolicy(PreTrainedPolicy):
    def __init__(self, config: ACTConfig):
        super().__init__(config)
        self.config = config
        self.model = ACT(config)

        # OT 使用策略本身的编码投影
        def encode_feature_for_ot(self, key: str, x: Tensor) -> Tensor | None:
            if key == "observation.state":
                return self.model.encoder_robot_state_input_proj(x)
            if key == "observation.environment_state":
                return self.model.encoder_env_state_input_proj(x)
            return None
```

### 7.2 joints+gripper 的 OT 配置

```python
ot_loss_cfg = OTLossConfig(
    features=[
        OTFeatureSpec(
            src_key="observation.state",
            tgt_key="observation.state",
            dim_slice=slice(0, 8),          # 前 8 维：7 joints + 1 gripper
            use_learned_embed=True,
            embed_name="joints+gripper",
            weight_embed=1.0,
            weight_label=1.0,
        )
    ],
    reg=0.01,
)
```

训练脚本在每个 step 从 `ot_pair` 中取出 `observation.state`，按照上述配置调用 OT loss 模块：

- embedding cost：基于 ACT 自身投影得到的表示；  
- label cost：在 8 维 joint+gripper 空间的 L2 距离；  
- 最终 OT loss 为两者加权和对应的 Sinkhorn OT。

---

## 8. 向后兼容与迁移策略

为避免破坏现有行为，建议逐步迁移：

**移除特化实现**  
   - 项目中所有使用 OT 的配置逐步迁移到 `OTLossConfig`；  
   - 移除 ACT 中硬编码的 `compute_ot_loss`；  
   - 不再支持独立 OTEmbeddingHead；仅通过各 policy 的编码提供 embedding。

---

## 9. 依赖与注意事项

- OT 模块依赖 [POT](https://pythonot.github.io/)（Python Optimal Transport）：
  ```bash
  pip install pot
  ```  
- Sinkhorn 的 reg 值对数值稳定性和对齐形态非常敏感，需要通过实验调参：
  - reg 越小，π 趋向于更“尖锐”的对齐，但数值不稳定风险增加；  
  - reg 越大，对齐更平滑，更像 soft matching。
- 多 term cost 时，注意各 term 的尺度和权重，避免某一 term 统治全部 OT 行为。

---

## 10. 后续扩展点

1. **支持 unbalanced OT**：
   - 在 OTCostConfig 中增加 `tau1`, `tau2` 等参数；  
   - 使用 `ot.unbalanced.sinkhorn_knopp_unbalanced`。

2. **支持 batch 级别的 reg 动态调节**：
   - 根据特征范数或距离统计自动调整 reg。

3. **与 diffusion / VLA / GROOT 等 policy 的整合**：
   - 通过统一的 `OTEmbeddingHead` 和 `OTLossConfig`，让这些策略重用同一套 OT 对齐机制。

---


看了你现在这版配置，结构是对的，可以一次性对齐三路相机。关键点逐条确认如下：

  - 相机键写法
      - 你把 obs_keys 显式列成了三路相机："observation.images.third_person_cam_color", "observation.images.ee_cam_color", "observation.images.side_cam_color"（src/lerobot/scripts/train_config/
        act_fr3_ot_images_embedOnly.json:47）。这会让 OT 数据集把这三路都加载进样本，OK。
      - loss_config.features 里也分别为三路相机各写了一条 feature（:54-72），都是 use_learned_embed=true 且只用 embedding cost，OK。
      - 注意：这些相机名必须与数据集的实际键完全一致，否则会在 OT loss 里报缺键（compute_ot_loss_for_policy 要求 exact key；只有传通配符时才会自动解析，见 src/lerobot/policies/ot_train/ot_loss.py:412）。建议先确认
        dataset.meta.camera_keys 列表。
  - action 监督项
      - action 的 label cost 保留（:74-83），并通过训练循环自动注入到 obs（src/lerobot/scripts/lerobot_train.py:774-783），所以用 "src_key": "action" 是正确的。
      - dim_slice: [0, 8] 前提是你的动作维度前 8 维确为 7 关节+夹爪，请再次确认。
  - 权重与量级
      - 现在三路相机各自 weight_embed=1.0，最终 OT ground cost 会把三路图像项简单相加（每个 term 在组合时默认权重 1.0，见 src/lerobot/policies/ot_train/ot_loss.py:544），整体量级比单路相机会大约×3。
      - 两种平衡方式二选一：
          1. 把三条相机 feature 的 weight_embed 都改成 0.3333；或
          2. 保持 1.0 不变，但把全局 lambda_ot 从 0.1 降到 ~0.033。
      - 这样更容易复用你之前单相机的超参直觉，防止 action 的 label cost 0.02 被图像项完全淹没。
  - reg / tau / heuristic
      - 你设置的是 reg=1.0, tau_src=5.0, tau_tgt=5.0, heuristic=true（:85-88）。这是一组偏“平滑”的超参，合并三路相机后可能更平。建议训练时多看：
          - ot_pi_sum、ot_pi_diag（耦合矩阵是否过于散/尖）
          - 各相机项的 ot_cost/<term_name>（你没写 term_name，会默认用 src_key）
      - 如果 π 太散或对齐不敏感，可小幅下调 reg、tau 或整体减小 lambda_ot。
  - 小提醒（不改也能跑）
      - 我们在 OT 分支里直接把原始图像送入 ACT 的 backbone 做 embedding（ACTPolicy.encode_feature_for_ot），OT 源数据集目前没有套和主数据一致的图像变换与归一化。这不是配置错误，但两侧图像分布可能略有偏移；如果后续
        要更稳，可以在脚本里让 OT 源数据集也复用相同的 ImageTransforms/归一化。

  - W&B 里 ot_pi_sum/ot_pi_diag 合理，三路相机的 ot_cost/<键名> 都有值且量级相近
  - 主 BC 的 l1_loss 没被 OT 完全压制（若被压制，按上面建议调小三路相机权重或 lambda_ot）

  如需我帮你把三路相机的权重统一成 0.3333，并加上 term_name 方便看日志，我可以直接改这份 JSON。你也可以先确认下三路相机的真实 key 名（推荐打印 dataset.meta.camera_keys）。