# 通用 OT Loss 设计方案（LeRobot）

本文档总结 LeRobot 中 OT（Optimal Transport）损失的统一实现与训练对接方式，并对齐 ot-sim2real/robomimic 的实现要点。

目标：

- 模块化、可配置、可复用（跨策略）；
- 支持多特征组合与可学习/标签两类 cost；
- 与 ot-sim2real 的不平衡 Sinkhorn 求解保持一致；
- 在跨 DataLoader 混合场景下提供工程性对齐补丁并给出约束与建议。

---

## 1. 总体设计

- 仅使用 POT 的不平衡 Sinkhorn（`ot.unbalanced.sinkhorn_knopp_unbalanced`）求解 π；
- OT loss 定义：`ot_loss = Σ πᵢⱼ Mᵢⱼ`；
- 嵌入（embedding）来自策略自身编码：优先调用 `policy.encode_feature_for_ot`；
- 标签（label）在原始空间计算欧式距离；
- 对 embedding 与 label 的输入，统一在 batch 维之外展平（例如 `(B,S,D) → (B,S*D)`），再用 `torch.cdist(..., ...)**2`。

---

## 2. 代价与求解模块（ot_loss.py）

特征 → term → 总代价矩阵 M → 不平衡 Sinkhorn → OT Loss 与指标。

- OTFeatureSpec（单个特征项）
  - `src_key` / `tgt_key`：特征来源键（如 `observation.state`、`action`、`observation.images.xxx`）。
  - `dim_slice`：在最后一维的切片（如 `slice(0,8)` 选取 joints+gripper 子空间）。
  - `use_learned_embed` / `embed_name`：是否使用策略编码作为 embedding cost。
  - `weight_embed` / `weight_label`：term 内部两种 cost 的线性权重，形成 `M_term = w_e*M_embed + w_l*M_label`。
  - `term_name`：日志中该 term 的名字；
  - `term_weight`：term 级“外层”权重，用于多 term 的总成本组合 `M = Σ term_weight_i * M_term_i`。

- OTLossConfig（多特征组合）
  - `features: list[OTFeatureSpec]`；
  - `reg`：Sinkhorn 正则； epsilon
  - `tau_src` / `tau_tgt`：不平衡 OT 边缘松弛；任一不为空则使用不平衡；tau1/tau2
  - `heuristic`：可选对角先验（仅不平衡 OT 生效）。

- 求解
  - `M` 在 torch 中组合；
  - 使用 POT 在 numpy 上求解 π（输入 `M.detach().cpu().numpy()`）；
  - 若数值不稳定（NaN/Inf），回退为均匀耦合；
  - 返回指标：`ot_loss`、`ot_pi_sum`、（若方阵）`ot_pi_diag`、以及 per-term `ot_cost/<term_name>`。

---

## 3. 时间维与数值细节（对齐 ot-sim2real）

- 对 embedding 与 label 的输入，统一展平时间维（batch 外 flatten）：
  - `(B,S,D)` → `(B,S*D)`；图像类 `(B,S,C,H,W)` 视作 `(B,S,C*H*W)`；
  - 这样与 robomimic 的 `view(B, -1)` 一致，避免“只取首/末帧”的语义偏差；
- label 在原始空间 detach 后计算（我们默认传入的是原始观测/动作张量，等价于无梯度来源）。

---

## 4. 训练循环与数据管线对接

标准集成：

- 每步 `total = bc_loss + lambda_ot * ot_loss`； # lambda_ot论文中使用 0.1
- 从 `ot_dataloader` 获取 `ot_pair`，直接使用 raw obs；需要时将 `action` 注入 obs 以复用 `src_key/tgt_key == "action"` 的配置；
- 源 OT 数据集建议复用目标数据集的图像变换/统计（例如 ImageNet 统计）。

跨 DataLoader 混合（工程性补丁）：

- 仅对矢量/矩阵类键进行 2D/3D 的 rank 对齐（白名单：`action`、`observation.state`、`observation.environment_state`），不对图像键做隐式重复；
- 对缺失的 `action_is_pad` 自动用 False（零）填充；
- 若设备不一致（CPU/GPU），优先将 CPU 搬到非 CPU 设备；
- BC 源数据集复用目标数据集的 `delta_timestamps`（由 policy 的 `*delta_indices` 推导），以统一时间窗与 `*_is_pad` 掩码；
- 首次发生 device/rank/掩码对齐时输出有限条（≤8）debug 日志，便于排查。

若条件允许，推荐与 robomimic 相同的“单 DataLoader、单批次内切分”方案，可避免上述工程性补丁。

---

## 5. 多特征加权建议

- `weight_embed / weight_label`：控制“term 内部”嵌入与标签的配比；
- `term_weight`：控制“多 term 之间”的外层配比；
- 例如三路相机各占 1/3：

```json
{
  "src_key": "observation.images.ee_cam_color",
  "tgt_key": "observation.images.ee_cam_color",
  "use_learned_embed": true,
  "weight_embed": 1.0,
  "weight_label": 0.0,
  "term_weight": 0.3333,
  "term_name": "img_ee"
}
```

动作 label 可单独一个 term（如 `dim_slice=[0,8]` 对 joints+gripper），设置较小 `weight_label` 与 `term_weight`，与图像项平衡。

---

## 6. 依赖与参数提示

- 依赖 [POT](https://pythonot.github.io/)；
- 仅支持不平衡 Sinkhorn；`heuristic=true` 时加入对角先验矩阵 c；
- `reg`、`tau_src/tau_tgt` 对行为影响显著，需要网格化调参；
- 监控：`ot_pi_sum`、`ot_pi_diag`、`ot_cost/<term>` 的量级与变化趋势应与任务期望一致。

---

## 7. 迁移与扩展

- 可在未来引入 `time_reduce` 配置（first/last/flatten），当前默认 flatten 与 ot-sim2real 对齐；
- 可在跨 DataLoader 混合场景下，增加更细粒度的键级对齐/丢弃策略与告警。
