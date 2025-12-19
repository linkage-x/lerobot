 - OT“核心数学部分”（特征 → ground cost → unbalanced Sinkhorn → <π, M>）在当前实现里已经和 diffusion_policy_ot.py 高度一致，可以通过配置做到几乎 1:1。
  - 差异主要在于：数据管线（我们用 Dsrc/Dtgt + pair_info）、没有实现 heuristics 里的先验矩阵 c、以及 OT cost 构造和 loss 权重通过更通用的 config 表达。

  下面分块对比。

  ———

  1. Ground cost M 的构造

  diffusion_policy_ot.py（OTDiffusionPolicyUNet.train_on_batch）里 OT 部分大致是：

  - 特征：
      - obs_features = encoder(obs, goal) → [B, T, D]
      - ot_src_feat = obs_features[:B_ot_src].view(B_ot_src, -1)
      - ot_tgt_feat = obs_features[B_ot_src:B_ot_src+B_ot_tgt].view(B_ot_tgt, -1)
  - embedding cost：
      - M_embed = torch.cdist(ot_src_feat, ot_tgt_feat) ** 2
  - label：
      - label 类型由 ot_params["label"] 控制：
          - eef_pose → batch["obs"]["robot0_eef_pos"] 切片
          - 否则 → batch["actions"] 的前若干维
      - label 显式 .detach()（不回传）
      - M_label = torch.cdist(src_label, tgt_label) ** 2
  - 最终 ground cost：
      - M = emb_scale * M_embed + cost_scale * M_label

  当前 LeRobot 实现（src/lerobot/policies/ot_train/ot_loss.py）中：

  - 通过 OTFeatureSpec 配置特征来源和切片：
      - src_key, tgt_key, dim_slice（比如 "observation.state", slice(0, 8)）
  - embedding cost：
      - 若 use_learned_embed=True，通过 policy.ot_heads[embed_name] 得到 embedding，再 torch.cdist 平方：
        M_embed = torch.cdist(src_embed, tgt_embed) ** 2
  - label cost：
      - 始终 M_label = torch.cdist(src_slice, tgt_slice) ** 2
      - 是否 detach 留给上游决定（目前 ACT 用的是原始 obs/state，不牵涉 encoder 梯度，效果等价于 detach）。
  - feature 内部组合：
      - M_term = weight_embed * M_embed + weight_label * M_label

  → 映射关系：

  - emb_scale ↔ weight_embed
  - cost_scale ↔ weight_label
  - label 选择：由我们这边的 (src_key, tgt_key, dim_slice) 决定；可以配置成用 eef pose / action / state 等，功能上和 ot_params["label"] 一致，只是表达方式不同。
  - 我们支持多个 feature term（多种特征组合），diffusion_policy_ot 目前只用一个。

  差异：

  - diffusion_policy_ot 强制对 label .detach()，我们默认不 detach，但只要上游传入的是“原始观测/动作”，梯度也不会回传到数据本身，行为等价；如果要严格一致，可以在调用前对 label 特征显式 .detach()。

  ———

  2. Sinkhorn 调用（unbalanced OT）

  diffusion_policy_ot.py:229-235：

  a, b = ot.unif(ot_src_feat.shape[0]), ot.unif(ot_tgt_feat.shape[0])
  pi = ot.unbalanced.sinkhorn_knopp_unbalanced(
      a, b,
      M.detach().cpu().numpy(),
      ot_params["reg"],
      c=c,  # 可选 heuristic 先验
      reg_m=(ot_params["tau1"], ot_params["tau2"]),
  )

  要点：

  - a, b 为均匀边缘分布（ot.unif）；
  - 使用 unbalanced Sinkhorn；
  - reg = ot_params["reg"]，reg_m = (tau1, tau2)；
  - 允许一个额外的先验矩阵 c（当 heuristic=True 时是对角矩阵 / N）。

  当前实现（_compute_sinkhorn_plan，src/lerobot/policies/ot_train/ot_loss.py:143）：

  - marginals：

  a = torch.full((n_src,), 1.0 / n_src, ...)
  b = torch.full((n_tgt,), 1.0 / n_tgt, ...)
  a_np, b_np = ...

  和 ot.unif 等价。

  - balanced / unbalanced 逻辑：

  if tau_src is None and tau_tgt is None:
      # balanced: ot.sinkhorn(..., method="sinkhorn_log", warn=False)
  else:
      unbalanced_mod = getattr(_ot, "unbalanced", None)
      if not hasattr(unbalanced_mod, "sinkhorn_knopp_unbalanced"):
          # fallback: balanced
          ...
      else:
          if tau_src is not None and tau_tgt is not None:
              reg_m = (float(tau_src), float(tau_tgt))
          else:
              reg_m = float(tau_src or tau_tgt)
          pi_np = unbalanced_mod.sinkhorn_knopp_unbalanced(
              a_np, b_np, cost_np, reg, reg_m,
          )

  - 调用前使用 cost.detach().cpu().numpy() 与 diffusion_policy_ot 的 M.detach().cpu().numpy() 一致；
  - 调用后如果 pi_np 中存在 NaN/Inf，则回退为均匀耦合 1/(n_src * n_tgt) 保证稳定性。

  映射关系：

  - loss_config.reg ↔ ot_params["reg"]；
  - loss_config.tau_src, loss_config.tau_tgt ↔ ot_params["tau1"], ot_params["tau2"]；
  - marginals a,b 全部是均匀分布，和 ot.unif 一致。

  关键差异：

  - 我们当前没有暴露 heuristics 里的先验矩阵 c：
      - diffusion_policy_ot 在 heuristic=True 时用一个对角优先矩阵 c；
      - 我们的 _compute_sinkhorn_plan 目前调用 sinkhorn_knopp_unbalanced 时不传 c，行为相当于 heuristic=False 的情况。
      - 如果你需要这个先验，可以后续在 OTLossConfig 里加一个类似 heuristic_diag_prior 的开关，再在 _compute_sinkhorn_plan 里构造 c 并传给 POT。

  ———

  3. OT loss 与 BC loss 的组合

  diffusion_policy_ot.py:239-265：

  - OT loss:

    ot_loss = torch.sum(pi * M)
  - BC（实际上是 diffusion 的 L2）：

    l2_loss = F.mse_loss(noise_pred, noise)
    loss = l2_loss + ot_params["scale"] * ot_loss

  当前实现（src/lerobot/scripts/lerobot_train.py:110-129）：

  - 在 update_policy 里：

  loss, output_dict = policy.forward(batch)  # BC / main loss

  if lambda_ot != 0.0 and ot_src_obs and ot_tgt_obs and ot_loss_cfg:
      ot_loss, ot_metrics = compute_ot_loss_for_policy(...)
      loss = loss + float(lambda_ot) * ot_loss

  映射关系：

  - ot_params["scale"] ↔ cfg.ot.lambda_ot；
  - OT loss 定义完全一样：ot_loss = (pi * M).sum()（ot_loss.py:283-284）。

  差异：

  - diffusion_policy_ot 的“BC”是 diffusion 的噪声预测 L2 loss；我们的是 ACT 的 BC loss（行为克隆）；
  - 形式上都是 main_loss + λ·OT_loss，一致。

  ———

  4. 数据管线 / 批次拼接方式

  robomimic + ot-sim2real（ot-sim2real/robomimic/utils/train_utils.py:865）：

  - 有 bc_dataloader 和 ot_dataloader（后者来自 pair_info）；
  - 每步：
      - 从 ot_dataloader 取出 ot_batch = {"src": ..., "tgt": ...}；
      - B_ot = src_ot_batch["actions"].shape[0]；
      - ot_batch = concat_two_batch(src_ot_batch, tgt_ot_batch)；
      - 再与 bc_batch concat 成一个大 batch；
      - 传给 model.train_on_batch(input_batch, B_ot, ot_params, ...)，由模型内部把前 B_ot_src, B_ot_tgt 做 OT，剩下 BC。

  LeRobot 这边有两种：

  1）src/lerobot/policies/ot_train/ot_training.py:86-118

  - 完全是 robomimic 版 run_epoch_for_ot_policy 的泛化版本：
      - 同样的 OT / BC dataloader；
      - 同样的 concat+传给 train_on_batch(full, b_ot, ot_params, ...)；
  - 如果你在 LeRobot 里实现一个 robomimic 风格的 OT policy，这个函数能 1:1 对齐 diffusion_policy_ot 的训练入口。

  2）通用 ACT 训练脚本 src/lerobot/scripts/lerobot_train.py:640-777

  - 我们在默认 offline 训练 loop 里内嵌了 OT：
      - 主 dataloader 只来自目标数据集（BC）；
      - 额外开一个 OT dataloader（source+target+pair_info）；
      - 每步取一个 BC batch + 一个 OT pair batch：
          - OT batch 只用于构造 (ot_src_obs, ot_tgt_obs) 传给 compute_ot_loss_for_policy；
          - BC batch 走原来的 ACT BC pipeline；
      - 最终 total_loss = bc_loss + lambda_ot * ot_loss。

  相对 diffusion_policy_ot：

  - 思路一致：OT batch + BC batch 组合训练；
  - 我们把 OT 部分抽到一个 policy 无关的模块里，并通过 config 指定特征来源；
  - ACT 的例子是“跨两个 LeRobotDataset 做 sim2real OT”，而不是在同一个 robomimic batch 里切前 B_ot / 后 B_ot。

  ———

  5. 小结：一致 / 不一致点

  高度一致的部分

  - OT cost 形式：
      - M 由 encoder embedding cost + label cost 线性组合而成；
      - 使用 torch.cdist(..., ...) ** 2。
  - OT 求解：
      - 使用 POT；
      - 使用 M.detach().cpu().numpy() 做正向，梯度只通过 M 回传；
      - 支持 unbalanced Sinkhorn，reg / reg_m 结构与 reg, tau1, tau2 完全对应；
      - 边缘分布采用均匀分布 a,b。
  - OT loss：
      - ot_loss = (pi * M).sum()；
      - 和 BC/main loss 线性组合：main + λ·OT。

  有差异 / 尚未覆盖的部分

  1. 先验矩阵 c（heuristic）
      - diffusion_policy_ot 支持一个启发式对角先验（heuristic=True 时）；
      - 当前 LeRobot 实现没有 c 参数，等价于永远 heuristic=False。
  2. 特征来源与 label 类型（已对齐 robomimic）
      - 现已统一为“直接使用各 policy 的 obs encoder 输出”（如 ACT 的输入投影、Diffusion 的 rgb_encoder）。
      - 不再使用 OTEmbeddingHead 回退；若策略未提供可用编码，将在训练时报错提示实现 encode_feature_for_ot。
      - label 的种类在我们这边是通过 src_key/tgt_key 决定的，而不是 ot_params["label"] 的字符串枚举；
      - 目前 ACT 示例只配置了 joints+gripper 的 state OT，没有做 eef_pose / 多特征组合，但框架支持。
  3. detach 行为
      - robomimic 版对 label 显式 .detach()；
      - 我们保留梯度，靠“传入的是什么张量”决定是否有梯度（对于原始 obs/动作，差别不大）。
  4. 默认是否 unbalanced
      - robomimic 版本默认就是 unbalanced（配置了 tau1/tau2）；
      - 我们默认 balanced（tau_src/tau_tgt=None），需要在 loss_config 里显式设置 tau_src/tau_tgt 才会变成 unbalanced。

  5. window_size的理解，ot-sim2real/robomimic/scripts/ot_sim2real/hyperparam_ot.py的L226中其取值40,20,5,而你之前曾经说过window_size一般选0～2, 分析一下哪个是对的，为什么。


  总结一下：

  - 如果你在 LeRobot 的 loss_config 中设置：
      - 特征选取与权重与 ot_params 对应；
      - reg, tau_src, tau_tgt 对齐 reg, tau1, tau2；
  - 那么 OT 部分的数学行为（C 的构造、unbalanced Sinkhorn 求解、OT loss 定义）与 ot-sim2real/robomimic/algo/diffusion_policy_ot.py 的实现是一致的，
    除了暂时不支持的 heuristic 先验矩阵 c 这一点。
