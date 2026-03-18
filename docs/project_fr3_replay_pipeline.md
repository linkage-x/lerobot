## 根因分析结论（2026-03-18）

### SLAM 帧定义（已确认）
- DAS SDK 输出 T(W_s, base_link_V4)，NOT T(W_s, link_imu)
- 判据：state[0] ≈ Ry(15°)（干净Y轴旋转），与 joint_imu rpy=[π/2,0,-π/2] 不兼容
- W_s = 重力对齐世界系（Z=up），origin = 设备首帧位置

### 硬件关系（新厘清）
- DAS_Gripper_V4：数采设备，有 link_imu（SLAM 传感器），base_link = SLAM 输出帧
- DAS_Controller_V3：实际末端执行器，无 link_imu，夹爪与V4同构
- repo fr3_das_ati.urdf：建模 V3，gripper_base_link ≈ V4 base_link（同一安装接口）

### +Z offset 和 15° TCP 旋转根因
- 现有数据录制时物理安装倾角 ~15°Y → state[0] ≈ Ry(15°)
- 导致 R(B,W_s) = Ry(-15°)
- Ry(-15°) 使 X_Ws 在 B 系有 +0.259 Z 分量 → +Z 耦合
- 同时使所有 EE 旋转旋转 15° → TCP 方向误差
- transform chain 计算本身正确，误差来自数据录制时的物理倾斜

### 当前修复状态
- ati_das_joint rpy 已改为 "0 0 0"（URDF 模型修正）
- T_IE 值未变（fr3_hand_tcp_joint 未变）：R=[[0,0,1],[0,-1,0],[1,0,0]], t=[0.13,0,-0.04]
- 现有数据的 +Z offset 和 15° TCP 旋转无法通过代码修复

### 下一步方向
- 原则：不改数据
- 方案：基于 V4 link_imu 重新推导 T_IE，即
    T_IE_new = T(link_imu_V4, das_gripper_ee)
            = inv(joint_imu_V4) @ fr3_hand_tcp_joint
    joint_imu_V4: xyz=[0.002742, 0.0075232, -0.049069], rpy=[π/2, 0, -π/2]
    → R(link_imu, base_link) = [[0,-1,0],[0,0,1],[-1,0,0]]
    → R_IE_new = [[0,1,0],[1,0,0],[0,0,-1]]
    → t_IE_new = [0.0075, 0.0091, -0.1273]（待精确计算）
- 必要时修改 fr3_das_ati.urdf 添加 link_imu 节点
- 需要验证 R_reset @ R_IE_new^T 是否合理


要验证的关键数学

R_reset @ R_IE_new^T：

- R_reset = [[0,0,1],[0,-1,0],[1,0,0]]（来自 reset quat [0.7071,0,0.7071,0]）
- R_IE_new^T = [[0,1,0],[1,0,0],[0,0,-1]]（对称矩阵）
- 乘积 = [[0,0,-1],[-1,0,0],[0,1,0]] ≠ I

这意味着如果 SLAM 确实输出 T(W_s, link_imu)，则 R(B,W_s) ≠ Ry(-15°)，而是上面这个矩阵乘以 state[0]^T。此结论与 state[0]≈Ry(15°)=T(W_s,base_link) 假设相矛盾，需要用实验确认 SLAM 输出帧到底是 base_link
    还是 link_imu，或者查阅 DAS SDK 文档。