# BOX / Franka 标定、模型与重播交接

交接日期：2026-09-09。范围：当前 P0 单臂、BOX 2.0、20260903_193725 两条数据。本文用于接续调试，不作为整套系统已验收的结论。9 月 9 日已只读核对 Thor 当前入口及日志目录，最新运行仍为 9 月 8 日 17:14 启动的记录；本次交接未操作机械臂或夹爪。

## 一、当前状态与接手重点

目前已完成 P0 参考修正、原视频重新解算、991 帧 IK、MuJoCo 运动学检查及原生 panda-py 只动臂入口部署。两条数据均通过固定 88 mm 夹爪开口下的离线分段检查。最近一次真机执行了 Episode 0 约 44 秒重播，随后收到 KeyboardInterrupt，未完成整条；不能将本次记录写成两条全程真机验证通过。

接手时应继续使用独立目录 `replay_p0_native_arm_only_20260908`，不要回到旧 P0 模型、旧 IK 或自定义慢速播放器。夹爪暂不接入控制，真实夹爪必须空夹并能保持声明的开口。软件跟踪误差已达到毫米量级，但标定绝对精度尚未通过独立测量确认，尤其右侧 BOX 的 Cube→TCP 参数仍是从左侧复制的。

建议接手顺序：先核对文件版本及现场是否变化，再完成独立物理位置核验和剩余轨迹验证；夹爪网络及同步另行处理，不与机械臂控制器调试混在一起。

## 二、设备、代码与数据位置

| 对象 | 地址或路径 | 说明 |
|---|---|---|
| Thor | `ssh nvidia@192.168.111.122` | 主机名 upai-pro03，当前真机重播主机 |
| Franka FCI | `192.168.11.102` | Thor 侧地址 192.168.11.100；使用已核对的 FR3 / FCI protocol 9 限位 |
| 历史标定主机 | `corenetic@192.168.111.45` | 历史资料源，曾需经 Thor 中转；本次未复测该主机连通性 |
| 本地仓库 | `/Users/yanzeyu/codes/lerobot` | 下文“本地相对路径”均相对此目录 |
| Thor 仓库 | `/home/nvidia/lerobot` | 原数据与标定输入 |
| 当前数据集 | `/home/nvidia/lerobot/outputs/datasets/thor_gmsl2_10ch_v1_20260903_193725` | 单臂两条，不是双臂数据；Episode 0：571 帧，Episode 1：420 帧 |
| 采集 BOX | `1819152274`，数据字段 right | right 是采集设备标签，不表示当前控制的是双臂右机械臂 |
| 真机夹爪 BOX | `596523097` | 与采集 BOX 不是同一设备；当前入口不连接它 |
| Python | `/home/nvidia/Code/infer/.venv-fr3/bin/python` | 依赖路径由启动脚本设置；不要随意换解释器或 panda-py |

交接开发快照源自主仓库 `box` 的 `4b7fb4be9715047524baea9c91a5ef02f8beade2`，以及 opencv_kalibr 固定提交 `4a0f46c159369dbf3b97ded3dcf21ea23ce1dbc3`。随后为两仓库建立 `yzy/error_study` 保存本地修改；发布分支的提交以 Git 记录为准，不再把上述基线称为当前 HEAD。这些基线不等于标定快照来源。有效交接资产包含 outputs、独立脚本和编译产物，不能只 git clone 后直接复现；代码获取及未上传资产见 `docs/replay_handoff/README.md`。

本文件不包含登录密码。账号、密钥及现场操作权限由负责人另行交接。

## 三、P0 标定与坐标约定

### 3.1 当前使用的参考

P0 是第一次摆放位置的标签，不是“world 原点”，也不应设为单位矩阵。用户确认底座及固定相机在该位置标定后未移动，辅助 Marker 板与底座刚性连接，纸面与底座安装面基本齐平。后续若任一安装关系变化，需重新确认标定适用性。

早期误用了 2026-05-13 的 Hikon 辅助 Marker 参考，其 Marker 高度约为 −169～−141 mm，与现场不符。后来从下列目录找回 2026-05-26 单相机 hk_07 完整板参考：

```text
/home/corenetic/Code/lerobot/outputs/calibration/folded_0720/
  hikon_auxiliary_aruco_markers_in_base_single_hk07/summary.json
```

该参考使用完整识别 ID 20–29 的 412 帧，Marker 中心高度约 −4.87～+3.72 mm。目录中的 `0720` 不是这份结果的实际标定日期，不宜仅凭目录名选择版本。

当前 Thor 观测仍采用自身配套相机参数，不使用 Hikon 内参处理森云图像：

- 内参：`/home/nvidia/lerobot/outputs/calibration/thor_gmsl2_selfcal_0804_fisheye_intrinsics`。
- 外参：`/home/nvidia/lerobot/outputs/calibration/calib_20260902_103833_extrinsics`。
- P0 定位视频：`thor_gmsl2_10ch_v1_20260826_162513`。
- P0 采用 cam_13 完整板观测，以十个 Marker 中心做刚体配准；cam_14 只作独立交叉检查，没有参与平均。两者每台抽样 121 帧均完整识别十个 Marker。

P0 定位“单相机”与后续轨迹解算“七相机融合”是两个不同环节，不能混淆。

### 3.2 当前 P0 变换

约定 `T_A_B` 将 B 坐标下的点变换到 A 坐标：`p_A = T_A_B @ p_B`。当前 `T_world_base` 如下，平移单位 m；此处仅展示截断值，程序应读取完整精度 JSON / URDF。

```text
[[ 0.999960456, -0.002388449, -0.008566317,  0.006125456],
 [ 0.002188473,  0.999726626, -0.023278406, -0.000414923],
 [ 0.008619575,  0.023258738,  0.999692320, -0.012019270],
 [ 0.000000000,  0.000000000,  0.000000000,  1.000000000]]
```

目标从世界转到底座：`T_base_target = inv(T_world_base) @ T_world_target`。
旧 P0 平移约 `[−71.88, +14.65, +145.78] mm` 已不用于当前重播。新旧数值变化是参考及配套外参修正，不是机械臂实际移动了约 158 mm。

cam_13 十点配准 RMS 7.49 mm、最大残差 11.76 mm；与 cam_14 的 base 原点差 3.42 mm、朝向差 1.36°。该朝向差在 0.6 m 工作距离可产生约 14 mm 的位置差量级，仍需独立核验。逐帧稳定不等于绝对准确。

证据及完整精度输入位于本地：

```text
outputs/recalculation_20260907_193725/coordinate_chain_audit_20260907/
  恢复连接后的标定复核.md
  p0_recovered_hk07_candidate.json
  recover_p0_from_single_hk07.py
  inputs/corenetic_recovered/
```

## 四、URDF 与 Cube→TCP

### 4.1 使用哪份模型

| 用途 | 文件 |
|---|---|
| 本地修正 P0 模型 | `outputs/p0_recovery_replay_20260907/model/fr3_v2_p0_recovered.urdf` |
| Thor 路径适配模型 | 同目录 `fr3_v2_p0_recovered.thor.urdf` |
| 当前重播使用的打包模型 | `/home/nvidia/box_api/replay_p0_arm_only_20260908/model.urdf`，配套同目录 meshes |
| 当前模型另一个已固定副本 | `/home/nvidia/box_api/replay_p0_once_20260908/model.urdf` |
| 历史双臂 V2 | `src/lerobot/robots/franka_research3/assets/franka_fr3/dual_fr3_corenetic_gripper_v2_p1_p2.urdf` |

当前模型为 FR3 机械臂加 BOX 2.0 夹爪，不是只有夹爪。P0 修正时只改变 world_to_base 固定变换和资源路径，保留其余关节链、惯量、碰撞几何和限位。固定 TCP 帧名为 `corenetic_gripper_ee`，底座帧名 `base`，七个主动关节 `fr3_joint1`～`fr3_joint7`。

不要因文件名含 p0 就改用仓库 assets 中较早的 `fr3_corenetic_gripper_v2_p0.urdf` 或 `exports/fr3_p0_mujoco_20260831`；它们不是当前修正模型。P1/P2 双臂模型也未在此次修正 P0 流程中重新标定或全程验证，只作为历史资产保留。

控制器固定工具配置参考：质量约 0.749596417 kg，质心 `[0.009645203, -0.007980638, 0.066584057] m`；法兰到固定 TCP 为 Rz(−45°)、Z 平移约 0.171990007 m。惯量及完整矩阵实际读取 `/home/nvidia/box_api/replay_p0_arm_only_20260908/tool_reference.json`；新入口目录本身没有该文件。不应人工抄小数覆盖控制器。读取 libfranka 的 16 元素矩阵需按列主序还原。

### 4.2 Cube→闭合 TCP 来源和限制

使用 `hph/error_study` 对应 opencv_kalibr 提交 `6b4f9e7aef458b71b318909b074646c5ea33545a` 中的 `marker_to_tcp_calibration_20260825.json` 快照，当前有效快照：

```text
outputs/p0_recovery_replay_20260907/source/hikon_cube_tracking_offline/
  config_thor/marker_to_tcp_calibration_20260825.json
```

文件的 `T_cube_tcp` 对应 `link_lt_gripper_tcp` 闭合 TCP，当前 right 与 left 数值相同：

```text
[[ 0, -1,  0,  0.00006347498811817864],
 [ 0,  0,  1, -0.22384833424248549   ],
 [-1,  0,  0,  0.10505357346316048   ],
 [ 0,  0,  0,  1                   ]]
```

追踪到 replay 坐标约定需再右乘固定 `Rx(π)`，即 `link_lt_gripper_tcp → corenetic_gripper_ee`。漏乘或重复乘都会导致朝向错误。此次已通过 `closed_tcp_frame_hop.urdf` 明确该变换。

必须保留的限制：标定文件 `validated=false`；左侧平移来自 pivot 实测，旋转继承旧结果；右侧 BOX 1819152274 的平移是左侧复制值，不是右侧独立实测。文件还明确记录旋转歧义及拆装重复性未完成。用户要求复用该结果用于继续解算，并不等于这些待验证项已经关闭。

### 4.3 BOX 2.0 接触 TCP

实际接触中心为两个 link_contact 的中点。根据用户提供的几何公式：

```python
# d、z 单位均为 mm；d 是两指尖真实净距离，闭合时为 0
z = 49.699345 - sqrt(49.699345**2 - 5.474953*d - d**2/4)

# replay 固定 TCP 坐标约定下，沿局部 -Z 退缩
p_contact = p_fixed - R_fixed[:, 2] * (z / 1000)
```

完整链为 `T_world_contact = T_world_cube @ T_cube_tcp @ Rx(pi) @ Trans(0,0,-z_m)`。机械臂侧 FK 使用相同接触点定义，不给固定控制器 TCP 再重复叠加偏移。

当前追踪配置 `tracking_0902.yaml` 的 `contact_tcp.enable=false` 是有意的：视觉阶段输出固定 TCP，退缩公式由后续接触 IK 阶段应用一次。YAML 中禁用段落仍有旧常数，不能据此认定当前采用了旧公式，也不能直接启用它。

公式参考代码：`tools/fr3/recalculate_contact_tcp_20260907.py`；输入检查：`tools/fr3/replay_ik_trajectory_guarded.py`。前者属于较早一轮复算入口，不可直接不改路径重跑覆盖当前修正 P0 结果。当前结果来源以 `outputs/p0_recovery_replay_20260907/ik_manifest.json` 为准。

宽度假设为记录值就是真实指尖净距离，未使用旧 V1 垫块换算。公式定义域上限约 89.05 mm；越界应报错，不应截断为合法宽度。固定开口的只动臂重播不能重现原采集的开合及接触中心运动。

## 五、数据处理、仿真及碰撞现状

当前两条数据从原视频使用 0804 内参、0902 外参重新生成，七台相机为 cam_06、07、08、09、12、13、14。经过逐帧多相机角点优化和离线平滑，991 帧均成功，重投影误差中位数约 1.122 px、P95 1.593 px。夹爪宽度按 episode、帧号和时间戳严格匹配原 right BOX 表，只复用宽度，不复用旧位姿。

两条 IK 分别 571/571、420/420 成功。求解使用 MuJoCo FK 与 scipy least_squares，逐帧连续初值，每个 episode 独立重置；没有通过移动目标位置解决 IK。IK 本身没有碰撞约束，后续另做几何检查。

本地主要成果在 `outputs/p0_recovery_replay_20260907/`：`tracking_0902.yaml`、`tracking_summary.json`、`tracking_sidecar/`、`ik_manifest.json`、`contact_ik.right.csv`、`model/`、`episode_0_recovered_p0.mp4`、`episode_1_recovered_p0.mp4`。视频为 0.5 倍速度的运动学可视化，两侧画面是同一单臂的两个视角，不是双臂，也不证明真实力控跟踪或碰撞安全。

现场模型采用桌面高于底座安装面 110 mm、桌面位于 +X 侧。根据此前测量和建模，桌边 X=0.1336 m；用户测的是底座 +X 前外缘到桌边净距 62 mm。交接后若需重新测量，应同时复核底座坐标轴及该外缘到原点的几何换算。当前近似桌体为 X≥0.1336 m、Z≤0.110 m，Y 方向无限延伸，并检查 5 mm 扩展场景。

碰撞检查包括 21 个几何体、非排除连杆对和夹爪网格。两个内部铰链接触采用半径 6.7 mm、轴向 ±10.1 mm 的局部允许区域，不是整对忽略；刚性/直接父子对的排除仍需装配复核。未覆盖未建模线缆、负载、另一机械臂及现场其他障碍物。离散检查通过不等于连续碰撞、停止距离或全环境安全认证。

## 六、当前重播入口及依赖

主入口：`/home/nvidia/box_api/replay_p0_native_arm_only_20260908/run_native_arm.sh`。
源码本地副本：`tools/fr3/native_arm_only_20260908/`，含 README、native_arm.py、原生绑定、离线测试、XYZ 分析程序。

控制器为旧六条入口使用的原生 C++ JointTrajectory：速度因子 0.01、到首帧 0.03、每段最多 200 帧、圆角偏差参数 0.02 rad；刚度 `[300,300,300,300,120,80,30]`，阻尼 `[25,25,25,25,10,8,5]`。控制律与规划器源码比对一致，增加绑定以执行检查过的原生轨迹对象，并禁止自动故障恢复。不是自定义 Python OTG，也不是按 20 Hz 直接输出每帧关节角。

名义规划时长 Episode 0 / 1 约 106.25 / 95.31 秒，不含首帧移动、检查、分段停顿和重新规划；不是采集的 9.5 / 6.98 秒原速。旧五次多项式 181 / 131 秒版本已不作为当前入口。原生圆角会改变点间路径，不能称为严格逐点逐时刻零偏差重播。

入口需要下列目录同时存在，不能只拷贝一个 shell 文件：

```text
/home/nvidia/box_api/replay_p0_native_arm_only_20260908/  当前控制入口及隔离库
/home/nvidia/box_api/replay_p0_arm_only_20260908/         模型、场景、检查工具
/home/nvidia/box_api/replay_p0_once_20260908/             固定的数据节点及依赖
/home/nvidia/box_api/replay_p0_native_reuse_20260908/     原生规划独立导数审计库
```

保留 `python/`、`native/`、`build/`、模型 meshes、manifest 和外部 libfranka/Python 环境。AArch64 编译库不能复制到 x86 直接运行。FR3 第六关节轨迹约 3.77～4.03 rad，不能换成仍使用旧 Panda 3.7525 rad 边界的库，也不能简单放宽错误机型的限位来通过。

以下命令不连接硬件，只做离线检查；88 必须按实际固定开口修改：

```bash
bash /home/nvidia/box_api/replay_p0_native_arm_only_20260908/run_native_arm.sh \
  0 --gripper-width-mm 88
```

现场交接完成、夹爪空夹且能保持声明开口、范围清空、急停可用后，由操作者执行真机命令。该命令会移动到首帧并继续当前 episode，不是只读命令：

```bash
sudo bash /home/nvidia/box_api/replay_p0_native_arm_only_20260908/run_native_arm.sh \
  0 --gripper-width-mm 88 --execute
```

输入一次 YES。Episode 1 将 0 改为 1。程序保留网络、模型、工具配置、起步及分段路径检查，检查需要时间；异常停止，不自动复位或重试。没有修改实时内核或系统调度。不要绕过检查，亦不要把断网作为夹爪机械锁定。

当前 Python 入口的 TCP 误差判定在各段结束后执行，不是旧自定义播放器的连续超限停止逻辑。日志采样、终点检查与原生/硬件保护分别起不同作用，不能表述为两种播放器的保护完全相同。

版本核对：当前 timed_plan.json SHA256 为 `7fa69ec0c215de460095a0707d01fc93c850de5474b45d70296db21b6aa4a335`；打包 model.urdf SHA256 为 `2e10116497fac7ea8fe4f4217d9622a62f424ec32c1eb5aed088060e43e78535`。本地路径适配 URDF 哈希可能不同，不能仅凭哈希不同认定几何不同，应核对固定变换与资源路径。

不要继续使用 `run_franka_native_replay_20260903_193725.sh` 的旧 IK；该历史入口曾因初始位置错误被阻断。未完成的到位修复已归档到 `replay_p0_arm_settle_fix_20260908.abandoned`，未作为当前控制器使用。

## 七、最近真机记录与误差

当前最新日志：

```text
/home/nvidia/box_api/replay_p0_native_arm_only_20260908/logs/
  20260908T091408Z_3e4df280/
```

start 完成约 29.75 秒；replay_0 完成约 30.00 秒；replay_1 运行约 14.23 秒后记录 KeyboardInterrupt，replay_2 未执行。不能仅据此判断具体人工中断原因，也不能写成此次由碰撞或通信故障导致停止。

已执行重播部分共 2205 个约 50 Hz 样本，以下不含移动到首帧。误差为“控制器固定 TCP 反馈 − 同时刻原生规划固定 TCP”，坐标轴为 Franka base，单位 mm。

| 方向 | 平均绝对误差 | RMS | 最大绝对误差 |
|---|---:|---:|---:|
| X | 0.90 | 1.00 | 2.29 |
| Y | 1.27 | 1.56 | 3.64 |
| Z | 1.14 | 1.38 | 3.74 |

三维距离 RMS 2.31 mm、最大 3.94 mm。移动到首帧的终点同模型 FK 误差约 4.86 mm；第一重播段终点约 1.87 mm。控制器反馈依赖自身模型与配置，这些数据不是外部实测精度，也不包括新标定相对真实任务目标的系统误差。

每个运行分段保存 `native_plan.npz`、`audit.json`、`before.json`、`telemetry.npz`、`execution.json`。telemetry 列为：时间 1、q 7、dq 7、O_T_EE 16（列主序）、通信成功率 1、外力矩 7，共 39 列；记录频率约 50 Hz，不是控制频率。应按控制器时间对齐该段实际规划，不能用原始采集时间或名义其他段的计划直接相减。

本地日志及分析：`outputs/p0_recovery_replay_20260908/native_arm_only/20260908T091408Z_3e4df280/`，其中 `xyz_error_report.md/json` 可直接查看。分析脚本为 `tools/fr3/native_arm_only_20260908/analyze_xyz.py`。当前本地环境缺 matplotlib，JSON/Markdown 已生成，不将未生成的曲线图列为交付件。

## 八、未关闭事项及下一步

| 事项 | 当前依据 | 后续工作 / 验收要求 |
|---|---|---|
| P0 绝对定位 | 跨相机差约 3.42 mm / 1.36° | 用独立观测或量具核验多个工作区位置及朝向；不要用参与拟合的同一相机叠图替代独立验证 |
| 右 BOX Cube→TCP | 0825 文件为左侧复制，validated=false | 确认装配一致性，优先完成右 BOX 独立平移及朝向验证；公式不能补偿固定安装标定错误 |
| 真机全程 | Episode 0 部分完成 | 查明此次人工中断原因，现场复核后验证剩余段及 Episode 1，保留完整日志；不自动断点续跑 |
| 夹爪通信 | 曾有 discovery 重复、地址变化、反馈超时 | 先修复网络并单独测试，不直接恢复联合重播 |
| 夹爪宽度与同步 | 数据存在，但当前没有控制/反馈 | 核对读数与真实净距，确认模式稳定及新反馈时间戳，再接入实际原生轨迹时间/进度；不按总时长简单均分 |
| 碰撞与停止能力 | 当前有限场景离散检查通过 | 核实桌边、支架、线缆与负载，复核铰链允许区域和停止距离；双臂需另行完整验证 |

关于夹爪网络，9 月 8 日抓包已发现设备 MAC `00:80:e1:8e:38:59` 申请地址时，选择 Server-ID 192.168.100.1，却收到 192.168.1.1 的 DHCP NACK（wrong server-ID）。这是取址冲突的重要证据，不足以解释此前所有 200 ms 反馈超时。应由网络负责人核对交换机上联和 DHCP 服务，协调固定租约后，再连续验证指定 BOX 的新鲜反馈。历史 IP 192.168.1.119 / 192.168.2.134 不能视为当前地址。

本次没有重新检测夹爪通信，不将“不连接夹爪后能动臂”表述为夹爪问题已经修复。详细证据见 `outputs/p0_recovery_replay_20260908/gripper_network_diagnosis_20260908_1613.md`。

交接原则：保留原始数据、标定快照、旧输出及失败日志；新一轮采用新目录并记录输入哈希。任何标定、模型或时间策略变化都需重新解算或审计相应路径，不能仅修改命名或移除失败判定继续执行。
