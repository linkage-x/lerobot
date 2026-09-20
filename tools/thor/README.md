开机/重启后,在thor机器上执行
```
cd ~/lerobot && ./tools/thor/gmsl2/recover_argus.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1
```

然后在开发主机(通常是x86 架构host)的lerobot repo(origin/box分支)根目录执行
```bash
bash run/deploy.sh       # 默认 target=thor
bash run/deploy.sh thor  # 显式写法
```
脚本先通过 rsync 增量替换 Thor 上的代码，成功后再重启 gateway，最后在 host 启动 frontend。
工作站 FR3 遥操实例使用 `bash run/deploy.sh workstation`，部署到
`hph@192.168.100.155:/home/hph/Code/lerobot`，不会连接或启动 Thor 采集链路。
返还类似
http://localhost:5174/ 的页面, 可以直接访问.

Episode Replay 页面中的 `P0 Native Arm Replay · Checks Disabled` 使用独立入口
`tools/thor/run_p0_native_arm_unchecked.sh`。该入口不会运行项目侧轨迹审计、
机械臂状态安全检查或碰撞/场景检查。FR3 建连后，panda_py 会读取一次实测关节位置，
以 0.05 速度因子规划并执行到保存轨迹首位姿的 `start` 段；该起步路径也不做
审计或碰撞检查。之后的 `replay_*` 分段仍不使用实测位置重新规划。GUI 和脚本运行时都会显示高风险提示，
操作者必须清空现场、手持实体急停并输入 `YES`。软件 Abort 不能替代实体急停。

Calibration 页的 `P1_simple_eye_hand_calibration` 用夹爪附近固定的
tag36h11 ID 56/57（黑框有效尺寸 55 mm，70 mm 是背板总尺寸）重新定位移动后的
FR3 base。它复用当前 P0 的生产相机内外参，从
`thor_gmsl2_extrinisics_robot_base_0720` 背后的 211 组数据库记录中，先保留旧标定时
TCP Z 不低于 `0.250469 m` 的姿态（当前实测桌面接触 TCP Z=`0.100469 m`，保留
`150 mm` 余量），以及至少被 4 路相机有效观测的姿态。再从离历史可见相机最近的
2.5 倍候选池中按 TCP 平移/旋转覆盖选出 50 组；直接使用数据库
中的 pose 和 joint values，不做 pitch 翻转或重新 IK。停稳后同步记录实测 TCP 与 Thor 图像，并联合求解
`T_world_base`、`T_tcp_tag56`、`T_tcp_tag57`。质量门通过后只生成 P0 候选；操作者
还需在 GUI 中显式激活，下一次 unchecked P0 replay 才会读取经过 SHA-256 固定的
重定位关节轨迹。采集会移动真机且不做项目侧场景碰撞检查，必须输入
`P1_MOVE_FR3`，清空现场并手持实体急停。

## 当前 FR3 base 下的固定相机外参标定（独立进程，不使用 Web GUI）

新流程由 host 发起 SSH，但相机、AprilTag 检测和 FR3 仍全部连接在 Thor1；
不需要启动 data-collection gateway/frontend。首次运行或重标定：

```bash
bash tools/thor/run_p0_two_marker_calibration.sh -- \
  --existing recalibrate \
  --execute \
  --confirmation P0_TWO_MARKER_TEACHING
```

脚本通过 X11 在 host 显示多相机画面，检测固定在 EE 上的
`tag36h11 ID 6`（黑框尺寸 `160 mm`）。绿色表示该相机看到 tag，红色表示没有有效
检测；每格同时显示该相机累计有效捕捉数。机械臂进入全关节零刚度 teaching mode 后，用手拖到新姿态并
停稳，按 `Enter` 保存当前同步相机簇、实测 `T_base_ee`、关节位置和检测结果。
按 `q` 在各相机达到目标捕捉数后开始求解，`f` 强制尝试求解，`Esc` 放弃且
不会替换已有标定。

本流程不重新估计内参。默认读取已有的 fisheye 标定：

```text
/home/nvidia/lerobot/outputs/calibration/thor_gmsl2_selfcal_0804_fisheye_intrinsics/summary.json
```

除 `cam_03` 外，各相机使用同名 fisheye 内参；`cam_03` 暂时使用 `cam_13` 内参。
这只影响 `cam_03` 图像的畸变矫正/PnP：`cam_03` 的 `T_base_camera` 仍由它自己的
tag 图像和同步 FR3 pose 求解，不会复制 `cam_13` 外参。输出会显式写入
`intrinsics_source_camera` 和 `temporary_intrinsics_reuse`。输入必须声明为
`opencv_fisheye`/`fisheye`/`equidistant` 且含 4 个畸变系数，否则在连接机器人之前
失败。可用 `--intrinsics-summary PATH` 选择另一份已有内参。

程序联合求解固定的 `T_ee_tag6` 和每台相机的 `T_base_camera`，其中 `base` 就是
本次运行时的当前 FR3 base。通过质量门后才会原子更新
`outputs/calibration/p0_single_tag_camera_calibration/active.json`。

下次只使用已有标定并立即退出（不会连接相机或机械臂）：

```bash
bash tools/thor/run_p0_two_marker_calibration.sh --no-sync -- --existing reuse
```

不传 `--existing` 时会先询问使用已有标定还是重新标定。若 host 没有出现窗口，检查
本机 X server 以及 Thor 的 sshd `X11Forwarding` 设置。

若出现 libfranka `communication_constraints_violation`，先隔离测试同一个 teaching
controller。下面的命令不会启动 Argus、AprilTag 或 Tk，但仍会让真实 FR3 进入零
刚度模式，必须在现场支撑机械臂并手持急停：

```bash
bash tools/thor/run_p0_two_marker_calibration.sh -- \
  --existing recalibrate \
  --robot-only-test-seconds 20 \
  --execute \
  --confirmation P0_TWO_MARKER_TEACHING
```

程序每秒输出 `control_command_success_rate`。若 robot-only 稳定而完整标定失败，说明
相机/检测/显示负载是触发因素。完整标定现默认使用 2 个检测 worker，并将 frame-bus
预览降为每 15 帧一次；可进一步临时使用
`--detection-workers 1 --frame-bus-every-n 30 --detection-scale 0.25`。

standalone 启动器会保留 X11 给 OpenCV 窗口，但 Argus recorder 子进程会主动清除
转发的 `DISPLAY/WAYLAND_DISPLAY`。否则 NVIDIA EGL 会把 `localhost:10.0` 当作
Thor 本地 EGL display，随后所有相机依次报 `NvBufSurfaceMapEglImage failed`；这类
逐相机 drop 是同一个显示环境错误，不表示每路相机都损坏。自动/手动调用的
`recover_argus.sh` 也会清除这两个变量，确保内部 GStreamer 探针走 headless EGL。

相机采集不再固定 `[6,7,8,9,12,13,14]`，每次从 MAX96726 当前 locked links 动态
生成。已有内参仍以物理相机的 `cam_XX` 为身份；若拔插后某台物理相机的运行时编号
确实改变，只有确认物理相机身份后才能显式映射，例如当前 `cam_05` 原本是
`cam_06`：

```bash
bash tools/thor/run_p0_two_marker_calibration.sh -- \
  --existing recalibrate \
  --camera-alias cam_05=cam_06 \
  --execute \
  --confirmation P0_TWO_MARKER_TEACHING
```

可重复传入 `--camera-alias`。不要仅凭“画面能打开”猜映射；错误套用另一台相机的
内参会产生数值看似收敛、物理意义错误的 `T_base_camera`。

FR3 Python 环境使用 headless OpenCV，因此 standalone 的操作窗口由 Tk/Pillow
显示，AprilTag 绘制与图像处理仍使用 OpenCV。程序先确认窗口可用且连续收到至少
两个同步 cluster，之后才连接机器人并启用 teaching mode。运行中若超过 1 秒没有
新 cluster，Enter 会拒绝捕捉，避免把旧图像与当前 robot pose 配对。

`cam_02` 是 UMI camera，不需要参与固定相机外参标定，因此该
standalone 程序永久默认排除 `cam_02`；普通 Thor recorder 的相机策略不受影响。
`cam_03` 会进入同步集合并使用上述 `cam_13` 内参 fallback。若其他相机持续
`timed out waiting for Argus buffer`，
可以额外排除，例如 `cam_07`：

```bash
bash tools/thor/run_p0_two_marker_calibration.sh -- \
  --existing recalibrate \
  --exclude-camera cam_07 \
  --execute \
  --confirmation P0_TWO_MARKER_TEACHING
```

如果期间遇到某几路相机起不来的情况,在thor执行
```
~/lerobot/tools/thor/gmsl2/recover_argus.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1
```

recover的大招是完全断电重启:包括thor的电源和转接板电源(12v3A),全部断开至少3s后,重新上电启动.
