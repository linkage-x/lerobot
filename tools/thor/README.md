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

## P0 双 marker 标定（独立进程，不使用 GUI）

新流程由 host 发起 SSH，但相机、AprilTag 检测和 FR3 仍全部连接在 Thor1；
不需要启动 data-collection gateway/frontend。首次运行或重标定：

```bash
bash tools/thor/run_p0_two_marker_calibration.sh -- \
  --existing recalibrate \
  --execute \
  --confirmation P0_TWO_MARKER_TEACHING
```

脚本通过 X11 在 host 显示多相机画面。窗口中绿色表示同一路相机同时看到
tag36h11 56/57，黄色表示只看到其中一个，红色表示没有有效检测；每格同时显示该
相机累计有效捕捉数。机械臂进入全关节零刚度 teaching mode 后，用手拖到新姿态并
停稳，按 `Enter` 保存当前同步相机簇、实测 `T_base_ee`、关节位置和检测结果。
按 `q` 在满足最少姿态/相机/双 tag 覆盖后开始求解，`f` 强制尝试求解，`Esc` 放弃且
不会替换已有标定。

求解会按相机并行检测。若已有通过质量门的标定，会固定已保存的
`T_ee_tag56/T_ee_tag57`，多线程生成各观测的 robot-base pose 候选，只重算
`T_world_base`；首次运行则联合求解 base 和两个 marker→EE。通过质量门后生成新的
P0 重定位轨迹并原子更新兼容的 `p1_simple_eye_hand_calibration/active.json`。

下次只使用已有标定并立即退出（不会连接相机或机械臂）：

```bash
bash tools/thor/run_p0_two_marker_calibration.sh --no-sync -- --existing reuse
```

不传 `--existing` 时会先询问使用已有标定还是重新标定。若 host 没有出现窗口，检查
本机 X server 以及 Thor 的 sshd `X11Forwarding` 设置。

如果期间遇到某几路相机起不来的情况,在thor执行
```
~/lerobot/tools/thor/gmsl2/recover_argus.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1
```

recover的大招是完全断电重启:包括thor的电源和转接板电源(12v3A),全部断开至少3s后,重新上电启动.
