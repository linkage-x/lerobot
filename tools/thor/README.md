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

如果期间遇到某几路相机起不来的情况,在thor执行
```
~/lerobot/tools/thor/gmsl2/recover_argus.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1
```

recover的大招是完全断电重启:包括thor的电源和转接板电源(12v3A),全部断开至少3s后,重新上电启动.
