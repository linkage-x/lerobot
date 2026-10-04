# Thor 部署与操作入口

外参标定、轨迹生成、机器人回放和夹爪同步诊断请参阅
[Thor 外参标定与 P0 机器人回放操作指南](CALIBRATION_AND_REPLAY.md)。

开机/重启后，在 Thor 机器上执行：

```bash
cd ~/lerobot && ./tools/thor/gmsl2/recover_argus.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1
```

然后在开发主机（通常是 x86 架构 host）的 lerobot 仓库根目录执行：

```bash
bash run/deploy.sh       # 默认 Thor BOX UI + FR3 SpaceMouse 配置
bash run/deploy.sh thor  # 显式写法
bash run/deploy.sh --box-only  # 原始相机/BOX 配置
```

脚本先通过 rsync 增量替换 Thor 上的代码，成功后再重启 gateway，最后在 host
启动 frontend、空闲的 host FR3 服务和 SSH 通信隧道。SpaceMouse 插在 host；
host 负责 FR3 控制，Thor 负责相机、BOX 采集及收到 host 指令后的夹爪开合。
默认命令只连接 Thor，不需要工作站 SSH。打开脚本输出的 frontend 地址（通常为
`http://localhost:5173/`），在原 BOX 分支的 Live Record 页面使用：
`C` 连接用户勾选的盛云相机和 BOX 触觉/力传感器，`F` 移动 FR3 到起始位并启动
SpaceMouse 遥操，`E` 开始记录，`S` 保存，`D` 丢弃，`Esc` 退出。
FR3 故障时页面告警并停止运动；排除物理/Desk 故障后，再按 `F` 重新开始。

FR3 原生运行环境、SpaceMouse USB 权限、实时内核、测试步骤和数据字段见
[Thor FR3 SpaceMouse 操作指南](../../docs/thor_fr3_teleoperation.md)。
FR3 依赖缺失不会阻止部署或 `C` 连接传感器；`F` 会显示具体原因。
当前 Thor FR3 配置使用回放相同的 `192.168.11.102` 地址，以及
`fr3_teleop.realtime_mode: ignore`（libfranka `kIgnore`），不再强制要求
PREEMPT_RT。关节范围、通信超时和原生故障检查仍生效。需要严格实时检查时，
将该项改为 `enforce`。先在 host 执行 `bash run/setup_thor_fr3.sh --check`。
host 和 Thor 通过四时间戳探测估算时钟偏移，原始时间、映射时间和不确定度都写入
`fr3_state.jsonl`。断链会停止遥操，恢复网络后仍须显式按 F；不会自动恢复运动。

如果某几路相机无法启动，在 Thor 执行：

```bash
~/lerobot/tools/thor/gmsl2/recover_argus.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1
```

最后的恢复手段是完全断电重启：断开 Thor 和转接板（12 V/3 A）的全部电源至少
3 秒，然后重新上电。
