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
启动 frontend。
默认命令只连接 Thor，不需要工作站 SSH。打开脚本输出的 frontend 地址（通常为
`http://localhost:5173/`），在原 BOX 分支的 Live Record 页面使用：
`C` 连接全部可用盛云相机和 BOX 触觉/力传感器，`F` 移动 FR3 到起始位并启动
SpaceMouse 遥操，`E` 开始记录，`S` 保存，`D` 丢弃，`Esc` 退出。
FR3 故障时页面告警并停止运动；排除物理/Desk 故障后，再按 `F` 重新开始。

FR3 原生运行环境、SpaceMouse USB 权限、实时内核、测试步骤和数据字段见
[Thor FR3 SpaceMouse 操作指南](../../docs/thor_fr3_teleoperation.md)。
FR3 依赖缺失不会阻止部署或 `C` 连接传感器；`F` 会显示具体原因。

如果某几路相机无法启动，在 Thor 执行：

```bash
~/lerobot/tools/thor/gmsl2/recover_argus.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1
```

最后的恢复手段是完全断电重启：断开 Thor 和转接板（12 V/3 A）的全部电源至少
3 秒，然后重新上电。
