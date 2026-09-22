# Thor 部署与操作入口

外参标定、轨迹生成、机器人回放和夹爪同步诊断请参阅
[Thor 外参标定与 P0 机器人回放操作指南](CALIBRATION_AND_REPLAY.md)。

开机/重启后，在 Thor 机器上执行：

```bash
cd ~/lerobot && ./tools/thor/gmsl2/recover_argus.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1
```

然后在开发主机（通常是 x86 架构 host）的 lerobot 仓库根目录执行：

```bash
bash run/deploy.sh       # 默认 target=thor
bash run/deploy.sh thor  # 显式写法
```

脚本先通过 rsync 增量替换 Thor 上的代码，成功后再重启 gateway，最后在 host
启动 frontend。
工作站 FR3 遥操实例使用 `bash run/deploy.sh workstation`，部署到
`hph@192.168.100.155:/home/hph/Code/lerobot`，不会连接或启动 Thor 采集链路。
命令会返回类似 `http://localhost:5174/` 的页面地址，可以直接访问。

如果某几路相机无法启动，在 Thor 执行：

```bash
~/lerobot/tools/thor/gmsl2/recover_argus.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1
```

最后的恢复手段是完全断电重启：断开 Thor 和转接板（12 V/3 A）的全部电源至少
3 秒，然后重新上电。
