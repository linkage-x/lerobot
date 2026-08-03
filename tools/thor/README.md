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

如果期间遇到某几路相机起不来的情况,在thor执行
```
~/lerobot/tools/thor/gmsl2/recover_argus.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1
```

recover的大招是完全断电重启:包括thor的电源和转接板电源(12v3A),全部断开至少3s后,重新上电启动.