开机/重启后,在thor机器上执行
```
cd ~/lerobot && ./tools/thor/gmsl2/recover_argus.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1
```

然后在开发主机(通常是x86 架构host)的lerobot repo(origin/box分支)根目录执行
```
bash run/deploy.sh
```
此操作将本地最新代码增量同步到thor并执行gateway(运行在thor),frontend(运行在host),返还类似
http://localhost:5174/ 的页面, 可以直接访问.

如果期间遇到某几路相机起不来的情况,在thor执行
```
~/lerobot/tools/thor/gmsl2/recover_argus.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1
```

recover的大招是完全断电重启:包括thor的电源和转接板电源(12v3A),全部断开至少3s后,重新上电启动.