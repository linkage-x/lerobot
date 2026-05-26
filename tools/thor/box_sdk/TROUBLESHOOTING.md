# BOX 采集板传感器上行问题排障记录

本文用于提交给供应商定位 BOX 采集板传感器流不上行问题。重点结论：Thor 主机侧网络、UDP 监听、SDK 启动和 LeRobot 记录链路已基本排除；当前问题收敛到 BOX 板端没有向 Thor 发回 UDP 响应/传感器上行包，或板端目标地址、端口、网口、固件/协议配置不匹配。

## 1. 现场环境与拓扑

- Thor 主机：`nvidia@192.168.111.122`
- Thor BOX 网口：`enP2p1s0`
- Thor BOX 源地址：`192.168.2.44/24`
- BOX 采集板地址：`192.168.2.60`
- BOX MAC：`00:80:e1:00:00:00`
- SDK 本地监听/发送端口：`192.168.2.44:15000`
- BOX 目标端口：`192.168.2.60:15000`
- SDK 版本信息：`build_time=May 21 2026 11:24:48, commit=e2ea1a3`
- 接线：局域网路由器、Thor、夹爪/BOX 相关设备均通过网口接入同一个 PoE 交换机。该接线方式本身没有明显问题，前提是交换机没有 VLAN/隔离策略，且 BOX 板端目标 IP/端口配置正确。

## 2. 当前现象

点击 connect 后，LeRobot/Thor 侧可以启动 BOX SDK 会话，6 个 BOX sensor 行可以显示为 connected/running，状态和时间戳也会被写入 episode meta。但是 SDK 一直没有收到有效传感器缓存：

```text
get_sensor_cache: rc=4, err="no cached sensor data", valid=0
liwp=0 0
gripper timestamp=0
imu timestamp=0
sensor timestamps=0
```

查询类命令同样没有收到板端响应：

```text
set_mode 0 -> rc=0 ok
set_mode 1 -> rc=0 ok
get_mode -> rc=5 timeout
get_gripper_pos -> rc=5 timeout
```

说明：`set_mode rc=0` 只能证明 SDK 已把 UDP 命令发出，不能证明 BOX 板端已经收到或已回包。

## 3. Thor 侧已排除项

Thor 侧网络和监听检查结果如下：

```text
ip route get 192.168.2.60
192.168.2.60 dev enP2p1s0 src 192.168.2.44

ip neigh show 192.168.2.60
192.168.2.60 dev enP2p1s0 lladdr 00:80:e1:00:00:00

ss -lunp
192.168.2.44:15000 被 python/BOX SDK 进程绑定

rp_filter
enP2p1s0=0, all=0
```

连通性检查：

```text
ping -I enP2p1s0 192.168.2.60
3 packets transmitted, 3 received, 0% packet loss
```

原始 `AF_PACKET` 抓包结论：

- 能看到 Thor -> BOX 的 ICMP echo request。
- 能看到 BOX -> Thor 的 ICMP echo reply。
- 能看到 Thor -> BOX 的 UDP `15000 -> 15000` 命令包。
- 看不到任何 BOX -> Thor 的 UDP 响应或传感器上行包。
- 30 秒被动监听 BOX MAC，在不主动发 SDK 命令时没有看到 BOX 主动上行数据。

典型抓包片段：

```text
192.168.2.44 -> 192.168.2.60  ICMP echo request
192.168.2.60 -> 192.168.2.44  ICMP echo reply
192.168.2.44 -> 192.168.2.60  UDP 15000 -> 15000
# 未观察到：192.168.2.60 -> 192.168.2.44 UDP
```

因此，Thor 可以从 BOX 收到入站 IP 包，网口、路由、ARP、反向路径过滤和本机 UDP 监听不是当前主因。由于原始网卡层抓包也看不到 BOX 入站 UDP，LeRobot wrapper、gateway、recorder 和主机防火墙也不是第一嫌疑。

## 4. 当前判断

`get_sensor_cache` 返回 `rc=4 / no cached sensor data` 的含义是：Thor 侧 SDK 的传感器缓存没有收到可解码的传感器数据。结合原始抓包，当前更像是 BOX 板端没有向 `192.168.2.44:15000` 发送 UDP 数据，而不是 Thor 已收到但解析失败。

可能根因集中在板端：

- sensor push / ARM gateway 任务没有启动。
- 板端传感器流目标 IP/端口不是 `192.168.2.44:15000`。
- 板端把数据发到了其他网口、VLAN、广播/组播地址或其他端口。
- 板端固件与当前 SDK `e2ea1a3` 协议不匹配。
- 板端需要额外 enable/start-stream 命令，不能只依赖 `set_mode(0/1)`。
- 板端服务异常：ICMP 正常，但 UDP 协议服务没有运行或没有绑定/回包。

## 5. 需要供应商确认的问题

请供应商优先确认以下事项：

1. BOX 板端当前固件版本是否与 SDK `commit=e2ea1a3` 匹配。
2. 板端 sensor push / ARM gateway / UDP 服务是否已经启动，启动日志是否有报错。
3. 板端传感器上行目标是否配置为 `192.168.2.44:15000`，源/目标端口是否确认为 `15000`。
4. 板端是否可能把传感器流发到 `192.168.111.122`、其他历史 IP、广播/组播地址、其他网口或 VLAN。
5. 除 `Box.start(local_ip=192.168.2.44, local_port=15000, remote_ip=192.168.2.60, remote_port=15000)` 和 `set_mode(0/1)` 外，是否还需要额外命令开启传感器上报。
6. 是否有板端命令、串口日志、配置文件或管理工具可以直接查看/修改 sensor push 目标 IP 和端口。

## 6. 供应商可按此复现

在 Thor 上启动 SDK，并观察查询结果：

```text
Box.start("192.168.2.44", 15000, "192.168.2.60", 15000) -> rc=0
set_mode(0/1/0) -> rc=0
get_mode(800) -> rc=5 timeout
get_gripper_pos(500) -> rc=5 timeout
get_sensor_cache() -> rc=4 no cached sensor data
```

同步做网卡层抓包，预期会看到：

```text
Thor 192.168.2.44 -> BOX 192.168.2.60 UDP/15000
```

当前实际没有看到：

```text
BOX 192.168.2.60 -> Thor 192.168.2.44 UDP/15000
```

这就是需要供应商重点定位的缺口。
