# BOX 采集板传感器上行问题排障记录

本文记录 BOX 采集板传感器流不上行问题的最新结论。供应商确认夹爪/BOX 板端固定只向 `192.168.2.45` 上行数据；Thor 必须在 BOX 网口配置 `192.168.2.45/24`，并让 SDK 绑定 `192.168.2.45:15000`。此前使用 `192.168.2.44` 验证时收不到上行 UDP，是目标 IP 不匹配导致的误判。

## 1. 现场环境与拓扑

- Thor 主机：`nvidia@192.168.111.122`
- Thor BOX 网口：`enP2p1s0`
- Thor BOX 上行接收地址：`192.168.2.45/24`
- BOX 采集板地址：`192.168.2.60`
- BOX MAC：`00:80:e1:00:00:00`
- SDK 本地监听/发送端口：`192.168.2.45:15000`
- BOX 目标端口：`192.168.2.60:15000`
- 当前 Thor 上 SDK 日志版本：`build_time=May 26 2026 16:16:52, commit=c4f7a97`
- 接线：局域网路由器、Thor、夹爪/BOX 相关设备均通过网口接入同一个 PoE 交换机。该接线方式本身没有明显问题，前提是交换机没有 VLAN/端口隔离/ACL 策略阻断同交换机端口互通。

## 2. 修正后的配置

Thor 重启后必须保留 `192.168.2.45/24`。repo 已提供开机恢复服务：

```bash
cd ~/lerobot
bash tools/thor/box_sdk/install_box_net_service.sh
```

运行配置已改为固定绑定 `.45`：

```yaml
box_collection:
  enabled: true
  bind_ip: "192.168.2.45"
  bind_port: 15000
  remote_ip: "192.168.2.60"
  remote_port: 15000
```

Thor 侧地址和路由检查结果：

```text
ip -br addr show enP2p1s0
enP2p1s0 UP 192.168.111.122/16 192.168.2.45/24 ...

ip route get 192.168.2.60
192.168.2.60 dev enP2p1s0 src 192.168.2.45

ip neigh show 192.168.2.60
192.168.2.60 dev enP2p1s0 lladdr 00:80:e1:00:00:00 REACHABLE
```

## 3. 复测结果

按 `.45` 复测后，基础连通性正常：

```text
ping -I 192.168.2.45 -c 2 192.168.2.60
2 packets transmitted, 2 received, 0% packet loss
```

SDK 最小复现结果：

```text
Box.start("192.168.2.45", 15000, "192.168.2.60", 15000) -> rc=0 ok
set_mode 0 -> rc=0 ok
get_mode -> rc=0 ok 0
set_mode 1 -> rc=0 ok
get_mode -> rc=0 ok 1
set_mode 0 -> rc=0 ok
get_mode -> rc=0 ok 0
get_sensor_cache -> rc=0 ok valid=1
```

原始 `AF_PACKET` 抓包已确认 BOX 有 UDP 上行到 Thor：

```text
192.168.2.60 -> 192.168.2.45 UDP 15000 -> 15000
sniff_count 9636
```

因此，之前 `get_sensor_cache rc=4 / no cached sensor data` 的直接原因是 Thor 绑定/验证地址使用了 `192.168.2.44`，而板端固定只向 `192.168.2.45` 上行。

## 4. LeRobot wrapper 验证结果

修正 `bind_ip=192.168.2.45` 后，LeRobot `BoxClient` 已能解码并记录 6 个 BOX sensor：

```text
valid=true
last_rc=0
valid_poll_count=39
connected=[box_gripper, box_imu, box_trigger, box_six_d_force, box_touch_left, box_touch_right]
detected=[box_gripper, box_imu, box_trigger, box_six_d_force, box_touch_left, box_touch_right]
```

典型传感器字段：

```text
box_gripper.distance_m = 0.09785686433315277
box_gripper.timestamp  = 9386808
box_imu.timestamp      = 9388459
box_trigger.timestamp  = 9388423
box_six_d_force.timestamp = 9386732
box_touch_left.timestamp  = 3601178947
box_touch_right.timestamp = 9364164
```

代码侧也已兼容供应商 demo 中出现过的启动瞬态：如果 `gripper_data.distance` 已有有效非零值，即使 timestamp 暂时为 0，也会先记录 `box_gripper`，避免误判为无数据。

## 5. 当前结论

BOX 传感器流上行链路已打通，LeRobot wrapper 已能把 6 个 BOX sensor 标记为 seen/fresh，并记录 gripper distance 与各 sensor timestamp。后续如仍出现 `rc=4`，优先检查 `thor-box-net.service` 是否 active、Thor 是否仍持有 `192.168.2.45/24`，当前 recorder/gateway 是否使用了 `bind_ip: "192.168.2.45"`，并从 `192.168.2.45` 主动 probe 一次 `192.168.2.60` 以刷新 BOX 重上电后的 ARP/邻居状态。

## 6. 给供应商的简短结论

板端固定上行 IP 为 `192.168.2.45`。Thor 改为 `192.168.2.45/24` 并让 SDK 绑定 `192.168.2.45:15000` 后，网卡层已收到 `BOX 192.168.2.60 -> Thor 192.168.2.45 UDP/15000`，`get_sensor_cache` 已由 `rc=4 no cached sensor data` 变为 `rc=0 ok valid=1`；LeRobot `BoxClient` 进一步确认 6 个 BOX sensor 全部 seen/fresh，相关 timestamp 和 gripper distance 已可记录。
