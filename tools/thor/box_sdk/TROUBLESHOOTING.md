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

## 4. 当前剩余现象

`.45` 修正后，SDK 已能收到上行 UDP，`get_sensor_cache` 也已返回 `rc=0 valid=1`。本轮最小脚本里仍观察到：

```text
get_gripper_pos -> rc=5 timeout 0.0
cache valid=1, liwp=0 0, gripper timestamp=0, imu timestamp=0
```

这表示 UDP 上行链路已经恢复，但具体字段是否应有非零 timestamp、`get_gripper_pos` 是否应在 collection 模式下返回，还需要结合板端模式和传感器实际接入状态确认。

## 5. 后续验证重点

1. 点击 connect 后确认 6 个 BOX sensor 行是否全部进入 connected/running。
2. 保存一段 episode，检查 `meta.json` 中 `box_collection.snapshots[*].status.last_rc` 是否为 `0`，`last_error` 是否为 `ok`。
3. 如果 sensor timestamp 仍为 0，继续确认当前固件在 collection 模式下哪些 TLV 字段应填充 timestamp。
4. 如果 `get_gripper_pos` 仍 timeout，向供应商确认该接口是否只在 control 模式或特定 enable 命令后可用。

## 6. 给供应商的简短结论

板端固定上行 IP 为 `192.168.2.45`。Thor 改为 `192.168.2.45/24` 并让 SDK 绑定 `192.168.2.45:15000` 后，网卡层已收到 `BOX 192.168.2.60 -> Thor 192.168.2.45 UDP/15000`，`get_sensor_cache` 已由 `rc=4 no cached sensor data` 变为 `rc=0 ok valid=1`。当前剩余问题不是 UDP 上行不到达，而是确认各传感器字段/timestamp 和 `get_gripper_pos` 在当前模式下的预期行为。
