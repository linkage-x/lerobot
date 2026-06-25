# BOX 采集板传感器上行问题排障记录

本文记录 BOX 采集板传感器流不上行问题的排障历史。

> **⚠️ 2026-06-25 重大订正**：下文 §1–§6 描述的"板端固定只向 `192.168.2.45` 上行、
> 必须绑定 `192.168.2.45:15000`"是**旧固件**行为，现已**作废**。当前 v3 固件的 box 走
> DHCP、可落在任意网段（实测 `192.168.122.75`），并把数据流推到它从命令包学到的上位机
> 地址（由内核路由决定，与 `bind_ip` 无关）。因此 `bind_ip` **必须为 `0.0.0.0`**（通配）；
> 绑到 `192.168.2.45` 这类具体地址会在 box 不在该网段时**静默收不到任何包** → 0 box 样本、
> `observation.state/action` 全冻结。完整证据见 **§9（取代 §1–§6 的网络结论）**。

（以下 §1–§6 为旧固件时代的历史记录，保留备查。）

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

## 7. SDK 自动写 CSV 撑爆磁盘（临时绕过，待 SDK 修复）

`libbox_controller.so` 每次 `Box.start()` 后会在**进程 CWD** 无条件写一个
`box_sensor_data_<时间戳>.csv`（`append_sensor_csv`，路径是静态常量 `kCsvPath`），
约 **35 MB/分钟**。`.so` 里只暴露 `BOX_SDK_URDF` 一个 env，**没有关闭开关**。

**临时绕过**：`BoxClient.stop()` 会删除本会话新产生的 `box_sensor_data_*.csv`
（`cleanup_box_csv: true` 默认开启，按 `start()` 前后快照差集，只删本会话的）。
注意录制**进行中**该文件仍按 35 MB/min 增长，只有 `stop()` 后才回收——长 episode
仍需留意峰值占用。

**给供应商的请求**：为 CSV dump 增加 env 开关（如 `BOX_SDK_NO_CSV=1`）或可配置输出路径。
SDK 侧修复后，移除 `box_client.py` 的 `_cleanup_session_csv` / `cleanup_box_csv`。

## 8. 数据流"老化"与 DiscoveryKeepAlive 保活（2026-06-25 真机实测）

**结论先行**：BOX 的 ~200Hz 主动上行会在"最后一次与板端有通信后约 N 秒"老化停止；
`DiscoveryKeepAlive`（:15001 上每 3s 广播一次 REQ）**单独就足以**把数据流持续保活，
不需要再用 `set_mode` 周期重激活。因此 `box_client.py` 已：
- 按 demo.py 的方式显式实例化
  `DiscoveryKeepAlive(bind_port=15001, interval_ms=3000)`（`KEEPALIVE_INTERVAL_MS`）；
- 把 `rearm_interval_s` 默认改为 **0（关）**，仅保留为兜底开关。

### 实测方法

三组**全新 session**（`Box.start` → 持续读 `get_sensor_cache` → `Box.stop`）各跑 25s，
用 `SensorCache.liwp_index`（UDP 收包序号，每收到一包 +1；纯本地读，不向板端发包、
不会自己保活）作证据。冻结 = 序号钉死不动 = 板端已停推（此时 `valid` 仍可能=1，
返回的是**陈旧**缓存——正是 `stop_recording()` 去重后 0 samples 的来源）。

| 条件 | keepalive | set_mode 重激活 | 结果 |
|------|-----------|-----------------|------|
| C 裸奔 | 关 | 关 | ~600 idx/s 正常推，**t≈16–17s 突然冻结**，之后一直 FROZEN |
| A demo.py | 开（3s 广播） | 关 | **全程 24s 不冻**，稳定 ~600 idx/s |
| B 双开 | 开（3s 广播） | 开（0.5s） | 与 A 完全一致，无额外收益 |

测试环境：`box1672693301`，`fw=0x0100`，SDK `.so build_time=Jun 24 2026`，
box ip=`192.168.122.75`（新固件已无 `192.168.2.x` 网段限制）。
复现脚本当时为 `tools/thor/box_sdk/repro_ageout.py`（一次性诊断，验证后已删）。

### 与早期结论的关系（重要：老化窗口随固件而变）

早期排障曾得出"DiscoveryKeepAlive 不保活数据通道、必须 0.5s `set_mode` 重激活"
（commit `e57a1975` / `dec60625`，rearm 默认 0.5s）。本次实测推翻了"keepalive 不保活"
这一**一般化**结论。最可能的原因是**老化窗口随固件变化**：
- 早期固件老化窗口约 **~1s** → 默认 3s 的广播太慢、广播间隙里流就死，于是显得
  keepalive 没用、必须用 0.5s 的 `set_mode` 顶住；
- 当前固件老化窗口约 **~16s** → 3s 广播绰绰有余，单独即可保活，rearm 冗余。

判据是「**keepalive 周期 vs 老化窗口** 谁快」。所以保留 `rearm_interval_s` 兜底：
若换板/换固件后又出现 mid-session 断流，把它设回 `0.5` 即可恢复旧的 set_mode 顶流策略。

### 给供应商的问题

`fw=0x0100` 仍存在数据流老化（无周期通信 ~16s 后停推），与"200Hz 主动上报"的设计不符。
是否为预期行为？能否提供一个**正规的数据通道心跳/订阅保持**接口，并明确老化窗口时长，
而不是让上位机靠 `DiscoveryKeepAlive` 广播或 `set_mode` 重发来间接续命？

## 9. 真正的根因：`bind_ip` 必须 `0.0.0.0`（2026-06-25 实测，推翻 §1–§6 网络结论）

§1–§6 把"0 上行"归因于"板端固定向 `192.168.2.45` 上行、Thor 未绑 `.45`"。这在**旧固件**下
成立，但当前 v3 固件已无此限制；继续按 §1–§6 把 `bind_ip` 绑死 `192.168.2.45` 反而**正是**
后来一连串 0-sample 录制（如 `thor_gmsl2_11ch_v1_20260625_135316`，state/action 全 31 维常量）
的根因。

**机制**：box（DHCP，实测 `192.168.122.75`）从收到的命令包里学到"上位机地址"，这个地址是
内核按路由为"发往 box"挑的源地址——`/16` 路由下是 `192.168.111.122`，**不是** `192.168.2.45`
（`192.168.2.45/24` 不覆盖 `122.x`）。box 于是把 200Hz 流推到 `192.168.111.122`。接收 socket
若 `bind` 在 `192.168.2.45:15000`，就永远收不到 → `get_sensor_cache` 始终 `valid=0` → 0 样本
→ `observation.state/action` 全维灌常量 → replay 水平直线。

**同进程 back-to-back A/B（同一 box `1672693301` @`192.168.122.75`，keepalive 开、rearm 关）**：

| `bind_ip` | 7s 收包数 |
|-----------|-----------|
| `192.168.2.45` | **0**（NO DATA） |
| `0.0.0.0` | 2,887,447（OK） |
| `192.168.2.45`（再测） | **0** |

**这也解释了当天"时好时坏"**：box 的 DHCP IP 在变——落在 `192.168.2.x` 时 bind `.45` 能收
（`073552/081125/085938` 成功），落在 `122.x` 时全 0（当天大多数 + `135316`）。先前
"box 假死 / 老化 / 需断电"的判断**大多是这个网络错配的假象**：`demo.py`（bind `0.0.0.0`）
与独立 probe 全程稳推 ~600 idx/s、box `uptime` 连续 58 分钟未重启，即为反证。

**修复**：`bind_ip: "0.0.0.0"`（已改：模板 `tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml`
+ 活动 overlay `outputs/.active_task_config.yaml`）。overlay 由基准 config 深拷贝、只改
`dataset.*`（`tools/data_collection_gui/gateway.py:_build_task_overlay_config`），所以
**基准模板也必须改**，否则重新激活任务会把 overlay 覆盖回 `192.168.2.45`。`remote_ip` 可留
旧值，`BoxClient.start()` 会 `discover()` + `register_device()` 纠正到 box 的真实地址。

注意：这条修复与 §8 的老化结论**正交**——老化（无周期通信 ~16s 停推）在 `bind 0.0.0.0`
能正常收包的前提下才谈得上，由 `DiscoveryKeepAlive` 保活。`bind_ip` 错时连第一个包都收不到，
跟老化无关。
