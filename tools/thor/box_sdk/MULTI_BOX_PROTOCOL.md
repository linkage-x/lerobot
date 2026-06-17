# 多夹爪（多 BOX）自动检测 —— 协议草案与对接清单

面向 BOX SDK 供应商。目标：在**同一网段**接入 N 个 BOX（每个带一个夹爪 +
IMU/trigger/六维力/触觉），host 端**自动发现**并按 box 区分采集，无需逐个硬编码 IP。

本仓库已经实现了**host 端的一半**（fleet config + `BoxPool` + 可插拔
`BoxDiscovery`，见 `box_client.py`）。缺的另一半在 SDK/固件侧，需要按本文约定。

---

## 1. 为什么必须先和供应商定协议（这是关键路径）

当前 SDK 是**点对点**的，从代码可证三条硬约束，决定了多 box **无法**靠 host 端绕过：

1. **固件把数据推到固定的 host:port**（`ensure_box_net.sh`：固件配置为推到
   `192.168.2.45:15000`）。多个 box 会全部打到同一个 host:port。
2. **本机 UDP 端口必须固定 15000**（`需求整理 §5.1`、`BoxClientConfig.__post_init__`）。
   没法简单地“一个 box 起一个本地端口”。
3. **`get_sensor_cache()` 返回单个合并后的 `SensorCache`，不带来源标识**；
   `box_start(UdpConfig)` 只接受单个 remote。一个 `Box` 实例 = 一个 box。

结论：**区分同网段的 N 个 box，必须由 SDK 提供“按来源归属 / 枚举”能力**。SDK 形态没定之前，
host 端不应投机实现传输层（猜错就返工）。所以——**应当预先和供应商沟通协议**，让下一版 SDK
直接按下面的接口做，我们已就位的脚手架即可插上。

---

## 2. 需要供应商确认/实现的 4 个决策点

### 2.1 设备身份（identity）—— 用稳定 ID，别只靠 IP
- 每个 BOX 在**每个数据包**里携带一个**稳定唯一 ID**（出厂序列号 `box_serial`，
  16/32 字节定长），与 IP 解耦。IP 可能因 DHCP/换线变化，ID 不变。
- host 端以 `box_serial`（或其映射的 `box_id`）作为 namespace 前缀，落库为
  `box0/box_gripper`，保证数据集 key 跨会话稳定。

### 2.2 发现机制（discovery）—— SDK 给一个枚举调用
推荐 SDK 暴露：
```c
// 在 subnet 上发现当前在线的 BOX，timeout_ms 内返回
int box_discover(uint32_t timeout_ms, BoxInfo* out, size_t* count);
typedef struct {
  char     box_serial[32];   // 稳定唯一 ID
  char     ip[16];           // 当前 IP
  char     model[32];        // 夹爪型号 / 设备类型
  uint32_t fw_version;       // 固件版本
  uint32_t proto_version;    // 协议版本
  uint32_t sensor_mask;      // 该 box 实际带哪些 sensor（gripper/imu/...）
} BoxInfo;
```
底层实现两选一（供应商定，host 不关心）：
- **被动 beacon**：每个 box 周期性向广播/多播地址发一个 hello（含上面字段）；
- **主动查询**：host 广播一个 query，在线 box 回包。

孤立工业子网用简单的 **UDP 广播 beacon** 即可，不必上 mDNS/SSDP。

### 2.3 传输多路复用（transport）—— 一个 socket 收多 box
因本机端口固定 15000，N 个 box 都打到 `:15000`。SDK 需在同一 socket 上**按 source
IP / 包内 `box_serial` 解复用**，并提供按 box 取数的接口，二选一：
```c
// 方案 A：按 box 取（推荐，贴合现有 get_sensor_cache）
int box_get_sensor_data_by_id(const char* box_serial, SensorCache* out);
// 方案 B：注册回调，每包带来源
void box_set_packet_observer_v2(void* ctx,
    void (*cb)(void* ctx, const char* box_serial, const SensorCache* snap));
```
> 备注：现 SDK 已有 `box_set_packet_observer(handle, cb, ctx)`。**若它的回调能带上
> source 地址**，那方案 B 几乎是现成的——这是唯一可以*现在就验证*的检测路径，
> 请供应商确认其 callback 是否暴露来源 IP。

### 2.4 控制方向也要可寻址
control 模式下的 `set_mode` / `set_clamp_pos` / `set_trigger_zero` 等，需带
`box_serial` 形参，命令能精确发往指定 box。

---

## 3. 协议怎么定才“好”——几条原则

1. **自描述包**：每包带 `box_serial / model / proto_version / sensor_mask`，host 不再
   硬编码 `KNOWN_SENSOR_IDS`，按 `sensor_mask` 动态渲染设备行。
2. **版本协商**：`proto_version` 字段先行，host 与 box 能各自演进、互相识别。
3. **退化兼容（硬要求）**：新 SDK 必须仍能用**单 box 旧时序**跑通——单 box 不带
   `box_serial` 时行为与今天一致；本仓库 `box_id=""` 即此退化路径，wire/数据集不变。
4. **以接口契约对接，而非“支持多夹爪”这种模糊需求**：把本文 §2 的
   `box_discover` + 按 box 取数 + 自描述包 + 版本号，作为我们要编码对接的**数据契约**
   交给供应商。`BoxDiscovery.discover()`（见下）就是 host 侧契约的落点。
5. **ID↔IP 映射稳定**：host 端配置以 `box_id` 为准，IP 仅作当前连接信息，换 IP 不影响落库 key。

---

## 4. Host 端已就位的脚手架（供应商实现后即插即用）

`tools/thor/box_sdk/box_client.py`：

- `BoxFleetConfig` / `fleet_from_yaml_dict`：支持 `boxes:` 列表，每个 box 一份
  `BoxClientConfig`（带 `box_id`）。单 box 旧格式自动退化为 `box_id=""`。
- `BoxDiscovery` 可插拔接口：
  - `StaticBoxDiscovery` —— 今天用，返回配置里写死的 box 列表；
  - `SdkBoxDiscovery` —— **供应商的 `box_discover` 落点**。现为占位：调用时告警并退回
    static 列表，所以 `discovery: sdk` 现在就能安全写进配置。供应商交付后，把它的
    `discover()` 改成调 `box_discover` → 把 `BoxInfo[]` 映射成 `BoxClientConfig[]` 即可，
    `BoxPool` 无需改动。
- `BoxPool`：拥有 N 个 `BoxClient`，聚合 `read/detect/observed_rates/recording`，按
  `box_id` 用 `namespace_sid()` 加前缀；单 box 空 id 时直通现有实现（零行为变化）。
- 录制：`box_sensors.jsonl` 的 `sid` 会变成 `box0/box_gripper`；下游按前缀分组即可。

配置示例见 `thor_gmsl2_11ch_example.yaml` 里 `box_collection` 下方注释。

---

## 5. 给供应商的一句话清单

> 请在下一版 SDK 提供：(1) 每包携带稳定 `box_serial` 与 `sensor_mask`/`proto_version`；
> (2) `box_discover(timeout)` 子网枚举；(3) 同一 15000 端口上按来源解复用 +
> `box_get_sensor_data_by_id()`（或带来源的 packet observer）；(4) 控制接口按
> `box_serial` 寻址；(5) 保持单 box 旧时序兼容。并请确认现有
> `box_set_packet_observer` 回调是否已带 source IP。
