# 在线推理读取 GMSL2 相机流的接口说明

日期：2026-07-07

## 结论

在线推理需要读相机画面时，不应该让模型进程自己再打开
`nvarguscamerasrc` 或新的 Argus camera session。当前实现的边界是：

```text
argus_online_sync recorder 独占相机流
  -> 形成 full SOF cluster
  -> 同步帧进入硬件 encoder
  -> 同步帧同时发布到 frame bus

模型 / 控制进程
  -> 只读 frame bus
  -> 不直接打开相机
```

这样可以保证模型拿到的多路图像来自同一个 recorder 接受的同步
logical frame，避免和录制路径抢 Argus 资源。

## 已实现接口

`argus_online_sync` 增加了一个可选的 recorder-owned frame bus：

```text
/dev/shm/lerobot_online_sync/
  latest_cluster.json
  slot0_cam_03.nv12
  slot0_cam_06.nv12
  ...
  slot1_cam_03.nv12
  slot1_cam_06.nv12
  ...
```

默认关闭。打开后，recorder 在每个 accepted full SOF cluster 被送入
encoder 前，把该 cluster 的 NV12 图像写入 frame bus，并原子更新
`latest_cluster.json`。

`slot0/slot1` 是双缓冲，避免读端看到半更新的 cluster。读端只需要读
`latest_cluster.json` 指向的文件。

## 如何打开

在 YAML 里显式配置：

```yaml
sensors:
  cameras:
    defaults:
      recorder_backend: argus_online_sync

    online_sync:
      enabled: true
      sync_source: sof_tsc_ns
      tolerance_ms: 1.0
      startup_full_clusters: 30
      frame_timeout_ms: 1000
      missing_frame_policy: fail_episode
      stop_mode: full_cluster

      frame_bus_dir: /dev/shm/lerobot_online_sync
      frame_bus_every_n: 1
```

参数含义：

- `frame_bus_dir`: 为空表示关闭；建议使用 `/dev/shm/...`。
- `frame_bus_every_n`: 每 N 个 accepted full cluster 发布一次。`1` 表示每帧发布。

注意：发布 raw NV12 会多一次 CPU/内存拷贝。纯采集时保持关闭；在线推理时再打开。

## Python 使用方式

```python
from tools.thor.gmsl2.online_sync_frame_client import ThorOnlineSyncFrameClient

client = ThorOnlineSyncFrameClient(
    root="/dev/shm/lerobot_online_sync",
    cameras=["cam_03", "cam_06", "cam_07", "cam_08"],
)

cluster = client.get_latest(timeout_s=0.1)
if cluster is None:
    raise RuntimeError("no synchronized camera cluster available")

print(cluster.logical_frame_index)
print(cluster.sync_timestamp_ns)
print(cluster.max_delta_ns)

frames = {
    name: frame.read_nv12()
    for name, frame in cluster.frames.items()
}
```

如果环境安装了 `numpy` 和 `cv2`，可以直接转 RGB：

```python
images = {
    name: frame.as_rgb()
    for name, frame in cluster.frames.items()
}
```

`as_rgb()` 返回 `height x width x 3` 的 RGB numpy array。

## cluster metadata

`latest_cluster.json` 示例：

```json
{
  "version": 1,
  "publish_seq": 7,
  "slot": 1,
  "recording": true,
  "episode_index": 3,
  "logical_frame_index": 360,
  "sync_source": "sof_tsc_ns",
  "format": "nv12",
  "width": 1920,
  "height": 1080,
  "min_sof_tsc_ns": 123456789000,
  "max_sof_tsc_ns": 123456795000,
  "max_delta_ns": 6000,
  "cameras": {
    "cam_03": {
      "path": "/dev/shm/lerobot_online_sync/slot1_cam_03.nv12",
      "camera": "cam_03",
      "logical_frame_index": 360,
      "local_frame_number": 1234,
      "sensor_timestamp_ns": 123456780000,
      "sof_tsc_ns": 123456789000,
      "eof_tsc_ns": 123456800000,
      "internal_frame_count": 1234
    }
  }
}
```

关键字段：

- `publish_seq`: frame bus 发布序号，读端可用于判断是否拿到新 cluster。
- `recording`: 当前 cluster 是否来自 episode recording 窗口。
- `logical_frame_index`: recorder 内部同步帧序号。
- `sync_source`: 当前为 `sof_tsc_ns`。
- `max_delta_ns`: 本 cluster 内各路 SOF 最大差值。
- `path`: 对应相机的 NV12 raw 文件。

## 推荐推理循环

在线控制建议使用 latest-frame 语义，模型慢时跳过旧帧，不反压 recorder：

```python
client = ThorOnlineSyncFrameClient("/dev/shm/lerobot_online_sync")
last_seq = None

while running:
    cluster = client.get_latest(
        timeout_s=0.05,
        min_publish_seq=None if last_seq is None else last_seq + 1,
    )
    if cluster is None:
        continue

    last_seq = cluster.publish_seq
    images = {name: frame.as_rgb() for name, frame in cluster.frames.items()}
    action = policy.infer(images, timestamp_ns=cluster.sync_timestamp_ns)
    controller.send(action)
```

## 与录制保存的关系

frame bus 和 episode 保存解耦：

- 保存视频仍由 `argus_online_sync` 的 encoder/mux 路径负责。
- frame bus 只发布最新同步 cluster，默认 drop-old。
- 模型读得慢不会让 recorder 等模型。
- episode 保存成功与否仍以 `online_sync_manifest.json` 为准。

录制窗口内的同步合同不变：

```text
只有 full SOF cluster 才能进入 encoder；
只有同一个 full SOF cluster 才会发布给模型；
如果 recording 窗口内缺 cluster，episode fail，不补帧。
```

## 为什么不要自己开相机

错误方式：

```text
模型进程 -> nvarguscamerasrc / Argus
recorder -> argus_online_sync
```

风险：

- 两边抢 Argus/camera 资源。
- 可能扰动 8 路相机栈。
- 模型看到的帧和 recorder logical frame 不一定一致。
- 应用层仍拿不到同一 raw frame 对应的同步 metadata。

正确方式：

```text
recorder 统一采集和同步；
模型只读 recorder 发布的 latest full cluster。
```

## 当前限制和后续升级

当前 frame bus 是 tmpfs raw NV12 文件接口，优点是简单、易调试、Python 易接入；代价是每次发布会多一次内存拷贝。

如果后续模型需要稳定 60 Hz 全 8 路输入，建议继续升级为：

- CUDA/DMABUF zero-copy IPC；
- 或共享内存 ring buffer；
- 或由 recorder 直接提供缩放后的推理尺寸。

如果森云后续提供真正的 PWM trigger timestamp 或 trigger frame id，应把
`sync_source` 从 `sof_tsc_ns` 升级为厂商硬件时间/帧号；frame bus 的 JSON
结构可以保持不变，只替换同步字段来源。

