# SG16A_AGTH_G3Y_A1 / MAX96726 链路 locked 但 STREAMON 失败

## 问题摘要

我们在 Jetson Thor 上使用 SENSING SG16A_AGTH_G3Y_A1 板卡和 AR0234 GMSL2 相机。按照供应商建议读取 MAX96726 的 link-lock 寄存器后，可以看到预期的多条 GMSL 物理链路处于 locked 状态，但其中多条 locked 链路在启动 V4L2 streaming 时失败。

目前现象更像是 MAX96726 / AR0234 驱动、I2C alias、解串器路由或 link-to-video 映射问题，而不是上层 LeRobot 录制程序问题。

## 环境信息

```text
Jetson 平台: Thor
JetPack / L4T: JetPack 7.0 / L4T R38.2.x
板卡: SG16A_AGTH_G3Y_A1
相机: SG2-AR0234C-G2F, 1920x1080 @ 60 fps, RAW10
解串器: MAX96726
解串器 I2C 地址: 0x33
驱动模块: max96726.ko, sg2-ar0234c-g2f.ko
设备树配置: SG16A_AGTH_G3Y_A1 AR0234Cx16
```

## 干净重载流程

每次验证前，我们都会停止 Argus，重新加载相机模块，并把相机设为 free-run 模式，以排除外部触发线的影响，同时清掉 bypass mode：

```bash
sudo service nvargus-daemon stop
cd ~/lerobot
sudo ./tools/thor/gmsl2/setup_sync.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1 --fps 60 --num 16

for i in $(seq 0 15); do
  sudo v4l2-ctl -d /dev/video$i -c sensor_mode=0,trig_mode=0,bypass_mode=0
done
```

`setup_sync.sh` 会重新加载 `max96726.ko` 和 `sg2-ar0234c-g2f.ko`，提升 VI/ISP/NVCSI/EMC 时钟，并应用每路相机控制项。

## Link-Lock 寄存器检测方法

按照供应商建议，读取 MAX96726 寄存器 `0x0008`：

```bash
sudo i2ctransfer -f -y <bus> w2@0x33 0x00 0x08 r1
```

寄存器 bit 含义：

```text
bit0 = LOCKED_A
bit1 = LOCKED_B
bit2 = LOCKED_C
bit3 = LOCKED_D
```

当前采用的 bus 到 video 映射：

```text
bus17 -> video0..video3
bus18 -> video4..video7
bus19 -> video8..video11
bus20 -> video12..video15
```

请供应商确认：在当前 AR0234Cx16 dtbo 下，MAX96726 `REG0x0008` 的 A/B/C/D 是否严格对应每组 `video base+0..base+3`。

## 基线 Link-Lock 结果

11 路相机接入时，干净重载后读取到：

```text
bus17 max96726@0x33 REG0x0008: 0x0b
bus18 max96726@0x33 REG0x0008: 0x07
bus19 max96726@0x33 REG0x0008: 0x0e
bus20 max96726@0x33 REG0x0008: 0x0c
```

按上述映射解码后，locked video IDs 为：

```text
video0, video1, video3,
video4, video5, video6,
video9, video10, video11,
video14, video15
```

unlocked video IDs 为：

```text
video2, video7, video8, video12, video13
```

## Stream 测试方法

我们只对 locked video IDs 逐路顺序测试，并使用 free-run 模式：

```bash
sudo v4l2-ctl -d /dev/videoN -c sensor_mode=0,trig_mode=0,bypass_mode=0

timeout 8s v4l2-ctl -d /dev/videoN \
  --set-fmt-video=width=1920,height=1080,pixelformat=BA10 \
  --stream-mmap=3 --stream-count=30 --stream-to=/dev/null
```

能正常 stream 的通道，H.265 录制使用 Jetson Multimedia API 的
`10_argus_camera_recording` sample。JetPack 7 当前环境里没有
`nvarguscamerasrc` / `nvvidconv` / `nvv4l2h265enc` 这些 GStreamer 元件，
所以没有使用 `nvarguscamerasrc` 拉流。

```bash
/tmp/jetson_multimedia_api_gmsl2/samples/10_argus_camera_recording/argus_camera_recording \
  -i <camera_index> \
  -r 1920x1080 \
  -t H265 \
  -d 3 \
  -f /tmp/gmsl2_argus_h265_records/camXX_1080p60_isp_h265_3s.h265
```

裸 H.265 后处理为 MP4 的命令：

```bash
ffmpeg -y -r 60 \
  -i tools/thor/gmsl2/records/camXX_1080p60_isp_h265_3s.h265 \
  -c copy -tag:v hvc1 \
  tools/thor/gmsl2/records/camXX_1080p60_isp_h265_3s.mp4
```

## 基线 Stream 测试结果

```text
OK:   video0, video1, video4, video9
FAIL: video3, video5, video6, video10, video11, video14, video15
```

locked 但失败的通道报错一致：

```text
VIDIOC_STREAMON returned -1 (Operation not permitted)
```

stream 测试后再次读取 link-lock，寄存器状态仍保持稳定：

```text
bus17 0x0b
bus18 0x07
bus19 0x0e
bus20 0x0c
```

## 将已知可工作的相机从 video1 位置移动到 video7 位置

为了判断问题是否跟随相机本体，我们把之前在 `video1` 位置可工作的相机移动到了物理 `video7` 位置。

干净重载后，link-lock 状态按预期变化：

```text
bus17 0x09 -> video1 掉锁
bus18 0x0b -> video7 上锁
bus19 0x0e
bus20 0x0c
```

解码结果：

```text
LOCKED_VIDEO_IDS=0,3,4,5,7,9,10,11,14,15
UNLOCKED_VIDEO_IDS=1,2,6,8,12,13
```

这说明移动过去的相机/线缆可以在 `video7` 建立 GMSL link。

移动后的 stream 测试：

```text
video7: FAIL, VIDIOC_STREAMON returned -1
video1: FAIL, old position is now empty
video0: OK, 30 frames
video4: FAIL, VIDIOC_STREAMON returned -1
video9: OK, 30 frames
video3: FAIL, VIDIOC_STREAMON returned -1
video5: FAIL, VIDIOC_STREAMON returned -1
```

这降低了“相机模块本体坏”的可能性。失败更像是留在目标 link/port 或驱动路径上，而不是跟随相机移动。

## dmesg 证据

已知可工作相机移动到 `video7` 后，对应日志如下：

```text
ar0234c 18-0023: dser_link_check link:0x0b
ar0234c 18-0023: +++> Der-1 port 7 camera ar0234c-7 been detected!
max96726 18-0033: i2c-w, write failed
ar0234c 18-0023: Error turning on streaming
tegra-camrtc-capture-vi tegra-capture-vi: uncorr_err: request timed out after 2500 ms
```

probe/init 阶段也能看到 I2C 写失败：

```text
max96726 18-0033: i2c-w16 failed: slave=0x46 reg=0x301a val=0x00d9 err=-121
ar0234c 18-0023: write16 table failed: source=0x46 reg=0x301a val=0x00d9 ret=-121
ar0234c 18-0023: sensor_init failed
```

其它 locked 但 stream-on 失败的通道也有类似日志：

```text
max96726 xx-0033: i2c-w, write failed
ar0234c xx-002x: Error turning on streaming
```

## Jetson 重启后的复测结果

用户重启 Jetson 后，我们再次执行“停止 Argus、重载模块、free-run、只测
locked IDs”的流程。重载成功。

重启后读取到的 link-lock 状态为：

```text
bus17 0x0d -> video0, video2, video3 locked
bus18 0x0b -> video4, video5, video7 locked
bus19 0x0e -> video9, video10, video11 locked
bus20 0x0c -> video14, video15 locked

LOCKED_VIDEO_IDS=0,2,3,4,5,7,9,10,11,14,15
UNLOCKED_VIDEO_IDS=1,6,8,12,13
```

只对 locked IDs 做 stream-on 测试，结果为：

```text
OK:   video0, video2, video9
FAIL: video3, video4, video5, video7, video10, video11, video14, video15
```

失败形式仍然一致：

```text
VIDIOC_STREAMON returned -1 (Operation not permitted)
```

stream 测试后 link-lock 状态仍保持：

```text
LOCKED_VIDEO_IDS=0,2,3,4,5,7,9,10,11,14,15
UNLOCKED_VIDEO_IDS=1,6,8,12,13
```

重启后的 dmesg 仍显示相同类型的问题：

```text
max96726 xx-0033: i2c-w, write failed
ar0234c xx-002x: Error turning on streaming
tegra-camrtc-capture-vi tegra-capture-vi: uncorr_err: request timed out after 2500 ms
```

因此，重启 Jetson 并不能消除问题。具体 locked ID 集合和少数能工作的
video ID 会随当前接线/重载状态变化，但核心问题稳定复现：多条 MAX96726
链路已经 locked，却无法完成 V4L2 STREAMON。

## 当前判断

我们已经确认：

1. `/dev/video*` 节点存在，并声明 AR0234 1920x1080 @ 60 fps。
2. MAX96726 `REG0x0008` 显示预期的物理 GMSL 链路可以 locked。
3. 部分通道可以正常 60 fps 出流，基线中 `video0`、`video1`、`video4`、`video9` 可工作。
4. 已知可工作的相机从 `video1` 移到 `video7` 后，`video7` 可以 link-lock，但仍然在 STREAMON 失败。
5. 测试使用 free-run 模式，因此外部触发线缺失不是本次 STREAMON 失败原因。

剩余问题可以表述为：

> 多条 GMSL 链路已经 locked，但供应商驱动在这些链路上无法完成 sensor/deserializer stream-on，并伴随 MAX96726 I2C 写失败。

## 给 SENSING 的问题

1. 请确认在 AR0234Cx16 dtbo 下，bus17/18/19/20 上 MAX96726 `REG0x0008` 的 A/B/C/D bit 到 `/dev/videoN` 的准确映射。
2. 为什么以下已经 locked 的链路会在 `VIDIOC_STREAMON` 失败：`video3, video5, video6, video10, video11, video14, video15`？
3. 在 `video1 -> video7` 的换位测试中，为什么 `video7` 可以 link-lock，但随后对 alias `0x46` 的 I2C 写失败？
4. 这是否是 JetPack 7 / L4T R38.2 下 `max96726.ko` 或 `sg2-ar0234c-g2f.ko` 已知的 I2C alias、serializer routing 或 stream-on sequencing 问题？
5. 是否有适用于该板卡和 AR0234 相机组合的更新驱动包或 patch？
6. 针对单条 locked 但失败的链路，例如 `video7`，供应商推荐的隔离验证流程是什么？是否应该断电后只接这一路相机，再重载模块单独测试？

## 剩余本地自查

我们认为剩余值得做的本地自查只剩两个窄范围确认：

1. 把同一颗相机从 `video7` 接回 `video1`。如果 `video1` 再次可以 stream，则基本排除相机模块本体问题。
2. 断电后只接一条失败但 locked 的链路，例如只接 `video7`，重载模块后单独测试。

如果这些结果仍然一致，后续需要供应商提供映射确认、驱动解释或驱动更新。

## 2026-05-21 补充材料

已按供应商要求补充“加载驱动到打开 video7 节点”的完整日志：

```text
tools/thor/gmsl2/records/debug_logs/dmesg_load_to_video7.log
tools/thor/gmsl2/records/debug_logs/video7_open_stream.log
tools/thor/gmsl2/records/debug_logs/setup_sync.log
tools/thor/gmsl2/records/debug_logs/link_locks_before_video7.log
```

`video7` 当前识别为 `vi-output, ar0234c 18-0023`，1920x1080 BA10 60 fps，
`sensor_mode=0,trig_mode=0,bypass_mode=0`，打开节点后 stream-on 仍失败：

```text
VIDIOC_STREAMON returned -1 (Operation not permitted)
```

另外按建议临时清空 `/var/nvidia/nvcam/settings/` 后重新录制 `cam00`，
随后已恢复原 settings，`camera_overrides.isp` 仍指向驱动 repo：

```text
/home/nvidia/Desktop/SG16A_AGTH_G3Y_A1/camera_overrides.isp
```

对比文件：

```text
tools/thor/gmsl2/records/cam00_1080p60_isp_h265_3s.mp4
tools/thor/gmsl2/records/cam00_1080p60_no_nvcam_settings_h265_3s.mp4
tools/thor/gmsl2/records/cam00_isp_frame.jpg
tools/thor/gmsl2/records/cam00_no_nvcam_settings_frame.jpg
```

本次样本中，删除 nvcam settings 前后画面颜色没有肉眼明显变化。
请供应商确认 JetPack 7 / Argus 下该 ISP 文件的实际加载路径、缓存机制，以及这种现象是否符合预期。

## 2026-05-21 给供应商的回复要点

1. 失败复测的拉流命令是 `v4l2-ctl`，格式为 1920x1080 `BA10`，free-run：
   `sensor_mode=0,trig_mode=0,bypass_mode=0`，然后
   `--stream-mmap=3 --stream-count=30 --stream-to=/dev/null`。
2. 能出流通道的 H.265 文件是用 Jetson Multimedia API 的
   `argus_camera_recording -i <camera_index> -r 1920x1080 -t H265 -d 3` 录制，
   再用 `ffmpeg -r 60 -c copy -tag:v hvc1` 封装为 MP4。
3. ISP 文件当前在 Thor 上为：
   `/var/nvidia/nvcam/settings/camera_overrides.isp ->
   /home/nvidia/Desktop/SG16A_AGTH_G3Y_A1/camera_overrides.isp`。
   已按建议清空 settings 后录制一次 `cam00`，再恢复 settings；当前样本因为画面缺少鲜艳色块，
   删除前后没有看到明显颜色变化。
4. 后续如需更严谨验证 ISP 颜色，需要现场放置明亮红色/彩色目标后重新执行
   “清空 settings -> 无 ISP 录制 -> 清空 settings -> 放回 ISP -> 录制”的对比流程。
5. 转接板供电规格需要现场确认电源适配器标签和接线，软件侧无法可靠读出是否为 12V3A。
6. 当前驱动包已打包：
   `tools/thor/gmsl2/SG16A_AGTH_G3Y_A1_20260521.tar.gz`，
   SHA256:
   `e0d4b34d9f1514101ca5243f3d05afef7b0536c1b7afddceb8fe1db3d8d88716`。

## 2026-05-21 红色块 ISP 对比复测

已按建议在相机 `0..3` 前放置红色块后重新录制。流程为：

1. 使用当前 `/var/nvidia/nvcam/settings/` 录制 `camera_index=0..3`
2. 备份并清空 `/var/nvidia/nvcam/settings/`
3. 重启 `nvargus-daemon`
4. 再录制 `camera_index=0..3`
5. 恢复原 settings

恢复后已确认：

```text
/var/nvidia/nvcam/settings/camera_overrides.isp
-> /home/nvidia/Desktop/SG16A_AGTH_G3Y_A1/camera_overrides.isp
```

本轮只有 `camera_index=0` 在 ISP 和 no-settings 两种状态下均录制成功。
`camera_index=1/2/3` 使用 `argus_camera_recording` 均出现段错误或 Argus
连接失败，日志已保留。

落盘目录：

```text
tools/thor/gmsl2/records/gmsl2_red_isp_compare_20260521_060346/
```

对比文件：

```text
isp/cam00_red_isp_h265_3s.mp4
no_nvcam_settings/cam00_red_no_nvcam_settings_h265_3s.mp4
isp/cam00_red_isp_frame.jpg
no_nvcam_settings/cam00_red_no_nvcam_settings_frame.jpg
logs/*.log
```

`cam00` 的红色块画面可见，但 ISP 与 no-settings 抽帧肉眼差异仍然很小。
请确认 JetPack 7 / Argus 下是否存在 settings 缓存机制，或者
`camera_overrides.isp` 是否实际应用到当前 AR0234 pipeline。

## 2026-05-21 转接板电源插拔后复测

用户插拔转接板电源后重新 clean reload 复测，link-lock 仍然成立：

```text
LOCKED_VIDEO_IDS=0,2,3,4,5,7,9,10,11,14,15
UNLOCKED_VIDEO_IDS=1,6,8,12,13
```

但本轮所有 locked IDs 的 V4L2 stream-on 均失败：

```text
video0, video2, video3, video4, video5, video7, video9, video10, video11, video14, video15
```

失败形式：

```text
VIDIOC_STREAMON returned -1 (Operation not permitted)
max96726 xx-0033: i2c-w, write failed
ar0234c xx-002x: Error turning on streaming
tegra-camrtc-capture-vi: uncorr_err: request timed out after 2500 ms
```

Argus `camera_index=0..3` 录制也全部段错误，输出文件均为 0 bytes：

```text
ARGUS_RC=139
```

完整日志已落盘：

```text
tools/thor/gmsl2/records/gmsl2_powercycle_retest_20260521_061404/
```

该结果说明电源插拔后问题没有缓解，当前状态反而从“少数通道可 stream”退化为
“所有 locked 通道均 stream-on 失败”。请重点协助确认转接板供电要求、
上电时序、MAX96726/serializer 复位流程，以及驱动在电源插拔后的初始化顺序。

## 2026-05-21 按供应商建议验证 Argus/GStreamer

供应商建议使用 Argus/GStreamer 路径：

```bash
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
  'video/x-raw(memory:NVMM),framerate=60/1,format=NV12' ! \
  nvvidconv ! xvimagesink
```

初始环境中 `nvarguscamerasrc`、`nvvidconv`、`nvv4l2h265enc` 均缺失。
已安装 `nvidia-l4t-gstreamer`，并确认安装后这些 GStreamer 元件存在。
安装过程同时将 L4T 包升级到 `38.2.2`，系统提示需要 reboot 生效。

在 SSH 无显示环境中，原始 `xvimagesink` 命令会先失败于：

```text
Could not open display (null)
```

因此增加了 `fakesink` 与 H.265 文件 sink 测试来排除显示问题：

```bash
gst-launch-1.0 -v nvarguscamerasrc sensor-id=0 num-buffers=180 ! \
  'video/x-raw(memory:NVMM),framerate=60/1,format=NV12,width=1920,height=1080' ! \
  fakesink sync=false

gst-launch-1.0 -e -v nvarguscamerasrc sensor-id=0 num-buffers=180 ! \
  'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=60/1,format=NV12' ! \
  nvv4l2h265enc bitrate=12000000 ! h265parse ! filesink location=sensor0_argus_gst.h265
```

两条 Argus/GStreamer 管线均报：

```text
gstnvarguscamerasrc.cpp, execute:803 No cameras available
```

H.265 输出为 0 bytes。Jetson Multimedia API 的
`10_argus_camera_recording -i 0` 也返回：

```text
main.cpp, execute:623 No cameras available
```

日志位置：

```text
tools/thor/gmsl2/records/gst_argus_after_install/
```

当前结论：已按建议切到 Argus/GStreamer 路径验证，但 Argus 当前枚举不到相机。
与此同时 MAX96726 link-lock 仍然成立。建议下一步重启 Thor 让 38.2.2 L4T 包
完全生效，再重新加载供应商驱动并复测 Argus/GStreamer。

## 2026-05-21 Thor 重启后 Argus/GStreamer locked IDs 复测

Thor 重启后，GStreamer 插件已经可用：

```text
nvarguscamerasrc=0
nvvidconv=0
nvv4l2h265enc=0
```

但重载供应商驱动失败：

```text
SETUP_RC=1
insmod /home/nvidia/Desktop/SG16A_AGTH_G3Y_A1/ko/sg2-ar0234c-g2f.ko
insmod: ERROR: could not insert module .../sg2-ar0234c-g2f.ko: Invalid parameters
```

dmesg 显示该 AR0234 驱动与当前 L4T 38.2.2 内核 tegracam 符号版本不匹配：

```text
sg2_ar0234c_g2f: disagrees about version of symbol tegracam_v4l2subdev_register
sg2_ar0234c_g2f: Unknown symbol tegracam_v4l2subdev_register (err -22)
sg2_ar0234c_g2f: disagrees about version of symbol tegracam_device_register
sg2_ar0234c_g2f: Unknown symbol tegracam_device_register (err -22)
```

因此 `/dev/video*` 没有生成，Argus 无相机可枚举。

MAX96726 lock register 仍能读取：

```text
LOCKED_VIDEO_IDS=0,2,3,4,5,7,9,10,11,14,15
UNLOCKED_VIDEO_IDS=1,6,8,12,13
```

对上述 locked IDs 执行：

```bash
gst-launch-1.0 -v nvarguscamerasrc sensor-id=<id> num-buffers=180 ! \
  'video/x-raw(memory:NVMM),framerate=60/1,format=NV12,width=1920,height=1080' ! \
  fakesink sync=false
```

全部失败，典型日志为：

```text
gstnvarguscamerasrc.cpp, execute:803 No cameras available
```

完整日志：

```text
tools/thor/gmsl2/records/gmsl2_gst_locked_retest_20260521_065006/
```

请供应商提供适配当前 L4T 38.2.2 / kernel `6.8.12-tegra-38.2.2`
的 `sg2-ar0234c-g2f.ko`，或明确该驱动包应使用的 L4T/kernel 版本。

## 2026-05-21 系统恢复记录

已按“恢复到该驱动包对应的 L4T/内核版本”执行恢复：

1. 将 L4T 包从 38.2.2 downgrade 回 38.2.1。
2. 将已安装的 `nvidia-l4t-*` 包 hold 住，避免再次被 apt 自动升级。
3. 执行供应商 `install.sh`，恢复驱动包自带：
   - `/boot/Image`
   - `tegra-camera.ko`
   - `nvhost-nvcsi.ko`
   - AR0234Cx16 dtbo
4. 将 `/boot/extlinux/extlinux.conf` 默认启动项恢复为 `JetsonIO`，确保
   `tegra264-camera-ar0234cx16-overlay.dtbo` 生效。

当前恢复后状态：

```text
Linux upai-pro03 6.8.12-tegra #3 SMP PREEMPT Sat May 9 10:34:43 CST 2026
nvidia-l4t-core       38.2.1-20250910123945
nvidia-l4t-kernel     6.8.12-tegra-38.2.1-20250910123945
nvidia-l4t-camera     38.2.1-20250910123945
nvidia-l4t-gstreamer  38.2.1-20250910123945
DEFAULT JetsonIO
OVERLAYS /boot/tegra264-camera-ar0234cx16-overlay.dtbo
```

备份文件：

```text
/boot/Image.before_sensing_restore_20260521_065941
/boot/extlinux/extlinux.conf.before_sensing_restore_20260521_065941
/boot/extlinux/extlinux.conf.before_default_jetsonio_20260521_070237
```

恢复过程快照：

```text
tools/thor/gmsl2/records/restore_logs/restore_process_snapshot.log
```

恢复后最终验证：

```text
tools/thor/gmsl2/records/gmsl2_post_restore_final_verify_20260521_070505/
```

验证结果：

```text
SETUP_RC=0
/dev/video0..15 已生成
LOCKED_VIDEO_IDS=0,2,3,4,5,7,9,10,11,14,15
UNLOCKED_VIDEO_IDS=1,6,8,12,13
```

恢复后继续对 `sensor-id=0,2,9` 执行 Argus/GStreamer smoke test，仍失败：

```text
gstnvarguscamerasrc.cpp, execute:803 No cameras available
```

因此，38.2.2 ABI 不匹配问题已经排除/恢复；当前剩余问题仍是供应商驱动和
Argus 对这些 GMSL2 AR0234 相机的枚举/出流问题。

## 2026-05-21 恢复后重新完整复测

恢复到供应商定制内核和 JetsonIO AR0234 overlay 后，重新完整复测。

当前环境：

```text
Linux upai-pro03 6.8.12-tegra #3 SMP PREEMPT Sat May 9 10:34:43 CST 2026
DEFAULT JetsonIO
OVERLAYS /boot/tegra264-camera-ar0234cx16-overlay.dtbo
nvidia-l4t-core/camera/gstreamer/multimedia: 38.2.1, held
```

GStreamer 插件存在：

```text
nvarguscamerasrc=0
nvvidconv=0
nvv4l2h265enc=0
```

复测日志：

```text
tools/thor/gmsl2/records/gmsl2_rerun_after_restore_20260521_071200/
```

`setup_sync.sh` 成功，lock 状态为：

```text
LOCKED_VIDEO_IDS=0,2,3,4,5,7,9,10,11,14,15
UNLOCKED_VIDEO_IDS=1,6,8,12,13
```

V4L2 raw 30 帧 stream-on 对所有 locked IDs 均成功：

```text
video0, video2, video3, video4, video5, video7, video9, video10, video11, video14, video15
```

但 Argus/GStreamer 对全部 locked IDs 仍失败：

```text
gstnvarguscamerasrc.cpp, execute:803 No cameras available
```

当前结论：底层 GMSL link 和 V4L2 raw stream-on 已恢复并全部通过；剩余问题集中在
Argus/ISP 路径无法枚举这些 AR0234 RAW 相机。请供应商重点确认 `camera_overrides.isp`、
Argus sensor mode / camera provider 配置、device-tree Argus metadata，以及
AR0234 RAW camera 是否需要额外的 Argus 配置或 NITO/ISP 安装步骤。
