# Pika SDK

## 介绍

Pika SDK 是一款专为 Pika 系列设备设计的 Python 软件开发工具包，旨在提供简单易用且功能强大的编程接口。该 SDK 支持两种主要设备类型：Pika Sense 和 Pika Gripper，使开发者能够轻松地控制和访问这些设备的各项功能。

如果您在使用 Pika SDK 过程中遇到任何问题，或者有任何建议和反馈，请通过以下方式联系我们：

- GitHub Issues: https://github.com/agilexrobotics/pika_sdk/issues
- 电子邮件: support@agilex.ai

我们的技术团队将尽快回复您的问题，并提供必要的支持和帮助。

## 软件环境

- 架构：x86_64/arm64
- 操作系统：Ubuntu 20.04/22.04
- python 版本： ≤ 3.9

它应该可以在其他 Linux 环境中工作，但只有上面列出的环境会定期进行测试。

## 安装依赖项

```bash 
sudo apt update

sudo apt install cmake g++ git v4l-utils  libudev-dev pkg-config libgtk-3-dev build-essential zlib1g-dev libx11-dev libeigen3-dev freeglut3-dev liblapacke-dev libopenblas-dev libpcap-dev libatlas-base-dev libusb-1.0-0-dev pkg-config libglfw3-dev libssl-dev libglu1-mesa-dev python3-pip
```

## 安装 realsense sdk
如果系统中已安装了 realsense sdk，则跳过此步骤：

如何判断是否已经安装 realsense sdk：

```bash
realsense-viewer
```

如果有显示realsense  viewer，则表示已经安装，请跳过此步骤。

1、下载安装包

```bash
git clone https://github.com/IntelRealSense/librealsense
# 或者下载指定版本
git clone https://github.com/IntelRealSense/librealsense/releases/tag/vxxx   # 注意：请将 vxxx 替换为实际版本号
```

2、编译安装

```bash
cd librealsense
mkdir build 
cd build
cmake .. -DCHECK_FOR_UPDATES=false
make -j$(nproc)
sudo make install
```

## 安装 libsurvive

```bash
git clone https://github.com/cntools/libsurvive.git
cd libsurvive
sudo cp ./useful_files/81-vive.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
make
```

## 安装 SDK

### pip3 安装

```bash
pip3 install agx-pypika
```

安装过程中 `Building wheel for wxpython (setup.py) ...`  在编译 `wxPython` 这步需要耗时很长，请耐心等待。

### 源码安装

首先，您需要获取 SDK 源代码。您可以通过克隆 GitHub 仓库或直接下载压缩包的方式获取代码。如果您选择使用 Git，可以在终端中执行以下命令：

```bash
git clone https://github.com/agilexrobotics/pika_sdk.git
cd pika_sdk
```

接下来，安装 SDK 所需的基本依赖库。这些库是 SDK 正常运行的必要组件：

```bash
pip3 install -r requirements.txt  
```

如果下载速度慢的话，可以使用清华源进行安装

```bash
pip3 install -r requirements.txt  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple 
```

最后，通过以下命令安装 SDK 本身。使用 `-e` 参数可以以开发模式安装，这样您对源代码的修改会立即生效，无需重新安装：

```bash
pip3 install -e .
```

### 测试

安装完成后，您可以运行 SDK 自带的测试脚本，验证安装是否成功：

```bash
python3 test_sdk.py
```

如果测试脚本输出成功信息，说明 SDK 已正确安装，您可以开始使用它来开发应用了。

## 多设备自动检测与绑定

Pika SDK 提供了多设备自动检测与绑定工具，可以帮助您在有多个 Pika 设备连接到系统时，自动检测并绑定设备。详细使用说明请参考 `tools/多设备自动检测与绑定工具使用说明.md` 文件。

开始运行代码之前，请先进行设备的检测与绑定，如没做好绑定，则会出现设备打开失败的错误。

运行设备检测绑定工具：

```bash
cd pika_sdk
python3 tools/multi_device_detector.py
```

## 示例代码

SDK 提供了多个示例代码，位于 `examples` 目录下，包括：
- gripper：
  - `gripper_example.py`: 演示如何使用 Pika Gripper 设备的基本功能
  - `quickly_ctrl_gripper.py`: 演示如何控制夹爪
  - `quickly_open_camera.py`: 演示如何快速打开相机并保存图像
  - `quickly_set_zero_point.py`: 演示如何设置gripper零点(在夹爪行程不对时进行校准)

- sense：
  - `sense_example.py`: 演示如何使用 Pika Sense 设备的基本功能
  - `quickly_open_camera.py`: 演示如何快速打开相机并保存图像
  - `vive_tracker_example.py`: 获取pika sense的位姿信息

这些示例代码展示了 SDK 的基本用法和常见功能，可以作为您开发自己应用的参考。

## API 参考

详细使用说明请参考 `pika_sdk/Pika SDK API 文档.md` 文件。

## 错误处理

Pika SDK 使用 Python 的日志系统记录错误和警告信息，便于开发者进行调试和问题排查。默认情况下，日志级别设置为 INFO，记录基本的操作信息和错误。如果您需要更详细的日志信息，可以将日志级别设置为 DEBUG：

```python
import logging
logging.getLogger('pika').setLevel(logging.DEBUG)  # 设置为 DEBUG 级别以获取更详细的日志
```

SDK 中的大多数方法都会在发生错误时返回特定的错误码或 False 值，并在日志中记录详细的错误信息。建议在开发过程中密切关注日志输出，及时发现和解决问题。

对于常见的错误情况，如设备未连接、相机初始化失败等，SDK 会提供清晰的错误提示，并尽可能地进行优雅的错误处理，避免程序崩溃。

## 注意事项

在使用 Pika SDK 开发应用时，请注意以下几点：

1. 设备连接：使用前请确保 Pika 设备已正确连接到计算机，并且串口设备路径（如 '/dev/ttyUSB0'）正确。如果不确定设备路径，可以在 Linux 系统中使用 `ls /dev/ttyUSB*` 命令查看。

2. 资源释放：使用完毕后，务必调用 `disconnect()` 方法断开连接，释放资源。这对于相机等硬件资源尤为重要，否则可能导致资源泄漏或设备被锁定。

3. RealSense 相机：使用 RealSense 相机功能需要安装 pyrealsense2 库。如果您不需要深度相机功能，可以不安装此库，SDK 会自动降级只使用鱼眼相机。

4. Vive Tracker：使用 Vive Tracker 功能需要安装 pysurvive 库。确保已正确安装并配置 SteamVR 和基站设备。

5. 权限问题：如果遇到串口访问权限问题，可能需要将用户添加到 dialout 组：`sudo usermod -a -G dialout $USER`，添加后需要重新登录系统使权限生效。

6. 相机设备 ID：鱼眼相机的设备 ID 可能因系统配置和连接顺序而异。如果默认 ID 无法正确访问相机，请使用 `set_fisheye_camera_index()` 方法设置正确的设备 ID。

7. RealSense 序列号：RealSense 相机可以通过序列号进行唯一标识。如果系统连接了多个 RealSense 相机，请使用 `set_realsense_serial_number()` 方法指定要使用的相机。

8. 多线程安全：SDK 内部使用了线程锁保证数据访问的线程安全性，但在多线程应用中，仍需注意避免并发访问可能导致的问题。

9. 错误处理：在生产环境中，建议对所有可能失败的操作进行错误检查和异常处理，确保应用的稳定性和可靠性。

10. 性能考虑：处理图像和深度数据可能需要较高的计算资源，特别是在高分辨率和高帧率下。请根据您的应用需求和硬件条件，合理设置相机参数。

通过遵循以上注意事项，您可以更加顺利地使用 Pika SDK 开发各类应用，充分发挥 Pika 设备的功能和性能。



