# 在前端配置和迁移硬件

打开 GUI 的 **Device Manager → Hardware configuration**。型号目录来自后端，表单由驱动配置类生成。设备选型和连接参数保存在本地配置中，原来的示例 YAML 保留。

## 日常使用

1. 初次进入时，编辑器读取 gateway 当前使用的配置。也可以从 **Saved configurations** 选择已有配置，或通过 **Duplicate setup** 创建另一套组合。
2. 选择 **Robot model**，填写 IP 或串口、设备 ID。选择另一种机器人会清空之前的末端、安装参数和操控器，防止沿用不兼容的配置。
3. 选择 **End effector**。目录只显示注册为兼容的末端。Franka Hand 使用机械臂的 IP；Pika / DAS 使用各自的串口；Corenetic 使用其网络连接参数。
4. 选择操控器，填写设备参数。SO-100 / SO-101 使用对应 leader；FR3 使用笛卡尔操控器。
5. 通过 **Add camera** 添加相机，为每台相机选择型号、角色名称、序列号或设备路径、分辨率和采样率。角色名称就是数据集里的相机键，例如 `ee`、`side`；不要把两台物理相机绑定到同一个设备。
6. **Check parameters** 验证参数、型号兼容性、相机绑定、URDF 与 TCP 是否存在，并显示最终运行 YAML。这个操作不连接或移动硬件，也不代表已经通过真机验收。
7. **Save draft** 只保存草稿；**Save and apply** 保存并切换后续会话使用的配置。录制、遥操作、回放、策略运行、标定等任务活动期间，后端拒绝切换。
8. 切换成功后，进入 Teleoperation / Live Record 使用新配置。gateway 重启后会恢复最后应用的版本。**Use source configuration** 返回启动时指定的原始 YAML。

导入、导出使用 JSON 配置文件。导入会生成新配置 ID，因此不会覆盖已有配置。将文件带到另一台机器后，在前端更新 IP / port / serial 并检查参数即可。

## 搬迁与更换末端

- 只修改连接参数时，TCP、工作空间和 home 等安装参数保持不变。
- 换选末端型号时，编辑器切换到该末端的标准 URDF / TCP 预设，并清除前一个末端的 home 和工作空间覆盖值。
- 改变转接板、手指或设备相对位置后，在 **Mounting, TCP, workspace and home** 中更新对应参数及现场标定。连接地址与安装几何分别管理。
- 标准预设描述标准安装方式；已有数据和 checkpoint 是否适合当前安装仍需按其动作、工具坐标和相机契约判断。

## 当前功能范围

| 组合 | 前端实例配置 | 遥操作 / GUI 录制 | 训练视图 | GUI 仿真、回放和策略运行 |
| --- | --- | --- | --- | --- |
| FR3 + Pika | 支持 | 使用原有 Cartesian 管线 | 支持 | 使用现有 Pika 流程 |
| FR3 + Franka Hand / DAS / Corenetic | 支持 | 使用对应末端驱动 | 支持 | 当前组合尚未接通，相关动作由后端拒绝 |
| SO-100 / SO-101 + 对应 leader | 支持 | 使用原生关节动作和观测 | 使用 LeRobot 通用 CLI | FR3 专属功能不开放 |
| 已注册的其他 Robot | 自动生成表单 | 原生动作匹配时可复用；否则注册转换管线 | 根据其 schema 接入 | 按实现能力接入 |

相机目录包含 RealSense、Hikrobot、OpenCV 和 GMSL2。配置类可加载代表表单可用，设备 SDK、系统驱动及硬件仍需安装。SO 系列需要已完成电机配置和标定。

Thor 的专用多路 GMSL2 / BOX 采集流程继续保留原有自动发现和同步配置；这类配置在编辑器中明确标识。为机器人添加 GMSL2 相机实例使用 `Gmsl2CameraConfig`，不会将专用 Libargus 编码管线冒充为普通相机实例。

## 本地存储与接口

配置保存在 `outputs/hardware/<源配置路径的命名空间>/`：

- `<id>.json`：可编辑配置、生成的运行配置、版本摘要。
- `<id>.yaml`：应用时生成、由现有运行入口读取的 YAML。
- `active.json`：已经应用的完整版本。保存草稿不会改变它。

配置采用 revision 检查，旧浏览器不能静默覆盖其他人的编辑。不同启动源配置的本地选择相互独立。代码同步原有的 `outputs/` 排除规则会保留这些本地配置。

HTTP 接口：

| 接口 | 用途 |
| --- | --- |
| `GET /api/hardware` | 型号目录、当前配置、已保存配置、运行状态 |
| `GET /api/hardware/catalog` | 型号与字段描述 |
| `GET /api/hardware/setup?id=...` | 读取指定草稿 |
| `POST /api/hardware/validate` | 检查 `{document}` 并返回运行 YAML |
| `POST /api/hardware/save` | 保存 `{document, revision}` |
| `POST /api/hardware/activate` | 应用 `{id, revision}` |
| `POST /api/hardware/reset` | 恢复启动源配置 |

## 实现新硬件

机器人、相机和操控器继续实现已有的 `Robot` / `Camera` / `Teleoperator` 接口，使用现有配置注册及工厂。然后在可安装插件的 `__init__.py` 中注册一份 `HardwareModel`，前端不需要再增加对应型号的代码。

```python
from lerobot.hardware import HardwareModel, register_model

register_model(HardwareModel(
    kind="robot",
    type="my_robot",                 # 已由 RobotConfig 注册的 type
    label="My robot",
    config_class=MyRobotConfig,       # dataclass；作为表单和参数校验的依据
    exclude=("cameras",),
    primary=("ip", "port", "id"),   # 常用参数，其余参数放入高级设置
    capabilities=("record", "teleoperate"),
))
```

插件沿用 `lerobot_robot_*`、`lerobot_camera_*` 或 `lerobot_teleoperator_*` 包名前缀。配置类负责参数校验，驱动负责硬件访问；目录读取和参数检查不创建驱动实例。

驱动的原生组合字段与 FR3 不同时，可通过 `HardwareModel.assemble` 将组合后的机器人 / 末端 / 相机字典转换为自身 `RobotConfig` 的结构；返回字典须保留选定的 `type`。相机插件可声明 `identity_fields=("host", "port")` 等设备标识字段，用于检查重复绑定。

当操控器与机器人动作字段不同，在 `HardwareModel.recording_processors` 中提供 `(cfg, robot, teleop) -> (teleop_action_processor, robot_action_processor, robot_observation_processor)`。未提供转换时，通用录制入口要求双方动作字段一致。转换仍需由适配者保证单位、方向和控制含义一致。

FR3 的单标量夹爪可独立扩展：

```python
from dataclasses import dataclass
from lerobot.hardware import HardwareModel, register_model

@dataclass
class MyGripperConfig:
    port: str
    speed: float = 0.1

register_model(HardwareModel(
    kind="end_effector",
    type="my_gripper",
    label="My gripper",
    config_class=MyGripperConfig,
    compatible_robots=("franka_research3",),
    primary=("port",),
    robot_fields={
        "port": "gripper_config.port",
        "speed": "gripper_config.speed",
    },
    mounting={
        "urdf_path": "path/to/fr3_with_my_gripper.urdf",
        "target_frame_name": "my_tcp",
    },
    end_effector_factory=MyGripperDriver,
))
```

`MyGripperDriver(config)` 实现公开的 `EndEffectorDriver` 协议：`connect()`、`disconnect()`、`get_position()`、`set_position(value)`。这里的 position 是实测归一化开口，0 关闭、1 打开；命令 I/O 不应阻塞机械臂实时循环。灵巧手等多维动作通过自己的 Robot 适配器表达。

注册硬件描述后，配置表单、参数校验、保存和运行配置生成均复用公共代码。仿真资源和 FR3 专属任务不会因为注册了通信驱动而自动获得支持。
