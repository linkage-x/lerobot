# FR3 MuJoCo Sim Teleop Architecture

日期：2026-04-22

本文整理当前 `tools/fr3/fr3_mujoco_teleop.py` 对应的 FR3 仿真 teleop 运行架构，聚焦以下几部分：

- 入口脚本与运行时配置
- SpaceMouse 到 FR3 控制链
- 主线程 / 控制线程 / 连续物理线程的职责划分
- viewer 与 camera stream 的渲染路径
- `MUJOCO_GL` 的实际约束

## 1. 总体组件图

```mermaid
flowchart TD
    U["User and SpaceMouse"]
    C["Docker compose lerobot-fr3-sim-teleop"]
    E["tools fr3 fr3_mujoco_teleop py"]
    GL["configure mujoco gl backend"]
    T["SpaceMouseTeleop"]
    ENV["FR3MujocoEnv"]
    LOOP["run sim teleop loop"]
    CT["Control Thread _run_control_loop"]
    MT["Main Thread viewer and camera render"]
    PT["Continuous Physics Thread _continuous_physics_loop"]
    HTTP["HTTP Camera Stream localhost 18765"]
    PG["pygame 2x2 Preview"]
    VW["MuJoCo Passive Viewer"]
    B["Browser"]

    U --> T
    C --> E
    E --> GL
    E --> T
    E --> ENV
    E --> LOOP
    LOOP --> CT
    LOOP --> MT
    ENV --> PT
    MT --> VW
    MT --> HTTP
    MT --> PG
    HTTP --> B
```

## 2. 线程与职责划分

```mermaid
flowchart LR
    subgraph Main["Main Thread"]
        M1["env reset"]
        M2["launch passive viewer"]
        M3["create camera renderer"]
        M4["periodic viewer sync"]
        M5["periodic camera snapshot render"]
        M6["http latest jpeg update"]
        M7["pygame preview update"]
    end

    subgraph Control["Control Thread"]
        C1["teleop get action"]
        C2["env step teleop action"]
        C3["update shared latest info"]
        C4["maintain 200 Hz cadence"]
    end

    subgraph Physics["Continuous Physics Thread"]
        P1["apply continuous control tick"]
        P2["mj step"]
        P3["update tcp and visual state"]
    end

    subgraph Shared["Shared State"]
        S1["LatestTeleopInfo"]
        S2["env data and env model"]
        S3["viewer data snapshot"]
        S4["camera render data snapshot"]
        S5["LatestCameraFrame"]
    end

    C1 --> C2 --> C3 --> S1
    C2 --> S2
    P1 --> P2 --> P3 --> S2
    M4 --> S1
    M4 --> S3
    M5 --> S1
    M5 --> S4
    M5 --> S5
    M6 --> S5
    M7 --> S5
```

## 3. 控制链路

```mermaid
sequenceDiagram
    participant SM as SpaceMouseTeleop
    participant CT as Control Thread
    participant ENV as FR3MujocoEnv
    participant IK as MuJoCo IK / FK
    participant PHY as Continuous Physics Thread

    loop 200 Hz
        CT->>SM: get_action()
        SM-->>CT: teleop_action
        CT->>ENV: step teleop action without camera obs
        ENV->>IK: current pose to desired pose to target joints
        ENV->>ENV: update servo or otg targets
        ENV-->>CT: info with target pose tcp pose markers gripper
        CT->>CT: publish latest shared info
    end

    loop continuous_physics_frequency
        PHY->>ENV: apply continuous control tick
        ENV->>ENV: ctrl and gravity compensation
        ENV->>ENV: mj step and mj forward
    end
```

## 4. 渲染链路

```mermaid
sequenceDiagram
    participant MT as Main Thread
    participant ENV as FR3MujocoEnv
    participant VD as viewer_data
    participant CD as camera_render_data
    participant VR as MuJoCo Viewer
    participant CR as Camera Renderer
    participant HF as _LatestCameraFrame
    participant BW as Browser / pygame

    loop viewer refresh
        MT->>ENV: copy visual state for viewer
        MT->>VR: update markers and viewer sync
    end

    loop camera_fps
        MT->>ENV: copy visual state for camera render
        MT->>CR: update scene and render three cameras
        MT->>MT: build 2 by 2 rgb grid
        MT->>HF: store latest JPEG bytes
        HF-->>BW: browser and pygame consume latest frame
    end
```

## 5. FR3 MuJoCo Env 内部关系

```mermaid
flowchart TD
    A["step teleop action"]
    B["normalize teleop action"]
    C["compute desired pose from teleop"]
    D["inverse kinematics"]
    E["servo target joints or otg target joints"]
    F["build info without camera obs"]
    G["continuous physics loop"]
    H["apply continuous control tick locked"]
    I["mj step and mj forward"]
    J["copy visual state"]
    K["render"]

    A --> B --> C --> D --> E --> F
    G --> H --> I
    J --> I
    K --> J
```

## 6. 关键约束

- `tools/fr3/fr3_mujoco_teleop.py` 是入口，负责解析参数、构建 `SpaceMouseTeleop`、构建 `FR3MujocoEnv`、配置 `MUJOCO_GL`、启动 viewer 和 teleop loop。
- 当同时启用 `viewer` 与 `--enable-cameras` 时，运行时默认会把 `MUJOCO_GL` 从缺省值或 `egl` 切到 `glfw`。
- 控制与渲染已经解耦：
  - 控制线程只做 `get_action -> step_teleop_action`
  - 主线程独占 MuJoCo viewer 与 camera renderer
- camera stream 不再通过磁盘写图传递，而是维护内存中的最新 JPEG。
- viewer 和 camera renderer 都基于 `copy_visual_state()` 的快照工作，避免直接把长时间渲染锁持有在控制线程上。
- `FR3MujocoEnv` 仍保留 continuous physics 线程，用于持续推进 MuJoCo 物理和 servo 执行。

## 6.1 已验证的 SpaceMouse 旋转语义

- 在当前 `pika_task_tcp` 轴约定下，`pitch` 与 `yaw` 的 SpaceMouse 语义和末端行为一致。
- `roll` 的 SpaceMouse 原始正方向与 FR3 sim 末端 `wx` 语义相反。
- 因此，`tools/fr3/fr3_mujoco_teleop.py` 默认将 `--scale-wx` 设为 `-0.001944`，等价于显式传入：

```bash
python tools/fr3/fr3_mujoco_teleop.py --scale-wx=-0.001944
```

- 这个修正只反转 `wx`，不修改 `wy` / `wz` 的默认符号。

## 7. 当前推荐运行方式

```bash
docker compose -f docker/docker-compose.yml --profile sim --profile teleop run --rm \
  -e DISPLAY=$DISPLAY \
  -e PYTHONPATH=/workspace/src \
  lerobot-fr3-sim-teleop \
  python tools/fr3/fr3_mujoco_teleop.py
```

在当前仓库默认配置下，上述命令会走：

- `MUJOCO_GL=glfw`
- `viewer` 保留
- `camera_fps=30`
- `fps=200`
- `continuous_physics=True`
- `use_otg=False`

## 8. 代码落点

- 入口脚本：`tools/fr3/fr3_mujoco_teleop.py`
- teleop 主循环：`src/lerobot/envs/fr3_mujoco_teleop.py`
- FR3 MuJoCo 环境：`src/lerobot/envs/fr3_mujoco.py`
- SpaceMouse teleop：`src/lerobot/teleoperators/spacemouse/teleop_spacemouse.py`
