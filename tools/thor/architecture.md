# Thor 数据采集全局软件架构

本文以 `tools/thor/DEPLOYMENT.md` 为入口，并按当前代码实现梳理 Thor / Jetson
数据采集链路。核心系统是 `tools/data_collection_gui` 前端 + Python gateway，
由 gateway 启动 `tools/thor/gmsl2/thor_record.py`，再由 recorder 协调 11 路
GMSL2 相机、BOX 采集板、预览、自动恢复和数据集写入。

## 1. 运行时总览

```mermaid
flowchart LR
    operator[操作员浏览器<br/>Live Record / Device Manager / Replay / QC]
    vite[Vite 前端<br/>tools/data_collection_gui/frontend<br/>:5173]
    gateway[Python Gateway<br/>tools.data_collection_gui.gateway<br/>ThreadingHTTPServer :8765]

    subgraph Gateway["Gateway 进程"]
        api["HTTP API<br/>/api/snapshot<br/>/api/handheld/record/*<br/>/api/device-preview/*<br/>/api/replay/*<br/>/api/processing/*"]
        state[GatewayState<br/>设备状态 / recorder 状态<br/>replay / tasks / dataset cache]
        stdout_reader[recorder stdout reader<br/>只排空 pipe]
        stdout_consumer[recorder-output-consumer<br/>解析 recorder 输出并更新状态]
        dataset_refresher[dataset-stats-refresher<br/>离线扫描 outputs/datasets]
        idle_preview[Idle camera preview<br/>临时 gst-launch / cached JPEG]
    end

    recorder[Thor Recorder 进程<br/>tools/thor/gmsl2/thor_record.py]

    subgraph Recorder["Recorder 进程"]
        stdin_fsm[stdin 命令 FSM<br/>start / save / discard / quit / preview_demand]
        box_client[BoxClient<br/>tools/thor/box_sdk/box_client.py]
        pcs[PersistentCameraSession<br/>tools/thor/gmsl2/persistent_session.py]
        preview_ctl[recorder-owned preview controller<br/>按需 tee -> JPEG]
        lr3[Lr3Writer<br/>tools/thor/gmsl2/thor_lerobot_v3.py]
        auto_recover[auto_recover<br/>recover_argus.sh fallback]
    end

    subgraph Workers["每路相机一个 worker 子进程"]
        w0[worker cam_00<br/>persistent_session_worker.py]
        wN[worker cam_NN<br/>N x worker]
        gst[nvarguscamerasrc<br/>NV12 -> nvv4l2h265enc<br/>matroskamux / splitmuxsink]
    end

    subgraph Hardware["Thor 硬件 / 系统服务"]
        max96726[MAX96726 lock check<br/>check_max96726_locks.sh]
        argus[nvargus-daemon]
        pwm[PWM / v4l2 controls<br/>setup_sync.sh / pwm.sh]
        cams[SG2-AR0234C-G2F<br/>GMSL2 cameras]
        box_hw[BOX 采集板 MCU<br/>192.168.2.60:15000]
        box_net[BOX 网口持久化<br/>ensure_box_net.sh / systemd<br/>192.168.2.45/24]
    end

    subgraph Outputs["数据与日志"]
        episodes[outputs/datasets/.../episodes<br/>episode_NNNNNN/cam_XX.mkv<br/>episode_NNNNNN/meta.json]
        parquet[LeRobot v3 BOX tabular sidecar<br/>meta/info.json<br/>data/chunk-000/file-000.parquet]
        logs[outputs/logs/data_collection_gui<br/>recorder_YYYYMMDD_HHMMSS.log<br/>run/logs/gateway.log]
        previews["/tmp/thor_gmsl2_preview<br/>cam_XX.jpg"]
    end

    operator --> vite
    vite -->|fetch /api/*| api
    api --> gateway
    gateway --> state
    gateway -->|Popen + stdin pipe| recorder
    recorder -->|stdout lines| stdout_reader --> stdout_consumer --> state
    dataset_refresher --> state
    api -->|idle preview request| idle_preview
    api -->|recording preview request| previews
    api -->|preview_demand stdin| recorder

    recorder --> stdin_fsm
    recorder --> box_client
    recorder --> pcs
    recorder --> preview_ctl
    recorder --> lr3
    recorder --> auto_recover

    pcs -->|multiprocessing.Queue cmd/event| w0
    pcs -->|multiprocessing.Queue cmd/event| wN
    w0 --> gst
    wN --> gst
    gst --> argus --> cams
    gst --> episodes
    preview_ctl --> pcs
    w0 --> previews
    wN --> previews

    recorder --> max96726
    recorder --> pwm
    pwm --> cams
    auto_recover --> argus
    auto_recover --> pwm
    box_client -->|box_sdk wheel + UDP 15000| box_hw
    box_net --> box_hw
    box_client --> parquet
    box_client --> episodes
    lr3 --> parquet
    recorder --> logs
    gateway --> logs
```

## 2. 组件职责

| 层级 | 代码入口 | 职责 |
| --- | --- | --- |
| 前端 | `tools/data_collection_gui/frontend/src/App.tsx`, `api.ts` | 轮询 `/api/snapshot`，发起 Connect/Start/Save/Discard/Replay/QC 请求，显示设备、预览、日志和数据集状态。 |
| Gateway | `tools/data_collection_gui/gateway.py` | 单机 HTTP API、全局状态机、启动/停止 recorder、解析 recorder stdout、维护设备状态、缓存数据集扫描、提供相机/BOX 预览和 Replay/QC/任务导出入口。 |
| Recorder | `tools/thor/gmsl2/thor_record.py` | Thor 采集主编排：读 YAML、检测相机、配置硬同步、启动 BOX、连接持久相机 session、处理 stdin 命令、保存/丢弃 episode、写 meta 和 BOX v3 sidecar。 |
| GMSL2 session | `persistent_session.py`, `persistent_session_worker.py` | 每路相机一个 worker 子进程；父进程只做控制和错误聚合；worker 内持久 GStreamer pipeline，用 `splitmuxsink split-now` 切 episode。 |
| BOX 适配 | `tools/thor/box_sdk/box_client.py` | 包装 vendored `box_sdk` wheel，配置 UDP，轮询 `SensorCache`，解码 gripper/IMU/trigger/force/touch，支持 live preview 和 episode 样本记录。 |
| 数据写入 | `thor_lerobot_v3.py` | 将 BOX 快照/样本写成轻量 LeRobot v3 parquet；相机视频保持硬编码 MKV，不把像素拉进 Python。 |
| 恢复/部署 | `recover_argus.sh`, `setup_sync.sh`, `ensure_box_net.sh`, `run/deploy.sh` | Argus/模块/PWM 恢复，BOX 网口持久化，本地到 Thor 的 rsync + gateway 重启 + Vite 启动。 |

## 3. Connect 与录制时序

```mermaid
sequenceDiagram
    participant UI as "Frontend UI"
    participant GW as "gateway.py"
    participant REC as "thor_record.py"
    participant BC as "BoxClient and box_sdk"
    participant PCS as "PersistentCameraSession"
    participant W as "N camera workers"
    participant ARGUS as "nvargus-daemon"
    participant FS as "outputs datasets"

    UI->>GW: POST /api/handheld/record/connect
    GW->>GW: suspend idle camera previews
    GW->>REC: Popen python thor_record.py --config-path ... --skip-argus-probe
    REC->>REC: load YAML and detect MAX96726 locked sids
    REC->>REC: apply PWM and v4l2 controls when hardware sync is enabled
    REC->>BC: start UDP listener and poll loop
    BC-->>REC: live sensor rates and connected BOX devices
    REC->>PCS: connect persistent session
    PCS->>W: spawn one worker per active sid
    W->>ARGUS: nvarguscamerasrc -> PLAYING
    W-->>PCS: playing fragment and error events
    alt too few cameras or all workers fail
        REC->>PCS: disconnect partial session
        REC->>ARGUS: run recover_argus.sh via auto recover
        REC->>PCS: retry connect once
    end
    REC-->>GW: stdout Cameras Box devices Connected K pipelines
    GW->>GW: mark active devices, armed state
    GW-->>UI: snapshot

    UI->>GW: POST /api/handheld/record/start
    GW->>REC: stdin start command
    REC->>PCS: start_episode(episode_dir, idx)
    PCS->>W: cmd_q start_episode
    W->>W: splitmuxsink split-now opens EPISODE fragment
    REC->>BC: start_recording(t0_wall_s)
    loop while recording
        REC->>PCS: poll stream health
        REC->>BC: read live snapshot and record dedup samples
        REC-->>GW: stdout Recorded X frames
    end

    UI->>GW: POST /api/handheld/record/stop-save
    GW->>REC: stdin save command
    REC->>BC: stop_recording()
    REC->>PCS: stop_episode(handle)
    PCS->>W: cmd_q stop_episode
    W->>W: force IDR then split-now and wait fragment closed
    W-->>PCS: episode_done(fragment)
    REC->>FS: write episode meta.json and sensor_samples.json
    REC->>FS: append BOX LeRobot v3 parquet
    REC-->>GW: stdout Episode saved
    GW-->>UI: snapshot with saved episode count
```

## 4. GMSL2 worker 隔离模型

```mermaid
flowchart TB
    parent[PersistentCameraSession 父对象<br/>connect / start_episode / stop_episode / poll_errors]

    subgraph PerSid["每个 sensor-id 独立子进程"]
        cmdq[(cmd_q<br/>start_episode / stop_episode / disconnect / preview_on/off)]
        worker[persistent_session_worker.run_worker]
        evtq[(evt_q<br/>playing / fragment / error / eos / episode_done)]
        pipeline[GStreamer pipeline<br/>nvarguscamerasrc sensor-id=N<br/>tee<br/>queue -> nvv4l2h265enc -> matroskamux -> splitmuxsink<br/>optional preview tee -> jpeg appsink]
    end

    warmup[_warmup fragments<br/>cam_NN_warmup_*.mkv]
    ep[episode_NNNNNN/cam_NN.mkv]
    jpg["/tmp/thor_gmsl2_preview/cam_NN.jpg"]

    parent --> cmdq --> worker --> pipeline
    pipeline -->|format-location-full WARMUP| warmup
    pipeline -->|format-location-full EPISODE| ep
    pipeline -->|new-sample preview| jpg
    pipeline -->|bus ERROR/EOS + fragment events| evtq --> parent

    parent -->|partial failure: drop failed sid| parent
    parent -->|retry: terminate old worker and respawn| worker
```

设计要点：

- GStreamer pipeline 在 Connect 时创建一次，episode 之间保持 warmup 状态，避免每个 StartEpisode 重建 11 路 `nvarguscamerasrc`。
- 一路相机一个 worker，避免单进程持有多个 Argus `CaptureSession` 时一处 `set_state(PLAYING)` 死锁拖死全部相机。
- `splitmuxsink` 的 `format-location-full` 决定 warmup 或 episode 文件名；Stop 时等待 `splitmuxsink-fragment-closed`，减少尾部丢帧和空目录。
- Connect 支持 partial failure：失败 sid 从 active stream 中移除，成功相机继续录制；全部失败才让 recorder 退出。
- recorder-owned preview 是录制 pipeline 的 tee 分支，按 `preview_demand` 心跳启停，避免 Device Manager 再打开第二个 Argus client 抢同一 sensor。

## 5. 数据输出结构

```mermaid
flowchart TB
    root[dataset.root<br/>outputs/datasets/thor_gmsl2_11ch_v1]

    root --> episodes[episodes/]
    episodes --> epdir[episode_000000/]
    epdir --> mkv[cam_00.mkv ... cam_NN.mkv<br/>硬件 H.265 MKV]
    epdir --> meta[meta.json<br/>相机列表 / sync 设置 / split timing / BOX 快照]
    epdir --> samples[sensor_samples.json<br/>BOX dedup per-sensor samples]

    root --> warmup[_warmup/<br/>rolling warmup fragments<br/>保留少量最近文件]
    root --> meta_dir[meta/]
    meta_dir --> info[info.json<br/>LeRobot v3 feature schema]
    meta_dir --> tasks[tasks.parquet]
    meta_dir --> episodes_parquet[episodes/chunk-000/file-000.parquet]
    root --> data[data/chunk-000/file-000.parquet<br/>BOX observation.state/action rows]
    root --> processing[meta/processing.json<br/>QC / processing 状态]
    root --> replay_val[meta/gui_replay_validations.json<br/>MuJoCo validation 缓存]
```

数据契约：

- 相机视频为每路每 episode 一个 MKV，像素不进入 Python；Python 只控制 pipeline 和写元数据。
- `meta.json` 记录 active camera、locked/failed sid、硬同步、split 时间、PTS offset、BOX 快照等采集上下文。
- `Lr3Writer` 将 BOX 状态压成 LeRobot v3 tabular sidecar，供 GUI replay/QC 读取 `observation.state` / `action`。
- gateway 的数据集扫描在后台线程做，`/api/snapshot` 只读缓存，避免大目录扫描卡住 recorder stdout 和 preview 请求。

## 6. 部署与恢复拓扑

```mermaid
flowchart LR
    dev[开发机 repo<br/>/home/hanyu/Codes/lerobot]
    sync[run/sync_to_thor.sh<br/>rsync --delete<br/>排除 .git/node_modules/outputs/run logs]
    deploy[run/deploy.sh]
    thor[Thor / Jetson<br/>nvidia@192.168.111.122:~/lerobot]
    gateway_run[gateway on Thor<br/>python -m tools.data_collection_gui.gateway<br/>--host 0.0.0.0 --port 8765]
    vite_local[Vite on dev host<br/>localhost:5173]
    browser[Browser]

    box_setup[BOX runtime setup<br/>install_compat_links.sh<br/>setup_env.sh<br/>ensure_box_net.sh]
    gmsl_setup[GMSL2 runtime setup<br/>setup_sync.sh / pwm.sh<br/>recover_argus.sh]
    deps[System deps<br/>apt / pyarrow / box_sdk wheel / npm]

    dev --> sync --> thor
    deploy --> sync
    deploy -->|ssh restart| gateway_run
    deploy --> vite_local
    browser --> vite_local -->|proxy /api| gateway_run
    thor --> box_setup
    thor --> gmsl_setup
    thor --> deps
```

部署清单中的关键运行约束：

- Gateway 启动前需要 `ensure_box_net.sh` 保证 `enP2p1s0` 上有 `192.168.2.45/24`，BOX MCU 固定向 `192.168.2.45:15000` 推送 UDP。
- BOX wheel 的 native `.so` 依赖旧 soname，首次部署需跑 `install_compat_links.sh` 并安装 Boost / URDF / tinyxml2 兼容依赖。
- `BoxClient.start()` 会自动设置 `BOX_SDK_URDF`；`LD_LIBRARY_PATH` 仍由 `setup_env.sh` 或启动 wrapper 提供。
- GMSL2 硬同步依赖 passwordless `sudo` 调 `pwm.sh` / `v4l2-ctl`；`auto_recover` 会在 Connect 失败或成功率过低时调用 `recover_argus.sh`。
- SSH 后台启动 gateway 使用 `setsid ... </dev/null >log 2>&1 &; disown`，避免 SSH 退出导致进程收到异常退出。

## 7. 主要 API 与状态流

```mermaid
flowchart TB
    api_snapshot["GET /api/snapshot"]
    api_connect["POST /api/handheld/record/connect"]
    api_start["POST /api/handheld/record/start"]
    api_save["POST /api/handheld/record/stop-save"]
    api_discard["POST /api/handheld/record/stop-discard"]
    api_exit["POST /api/handheld/record/exit"]
    api_camera["GET /api/device-preview/camera.jpg"]
    api_box["GET /api/device-preview/box"]
    api_replay["POST/GET /api/replay/*"]
    api_processing["POST /api/processing/qc<br/>/api/processing/traj-gen<br/>/api/tasks/export"]

    state[GatewayState]
    recorder[Recorder subprocess]
    dataset[Dataset cache / filesystem]
    preview[Idle or recorder-owned JPEG preview]

    api_snapshot --> state
    api_connect -->|Popen| recorder --> state
    api_start -->|stdin newline| recorder
    api_save -->|stdin save| recorder
    api_discard -->|stdin n| recorder
    api_exit -->|stdin exit/q| recorder
    api_camera --> preview
    api_camera -->|preview_demand if recorder owns cameras| recorder
    api_box --> state
    api_replay --> dataset
    api_processing --> dataset
    dataset --> state
```

状态更新不是靠 recorder RPC 返回，而是靠 stdout 协议：

- `Cameras: ...` / `Cameras (active): ...` 更新相机设备状态。
- `Box devices: ...` / `Box rates: ...` 更新 BOX 设备状态。
- `Connected K pipelines...` 将 recording 状态推进到可 start。
- `Recorded X frames...` 更新进度。
- `WARNING:` / `ERROR:` 进入前端日志并标记失败设备。
- `Episode saved.` / `Episode discarded` 更新 episode 计数和 UI 状态。

## 8. 测试覆盖边界

```mermaid
flowchart LR
    tests[tests/scripts]
    gateway_tests[test_data_collection_gui_gateway.py<br/>默认 Thor config / recorder script / BOX devices]
    box_tests[test_thor_box_client.py<br/>BOX config / decode / polling / missing wheel fallback]
    pcs_tests[test_thor_persistent_session.py<br/>single-process API contract]
    mp_tests[test_thor_persistent_session_multiprocess.py<br/>worker event protocol / partial failure / retry / preview]
    recover_tests[test_thor_record_auto_recover.py<br/>auto_recover config / trigger / script runner]
    lr3_tests[test_thor_lerobot_v3_pts.py<br/>PTS / BOX v3 writer helpers]

    tests --> gateway_tests
    tests --> box_tests
    tests --> pcs_tests
    tests --> mp_tests
    tests --> recover_tests
    tests --> lr3_tests
```

这些测试主要覆盖纯 Python 协议、状态机和数据契约；真实 Argus、GStreamer、PWM、
BOX UDP 和 11 路外场稳定性仍依赖 Thor 上按 `DEPLOYMENT.md` 执行的自检与外场验证。
