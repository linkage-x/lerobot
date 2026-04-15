# FR3 Calibration Host UV Workflow (2026-04-22)

## Goal

`docs/fr3_calibration_teaching_pose_workflow_20260421.md` 使用的是 Docker 运行方式。  
本说明改为**本机（非 Docker）+ `uv`** 工作流，适配 Hikrobot/Hikon 相机不走容器的场景。

覆盖脚本：

- `third_party/opencv_kalibr/fr3_calibration/teaching_pose_recorder.py`
- `third_party/opencv_kalibr/fr3_calibration/execute_pose_and_capture.py`
- `third_party/opencv_kalibr/fr3_calibration/09_teaching_mode_lerobot.py`

## 1) One-Time Host Setup

先在主机安装 Hikrobot MVS 到 `/opt/MVS`（此项不由仓库脚本安装）。

然后在仓库根目录执行 `setup_host_env.sh`（命令来自 `tools/fr3/setup_host_env.sh`）：

```bash
cd /home/corenetic/Code/lerobot
source "$HOME/.local/bin/env"

UV_CACHE_DIR=/tmp/uv-cache \
INSTALL_SYSTEM_DEPS=1 \
BUILD_LIBFRANKA=1 \
WITH_PIKA_SDK=1 \
WITH_GEN_CON_SDK=0 \
bash tools/fr3/setup_host_env.sh
```

如果系统依赖、`libfranka` 已装好，可改成增量模式：

```bash
cd /home/corenetic/Code/lerobot
source "$HOME/.local/bin/env"

UV_CACHE_DIR=/tmp/uv-cache \
INSTALL_SYSTEM_DEPS=0 \
BUILD_LIBFRANKA=0 \
WITH_PIKA_SDK=1 \
WITH_GEN_CON_SDK=0 \
bash tools/fr3/setup_host_env.sh
```

## 2) Session Environment (Host + UV)

每次开新终端，先加载环境（对齐 handheld 文档风格）：

```bash
cd /home/corenetic/Code/lerobot
source "$HOME/.local/bin/env"

export UV_CACHE_DIR=/tmp/uv-cache
export LEROBOT_REPO_ROOT="$PWD"
export PYTHONPATH="$LEROBOT_REPO_ROOT/src:/opt/MVS/Samples/64/Python:/opt/MVS/Samples/32/Python"
export HIKROBOT_MVS_HOME=/opt/MVS
export MVCAM_COMMON_RUNENV=/opt/MVS/lib

export LD_LIBRARY_PATH="/opt/MVS/lib/64:/opt/MVS/lib:/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CMEEL_PREFIX="$(find "$LEROBOT_REPO_ROOT/.venv/lib" -path '*/site-packages/cmeel.prefix' -type d | head -n 1)"
if [[ -n "$CMEEL_PREFIX" ]]; then
  export LD_LIBRARY_PATH="$CMEEL_PREFIX/lib:$LD_LIBRARY_PATH"
fi

fr3_uv() {
  uv run --python .venv/bin/python python "$@"
}
```

可选：快速检查主机端导入能力（尤其是 Hikrobot MVS）：

```bash
uv run --python .venv/bin/python python - <<'PY'
import importlib
for name in ("placo", "panda_py", "ruckig", "pyspacemouse", "easyhid", "pika", "pika.sense", "pika.gripper"):
    importlib.import_module(name)
from MvImport import MvCameraControl_class as mvs
print("imports=ok")
print("mvs_module=", mvs.__file__)
PY
```

## 3) 生成 Host 版 Calibration YAML

当前 calibration YAML 里大量路径是 `/workspace/...`。  
本机运行时建议生成 `*.host.yaml`，统一替换成仓库绝对路径：

```bash
cd /home/corenetic/Code/lerobot

uv run --python .venv/bin/python python - <<'PY'
from pathlib import Path

repo = Path.cwd()
src_dir = repo / "third_party/opencv_kalibr/fr3_calibration"
dst_dir = src_dir / "host"
dst_dir.mkdir(parents=True, exist_ok=True)

files = [
    "teaching_pose_recorder.yaml",
    "execute_pose_and_capture_realsense.yaml",
    "execute_pose_and_capture_hikrobot.yaml",
    "09_teaching_mode_lerobot.yaml",
]

for name in files:
    src = src_dir / name
    if not src.exists():
        print(f"skip (not found): {src}")
        continue
    text = src.read_text(encoding="utf-8")
    text = text.replace("/workspace/src/lerobot", f"{repo}/src/lerobot")
    text = text.replace("/workspace/outputs", f"{repo}/outputs")
    dst = dst_dir / name.replace(".yaml", ".host.yaml")
    dst.write_text(text, encoding="utf-8")
    print(f"generated: {dst}")
PY
```

如果你之前已经生成过一次 host YAML，并出现类似  
`/home/.../Code/home/.../Code/lerobot/outputs/...` 的重复前缀，可直接修复：

```bash
cd /home/corenetic/Code/lerobot
find third_party/opencv_kalibr/fr3_calibration/host -name '*.host.yaml' -type f -print0 \
  | xargs -0 sed -i 's#/home/corenetic/Code/home/corenetic/Code/lerobot/outputs#/home/corenetic/Code/lerobot/outputs#g'
```

然后按需修改 host 配置里的：

- `robot.robot_ip`
- Hikrobot 场景下 `runtime.camera_device_ids`（替换 `HIK_SERIAL_0X`）

当前 `fr3_calibration/host` 的 Hikrobot 配置已经预置：

- `runtime.camera_exposure_us: 13000`
- `runtime.camera_gain_db: null`（开启自动增益）

## 4) UV Commands (Host)

### 4.1 记录教学位姿（轻量 JSON）

```bash
fr3_uv third_party/opencv_kalibr/fr3_calibration/teaching_pose_recorder.py \
  --config_path third_party/opencv_kalibr/fr3_calibration/host/teaching_pose_recorder.host.yaml
```

### 4.2 执行位姿并采集（RealSense）

```bash
fr3_uv third_party/opencv_kalibr/fr3_calibration/execute_pose_and_capture.py \
  --config_path third_party/opencv_kalibr/fr3_calibration/host/execute_pose_and_capture_realsense.host.yaml
```

### 4.3 执行位姿并采集（Hikrobot/Hikon）

```bash
fr3_uv third_party/opencv_kalibr/fr3_calibration/execute_pose_and_capture.py \
  --config_path third_party/opencv_kalibr/fr3_calibration/host/execute_pose_and_capture_hikrobot.host.yaml
```

只执行前 N 个 pose（例如前 250 个）可直接加 CLI  --execution.max_records=250 覆盖：
也可写在 YAML 的 `execution.max_records` 中。

### 4.4 Legacy 一体式教学记录 （这个最管用）

```bash
fr3_uv third_party/opencv_kalibr/fr3_calibration/09_teaching_mode_lerobot.py \
  --config_path third_party/opencv_kalibr/fr3_calibration/host/09_teaching_mode_lerobot.host.yaml
```


## 5) 推荐流程

1. `teaching_pose_recorder.py` 产出 JSON 位姿库  
2. `execute_pose_and_capture.py`（优先 `joint_space`）执行并抓图  
3. 如需额外误差评估脚本，请在该目录新增对应脚本与 YAML 后接入流程

## 6) 常见问题

- `ModuleNotFoundError: MvImport`：确认 `/opt/MVS` 已安装，且 `PYTHONPATH` 包含 `/opt/MVS/Samples/64/Python`。
- Hikrobot 打不开：确认 `runtime.camera_device_ids` 已替换真实序列号，`camera_transport_layer` 与现场一致（`usb`/`gige`）。
- 路径仍指向 `/workspace`：重新生成 `*.host.yaml`，并确保命令使用 `third_party/opencv_kalibr/fr3_calibration/host/*.host.yaml`。
