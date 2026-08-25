# Marker Rig -> TCP 解算接续记录（2026-08-25）

本文记录 `UMI Marker→TCP 重复性`、解算按钮、CAD/layout 数据边界、以及 Dataset Processing
生产接入的当前结论和实现状态。

> **状态（2026-08-25 下午）**：解算链路已从「复用生产 cube tracker」改为 metrology 四件套，
> layout resolver 已实现并有测试，前端已修好 BOX ID 枚举并补了格式说明。已部署到 Thor。

---

## 已确认结论

### 生产 rig frame

- 生产坐标系里的 marker rig frame 直接使用 CAD frame。
- pivot 专用治具的球心与 `TCP_closed` 完全重合，因此 `socket_beyond_tcp_mm = 0`。
- **当前 rig = 旧 cube 换了安装位置**（2026-08-25 操作者确认）。cube 的六面几何、marker id、
  实测边长都没变，变的只是它相对夹爪的安装位姿，也就是 lever arm。
  - 因此 marker layout **不需要重新估计**：角点表直接从生产 config 的 cube 模板生成。
  - 因此 `p_cube_tcp` **必须重解**：TCP 在 cube 系下的位置随安装位置改变。
  - 注意：这**不是** `docs/box_umi_distributed_marker_rig_design.html` 里 2026-08-18 那套
    分布式五臂 rig（DICT_6X6 / id 7,12,14,16,17 / 60.6 mm / 杠杆臂 102.3 mm）。
    两者不要混。当前这套仍是 36h11 + id 0-4 / 11-15 + 71.9 mm。

### 单点 pivot 能解什么

固定球心 pivot 的观测约束是：

```text
R_i @ p_rig_tcp + t_i = q_base
```

单点 pivot 只能约束 TCP 原点，**不能观测 rig→TCP 的旋转**。旋转必须来自 CAD、机械装配
datum，或多个已知不同 TCP 点的 pivot/Procrustes 数据。

**本轮决定：旋转继承原生产 cube 的 `R_cube_tcp`**，解算只更新 translation，
继承来源写进 bundle 的 `rotation_source`。

### CAD 真值和测量 layout 分开

保持三个 provenance 分离的输入，收敛到一个生产 artifact：

- `marker_rig_cad/v1`：CAD 真值（球心、贴标面、`T_rig_cube`）。
- `marker_layout/measured_v1`：实测角点（`bootstrap_marker_layout` 输出，自带任意 frame）。
- `marker_layout_resolved/v1`：**生产用**，由 resolver 生成。
- `marker_rig_to_tcp_calibration/v1`：生产 EE trajectory 用的 marker→TCP bundle。

---

## 本轮完成的实现

### 1. layout resolver（新）

`third_party/opencv_kalibr/metrology/cli/build_rig_layout_from_cube.py`

```bash
python -m metrology.cli.build_rig_layout_from_cube \
    --tracking-config third_party/opencv_kalibr/hikon_cube_tracking_offline/config_thor/april_cube_tracking_in_robot_base_thor.yaml \
    --cube left \
    --out outputs/metrology/newmount/marker_layout_resolved.json \
    [--cad-json cad.json | --t-rig-cube "..."] \
    [--measured-layout marker_layout_measured.json]
```

三个 provenance 分离的输入，一个 `marker_layout_resolved/v1` 输出：

- **cube 模板**：从生产 YAML 直接读 `marker_ids` / `cube_size_cm_xyz` / `marker_size_cm`，
  不在 resolver 里再抄一份数字（2026-08-12 的 left/right 错配就是抄出来的）。
- **CAD 刚体位姿** `T_rig_cube`：可选。默认 identity，即「rig 系就是 cube 系」。
  也接受 `T_cube_rig`（自动求逆并在 provenance 里标注）。校验正交 + det=+1。
- **实测修正**：可选。**先 Kabsch 刚性对齐到模板再替换角点**——测量 layout 自带任意 frame
  （`bootstrap_marker_layout` 以角点质心为原点），直接替换会把 cube 原点挪走，
  等于悄悄作废历史上每一个 `T_cube_tcp`。

**关键不变量（有测试守着）**：identity 位姿下，resolved layout 与生产 tracker 解析生成的
角点表 **bit-identical**（实测 max 差 3.5e-18 m）。这是「把 tracker 指向 layout 文件」
唯一的安全性论据——一旦漂了，换估计器就等于悄悄换了几何。

测试：`metrology/tests/test_rig_layout_from_cube.py`（10 项）。

### 2. 解算后端换成 metrology 四件套

`_run_marker_tcp_solve()` 不再调 `run_april_cube_tracking_local.sh`，改为四步：

```text
build_rig_layout_from_cube  ->  marker_layout_resolved.json
detect_rig_markers          ->  detections.npz      （只检有标定的相机）
track_marker_rig_in_base    ->  tracking_run/       （corner 级联合 BA）
pivot_marker_tcp_calibration->  pivot_report.json + marker_to_tcp_calibration.json
```

solve 目录结构：

```text
<session>/solve_<box>_<ts>/
  input_dataset/episodes/episode_00000N   # 软链到原始 episode，重新编号
  marker_layout_resolved.json
  detections.npz
  tracking_run/
  pivot_report.json
  marker_to_tcp_calibration.json          # 从现有生产 bundle 复制后 merge
  solve_summary.json
  solve.log
```

要点：

- **相机集合 = 该次所有 episode 都录到的 `cam_*.mkv` ∩ 外参 summary 里已标定的相机**。
  检测是整条链最贵的一步，没标定的相机在 tracking 里本来就会被跳过。
- **字典从生产 config 读**（`cube_tracker.aruco_dictionary`），不带默认值：
  字典错了不会报错，只会每个 id 解成别的东西然后「成功」拟合出垃圾。
- **解释器靠探测而不是猜**：`_marker_tcp_python()` 逐个候选跑
  `import cv2, numpy, scipy, yaml`。Thor 的 repo `.venv` 有 cv2 没 scipy，
  直接用它会先解码完 7 路 1080p 再死在 BA 的 import 上。
- **异步执行**：POST 只做校验（BOX、cube、样本、offset、旋转来源）然后起线程返回，
  面板通过快照轮询 `stage` / `message`。同时只允许一个解算在跑。
- **解算成功后 stage 回到 `capture`**（不是 `done`）：解完一个 BOX 不代表会话结束，
  `done` 会把「继续录样本 / 解第二个 BOX」锁死。

### 3. 旋转的 frame 组合

继承来的是 `R_cube_tcp`。若 `T_rig_cube` 非单位阵，写进 bundle 的必须是
`R_rig_tcp = R_rig_cube @ R_cube_tcp`——tracker 报的是 rig 位姿，直接写未组合的那个
就是一个恰好等于 CAD 旋转的静默 frame 错误。`_compose_rig_rotation()` 负责这件事，
并把组合过程追加到 `rotation_source`。

### 4. 生产接入：layout 跟着 bundle 走

`_write_ee_trajectory_override_config()` 在写 override YAML 时，会看 bundle 旁边有没有
`marker_layout_resolved.json`：

- `rig_frame_is_cube_frame: true` → 只覆盖 `ee_from_cube.marker_to_tcp_calibration_path`。
- `false` → **同时**设置 `cube_tracker.marker_layout_path`。否则 tracker 仍按解析 cube
  角点跑，却配上一个 rig 系的 `T_rig_tcp`，帧系对不上且无症状。

这样 UI 只需要一个路径输入框，两个 override 都会被正确施加，并记进 processing meta。

### 5. 前端修复

- **BOX ID 下拉永远是空的**（用户 2026-08-25 报的）：面板用 `deviceBoxId()` 取 id，
  但 `box_client.py:1938` 对单 BOX 车队**故意**保持 `box_id=""`（保持 sensor id 不带前缀、
  数据集兼容）。于是 `boxOptions()` 把唯一一台 BOX 过滤掉了，所有按钮永久禁用。
  新增 `deviceBoxIdentity()`：`box_id` → `sn` → `box<device_id>`，
  在这台机器上得到 `box1672693301`，正好就是生产 bundle 的 cube key 形式。
  **不要**拿它替换 `deviceBoxId()`：录制器命令里空 box_id 表示「那台唯一的 BOX」，
  换成具体 id 会匹配不到任何东西。
- **格式说明弹窗**：`CAD / 真值 JSON` 和 `static_transform.json` 两个输入框各加一个
  「格式说明」按钮，弹窗里给完整 JSON 示例 + 键名查找顺序 + 单位约定。
- offset 文案改为「球心与 TCP 重合，保持 0；非 0 只用于特殊治具」，
  并说明旋转继承来源。
- `App.tsx` 已把 `markerTcpCalibrationPath` 透传给 `api.queueTrajGen(...)`。

---

## 测试与验证

```bash
# 网关
PYTHONPATH=src:. python3 -m pytest tests/scripts/test_data_collection_gui_gateway.py -k 'marker_tcp or traj_gen'
# -> 10 passed

# layout resolver
cd third_party/opencv_kalibr && PYTHONPATH=. python3 -m pytest metrology/tests/ -q
# -> 254 passed

# 前端
cd tools/data_collection_gui/frontend && npm run build   # -> 通过
```

新增的网关测试：

- `test_marker_tcp_solve_runs_the_metrology_chain_and_writes_a_production_bundle`
  —— 四步顺序、episode 重编号、未标定相机被排除、字典、继承旋转与其 provenance、summary 内容。
- `test_marker_tcp_solve_returns_immediately_and_reports_progress_through_the_session`
  —— 异步返回、并发拒绝、失败落到 `stage=failed`。
- `test_marker_tcp_solve_rejects_a_bad_socket_offset_before_spawning_anything`
- `test_marker_tcp_solve_refuses_a_box_the_production_bundle_does_not_cover`
- `test_queue_traj_gen_writes_a_marker_tcp_override_config_and_records_it`
- `test_queue_traj_gen_carries_a_cad_rig_frame_layout_into_the_tracker`

另外用真实 parser 校验过网关拼出来的 argv：四个 CLI 全部接受。

**已知无关失败**：`test_marker_tcp_registers_static_transforms_and_writes_report` 在
`.venv` 下失败（该环境缺 scipy），在 conda python3 下通过；stash 掉本轮改动后同样失败，
非本轮引入。

---

## 仍未完成 / 需要注意

1. **解算链路还没在真机数据上跑过一次**。单测把四个子进程都 stub 了，argv 校验过，
   但端到端（真检测 → 真 BA → 真 pivot）还没跑。第一次跑请留意 `solve.log`。
2. **`T_rig_cube` 目前没有真值来源**。当前一律走 identity（rig 系 = cube 系），
   这在「旋转继承 + 只更新 translation」的口径下自洽。若要让 rig 系严格等于 CAD 系，
   需要从 Onshape 导出 cube 在 CAD 系下的刚体位姿，写成 `T_rig_cube` 喂给 resolver。
3. **`--measured-layout` 通路已实现但没在生产用过**。GUI 目前不暴露这个入口，
   只能命令行调。要不要接进 GUI 取决于是否要把 2026-08-19 反解出的逐 marker 边长
   （55.54 / 55.21 / 55.61 / 55.77 mm vs config 的 56.0）纳入生产。
4. **`cube_size_cm_xyz` 的 left/right 错配仍未修**。生产 config 里那段注释写明
   71.92/71.67/71.09 实际属于 id 11-15 的 cube，但 `check_thor_umi_cube_geometry()`
   的期望表仍是旧配对，改 config 会让 run 直接 abort。数值上只值 ≤0.07 mm，
   是记账问题不是精度问题——但 resolver 现在直接读这个 config，所以修的时候两边要一起改。
5. **重复性还是 0 组**。设计文档要求的五组（静止 / 慢速 / 遮挡 / 拆装重装 5 次 / 载荷）
   一组都没采。所以现在解出来的常量是**单次装夹**的值，不能当冻结常量用。

---

## 需要特别避免的误判

- 不要把单点 pivot 的输出当成 rotation 证据。它只证明 TCP 原点。
- 不要把 raw CAD JSON 和 measured layout JSON 混成一个「万能 JSON」。
- 不要在没有 layout override 的情况下默认认为 `marker_to_tcp_calibration.json` 足够表达
  「生产 rig frame = CAD frame」。（现在 `_write_ee_trajectory_override_config()` 会自动
  处理这件事，但前提是 layout 文件躺在 bundle 旁边。）
- 不要把当前这套（旧 cube 挪位置）和 0818 的分布式五臂 rig 混为一谈。
  字典、marker id、尺寸、杠杆臂全都不同。

---

## 部署

```bash
bash run/deploy.sh thor --no-frontend
```

**注意版本偏移**：`run/deploy.sh` 的默认形态会在本机起 vite dev server 指向 Thor 网关。
前端改动会被 HMR 立刻推到浏览器，网关改动不会——2026-08-25 的
「录制样本 → side must be left or right」就是这么来的：新前端发 `box_id=`、
旧网关要求 `side=left|right`。改了网关就要重新部署。
