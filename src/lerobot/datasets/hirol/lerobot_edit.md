用途：从数据集中拷贝一段 episode（支持 LeRobot v3 与 HIROL 目录）

- 位置：`src/lerobot/datasets/hirol/lerobot_edit.py`
- 功能：
  - LeRobot v3 数据集：创建一个子集数据集，包含第 i~j 个 episode（基于 v3 的 parquet+mp4 结构，完整重建 meta/data/videos）。
  - HIROL 目录：将 `episode_xxxx/` 文件夹按编号范围复制到目标目录，保持原有结构。
- 调用方式：作为模块运行，子命令风格，易于扩展新功能。

快速开始（v3 数据集）
- Dry‑run（仅列出，不写盘）：
  `python -m lerobot.datasets.hirol.lerobot_edit copy-slice --src /data/fr3_lerobot/1107_insert_tube_fr3_3dmouse_contain_ft_279eps --dst /data/fr3_lerobot/1107_subset_1_20 --start 1 --end 20 --dry-run`
- 真实拷贝（生成新的 v3 子集数据集）：
  `python -m lerobot.datasets.hirol.lerobot_edit copy-slice --src /data/fr3_lerobot/1107_insert_tube_fr3_3dmouse_contain_ft_279eps --dst /data/fr3_lerobot/1107_subset_1_20 --start 1 --end 20`

说明：v3 模式下不会有 `episode_xxxx/` 目录；该工具会读取 `meta/info.json` 的 `total_episodes`，按 episode_index 选择 i..j，并在 `--dst` 路径下重建一套完整的 v3 数据集（meta/data/videos），只包含所选 episodes（必要时对视频进行裁剪重编码）。

快速开始（HIROL 目录）
- Dry‑run：
  `python -m lerobot.datasets.hirol.lerobot_edit copy-slice --src /path/to/hirol_task --dst /path/to/output_subset --start 1 --end 20 --dry-run`
- 真实拷贝：
  `python -m lerobot.datasets.hirol.lerobot_edit copy-slice --src /path/to/hirol_task --dst /path/to/output_subset --start 1 --end 20`

参数说明
- `--src` 源数据根目录，v3 模式下为包含 `meta/info.json` 的数据集根；HIROL 模式下为直接包含 `episode_xxxx/` 的目录。
- `--dst` 目标目录，若不存在会自动创建。
- `--start, --end` Episode 的编号范围，1-based，包含端点；工具会从文件夹名（HIROL）或 episode_index（v3）中解析与筛选。
- `--mode`（仅 HIROL 模式有效）文件复制策略：`copy`（默认）/`hardlink`/`symlink`。
- `--fallback-copy`（仅 HIROL 模式）当 `hardlink/symlink` 失败时自动回退到普通复制。
- `--overwrite`（仅 HIROL 模式）允许覆盖目标中已存在的同名 episode 目录。
- `--dry-run` 两种模式均支持，仅打印将要处理的 episodes。

典型示例
- v3：拷贝第 101~150 个 episode 到新数据集：
  `python -m lerobot.datasets.hirol.lerobot_edit copy-slice --src /data/fr3_lerobot/1107_insert_tube_fr3_3dmouse_contain_ft_279eps --dst /data/fr3_lerobot/1107_subset_101_150 --start 101 --end 150`
- HIROL：同一磁盘上用硬链接节省空间：
  `python -m lerobot.datasets.hirol.lerobot_edit copy-slice --src /path/to/hirol_task --dst /path/to/hirol_subset_link_1_50 --start 1 --end 50 --mode hardlink --fallback-copy`

目录结构假设
- v3：根目录包含 `meta/info.json`；数据存在 `data/`（parquet）和 `videos/`（可选 mp4）。
- HIROL：源目录直接包含 `episode_0001, episode_0002, ...` 等子目录。

设计与可扩展性
- 自动识别模式：优先检测 v3（`meta/info.json`，`codebase_version` 前缀为 `v3`）。否则按 HIROL 目录处理。
- 三段式（HIROL）：发现（discover_episodes）→ 选择（编号区间）→ 操作（复制/链接）。
- v3 采用内部工具链：调用 `dataset_tools.delete_episodes` 以补写 meta/episodes、裁剪/重编码视频并重建索引。
- 子命令式 CLI：当前提供 `copy-slice`，未来可按同样模式新增 `rename`, `filter`, `merge` 等命令。
- 安全默认：支持 `--dry-run`；HIROL 下默认不覆盖；失败时清理未完成的目标目录以避免脏数据。

已知限制与提示
- v3：视频裁剪依赖 `PyAV`，需确保环境可用（例如 `pip install av`）。否则可设置 `download_videos=False` 的数据转换路径，但本工具默认复制/裁剪视频以保持一致性。
- HIROL：`hardlink` 只能在同一文件系统内工作；跨磁盘或不同挂载点会失败，可加 `--fallback-copy` 自动退回普通复制。
- `symlink` 需要下游工具正确处理符号链接；若不确定建议使用 `copy` 或 `hardlink`（仅 HIROL）。
- 超出范围会自动截断；Dry‑run 可用于先行确认。

开发备注
- 入口：`main()`；核心逻辑：`copy_slice(opts)`；数据结构：`EpisodeEntry`, `CopySliceOptions`。
- 单测/脚本可用 `--dry-run` 模式做快速验证。
