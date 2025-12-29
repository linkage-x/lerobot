用途：基于 HIROL 目录拷贝 episode 片段

- 位置：`src/lerobot/datasets/hirol/hirol_edit.py`
- 功能：将 HIROL 目录中第 i~j 个 episode（以 `episode_XXXX` 目录命名）复制到目标目录，保持原始结构。
- 适用：仅用于 HIROL 目录。若是 LeRobot v3 数据集，请使用 `lerobot_edit.py`（会自动识别 v3 并按 v3 方式处理）。

快速开始
- Dry‑run（只列出将要处理的 episodes，不写盘）：
  `python -m lerobot.datasets.hirol.hirol_edit copy-slice --src /data/fr3/1107_insert_tube_fr3_3dmouse_contain_ft_279eps --dst /data/fr3/1107_subset_1_20 --start 1 --end 20 --dry-run`
- 真实拷贝：
  `python -m lerobot.datasets.hirol.hirol_edit copy-slice --src /data/fr3/1107_insert_tube_fr3_3dmouse_contain_ft_279eps --dst /data/fr3/1107_subset_1_20 --start 1 --end 20`
- 同盘硬链接节省空间：
  `python -m lerobot.datasets.hirol.hirol_edit copy-slice --src /data/fr3/1107_insert_tube_fr3_3dmouse_contain_ft_279eps --dst /data/fr3/1107_subset_link_1_50 --start 1 --end 50 --mode hardlink --fallback-copy`

参数
- `--src` 源目录（直接包含 `episode_XXXX/` 子目录）。
- `--dst` 目标目录（不存在则创建）。
- `--start, --end` 1-based、包含端点；按文件夹名解析并自然排序。
- `--mode` `copy`（默认）/`hardlink`/`symlink`。
- `--fallback-copy` 链接失败自动回退到 `copy`。
- `--overwrite` 目标存在同名目录时覆盖（默认跳过）。
- `--dry-run` 只打印将要处理的 episodes，不写盘。

设计
- 明确只面向 HIROL 目录（`episode_XXXX/`）；非此命名会被忽略。
- 三段式：发现（discover_episodes）→ 选择（i..j）→ 操作（复制/链接）。
- 安全默认：不覆盖、支持 dry-run，失败时清理未完成目标，避免脏数据。

注意
- `hardlink` 仅同一文件系统有效；跨盘/挂载会失败，可配合 `--fallback-copy`。
- `symlink` 依赖下游工具是否正确处理符号链接，不确定时建议 `copy`/`hardlink`。

