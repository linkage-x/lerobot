# FR3 tele 历史数据备份（2026-11-02）

备份用于恢复早期人类示教、人工接管、漏斗纠正和历史实验。用户先授权原始 50 条示教，盘点后又确认 A/B/C/D 四组额外数据。tele 源文件保持原样。

状态：**已完成并校验**（2026-11-02 11:19，北京时间）。原始部分 47 个文件、额外部分 2,648 个文件均与 tele 源文件 SHA-256 一致，12 个备份内相对链接全部有效。校验报告保存在归档根目录。

## 位置与来源

- host：Dante。
- 归档：`/home/hph/Codes/lerobot_workstation/outputs/backups/tele_human50_20261102/`。
- tele：`hph@192.168.1.155`，主机名 `tele-MS-7E07`；同一网卡同时持有 `192.168.100.155`，额外备份最终通过该旧地址完成。
- 源工程：`/home/hph/Code/lerobot/`，以下来源路径均相对于其 `outputs/`。
- 归档在忽略的 `outputs/` 内，不随 Git 提交保存；本文件记录位置、范围和恢复方法。

原始 50 条在 `datasets/`，额外数据按源目录结构存于 `additional/outputs/`。`additional_backup_plan.json` 列出额外 25 个来源目录；`source_manifest.json` 和 `additional_source_manifest.json` 记录源文件大小、修改时间及 SHA-256。

## 原始 50 条人类示教

| 来源目录 | episode | 原始帧数 | 频率 |
| --- | ---: | ---: | ---: |
| `datasets/insert_20260821_145637` | 20 | 20,845 | 60 Hz |
| `datasets/insert_20260821_170816` | 20 | 18,905 | 60 Hz |
| `datasets/fr3_spacemouse_20260821_174204` | 10 | 10,118 | 60 Hz |
| 合计 | 50 | 49,868 | 60 Hz |

三批共 243,979,708 bytes，包含轨迹 parquet、腕部/侧视视频、episode 元数据、相机设置、处理及同步记录。对应 30 Hz 训练视图有 24,945 帧；其元数据和逐 episode 来源映射存于 `reference_training_view_meta/`，仅备份了该派生视图的元数据。

现行轨迹曾修复夹爪读数哨兵值。修复前文件也完整保存在每个数据集的 `meta/gripper_sentinel_repair/original/`；默认 `data/` 是修复后的版本，分析历史读数时需区分。

原始部分的 47 个文件 SHA-256 与 tele 一致，另外核对了轨迹帧数、episode 长度、50 条来源映射及六个视频的头部帧数，详见 `verification.json`。

## 已批准的额外范围

| 组 | 内容 | 大小（十进制） | 用途 |
| --- | --- | ---: | --- |
| A | 六批 DAgger 原始接管及 105 条漏斗纠正数据 | 484,254,404 bytes | 保存人工纠正来源 |
| B | 其他 59 条早期人类遥操作录制，含抓放和少量试录 | 249,897,228 bytes | 历史对照和重新筛选 |
| C | `analysis/`、`logs/rollout/`、`logs/training/`、`rollout_traces/`、`rollouts/`、`metrology/`、`unattended/`、`dagger/` | 881,463,514 bytes | 抓取快照、逐步轨迹、判定、标定和实验记录 |
| D | 示教基线、DAgger108、漏斗105 三阶段各一个 30k 推理检查点 | 334,520,660 bytes | 复现历史策略对照 |

四组共 1,950,135,806 bytes，约 1.95 GB。A 含历史版本和筛选前数据，不能直接相加作为独立样本数；B 也不能未经筛选直接混入当前 R6 训练。

D 保存以下三个任务的 `checkpoints/030000/pretrained_model/`：

1. `L4_full48_holdout22_40`。
2. `fr3_spacemouse-insert__delta_ee_from_prev_cmd__L4_full48_plus_dagger108_030000__pi05__20260908_213000`。
3. `fr3_spacemouse-insert__delta_ee_from_prev_cmd__L4_full48_plus_dagger108_plus_funnel105_030000__pi05__20260930_053656`。

它们含 LoRA 权重、策略/训练配置及归一化参数，仍依赖匹配的 `pi05_base`；不含优化器状态，不能独立恢复原训练进度。

未整包复制全部中间检查点及优化器状态（`train/` 共约 15.25 GB）、GUI 日志（约 1.48 GB）、空录制、重复 eeframe 派生数据和标为 unreadable 的隔离数据。其他训练视图需要时按来源与配方重新构建。

## 完整性与引用

原始部分使用 `SHA256SUMS`、`verification.json`；额外部分使用 `ADDITIONAL_SHA256SUMS`、`additional_verification.json`。校验清单从归档根目录执行。

分析目录中的 12 个绝对符号链接指向 A 组已备份的 DAgger 文件。备份中已改成内部相对链接，逐一核对目标存在且位于归档内，悬空链接为 0；源链接和本地链接映射保存在 `additional_portable_links.json`，tele 的链接不变。

JSON 配置、日志和 manifest 中的旧绝对路径按原文保留。恢复时在工作副本中设置新路径；不要改写归档，否则原始校验清单会失效。

## 校验与恢复

先校验归档：

```bash
backup=/home/hph/Codes/lerobot_workstation/outputs/backups/tele_human50_20261102
cd "$backup"
sha256sum -c SHA256SUMS
sha256sum -c ADDITIONAL_SHA256SUMS
```

恢复到单独的工作目录：

```bash
restore_root=/tmp/fr3_tele_restore_20261102
mkdir -p "$restore_root/outputs/datasets"
rsync -a "$backup/datasets/" "$restore_root/outputs/datasets/"
rsync -a "$backup/additional/outputs/" "$restore_root/outputs/"
```

这样恢复原始录制、额外数据、分析记录和三个模型，分析目录内的相对链接仍有效。原 50 条的完整 30 Hz 训练视图须参考 `reference_training_view_meta/il_view_manifest.json` 重建；仅复制元数据不足以恢复训练视图。

需要研究夹爪修复前读数时，在恢复后的工作数据集中，用 `meta/gripper_sentinel_repair/original/` 覆盖对应工作副本的 `data/` 和 `meta/`，保留归档不变。

tele 的 `192.168.1.155` 旧 known_hosts 记录与当前机器不同。备份前已核对 ED25519 主机密钥与历史 tele 地址 `192.168.100.155`、`192.168.1.200` 的可信记录一致，传输使用 `HostKeyAlias=192.168.100.155`，没有关闭主机密钥校验。新地址多次中途断连，最后改用 tele 已有的 `192.168.100.155` 地址续传完成，没有修改任何网络配置。以后重新连接仍须确认目标身份。
