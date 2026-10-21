# FR3 纠偏 trial 录制与下一轮训练（2026-10-21）

录制所用 checkpoint：`fr3_spacemouse-insert__delta_ee_from_prev_cmd__trial150_plus_funnel163_from030000__pi05__20261019_071246/030000`（下称 071246）

## 结论

1. **可以开下一轮训练。**
   - 纠偏数据已录满：153 个 episode，67799 帧。
   - 完整性检查、坐标系、标签语义、gateway QC 全部通过。
   - 三源合并预检通过（`--check-only`）：466 个 episode，178486 帧。
2. **训练配置：** 照 071246 的配方从 pi05_base 训全新 LoRA，只改两处：
   - 换成新视图（多了纠偏 153）；
   - 步数 30k → 45k，让 epoch 数和 071246 持平。
3. **用哪些数据：** 三个源全用，不剔除。
   - 这样和 071246 只差“加了纠偏数据”这一个变量。
4. **“策略长时间悬停在接管高度之上时，漏斗照样接手”：不做。**
   - 只有在 A 臂评估里“高位停住/漂走”成了主要失败方式时才重新考虑，见下文。

## 一、录制结果

### 设置

- 每轮都一样：B 臂、录整次 trial、纠偏示范。
- 策略步数上限 600，漏斗接管 dz 60–300 mm（其中 30% 抽成晚接管，dz 15–40 mm）。
- 无人值守，每轮 50 次。

### 各轮结果

| run | seed | 次数 | 抓住 / 自动判插入 | 首次落点 | 抬起重对 | 丢包 | 写入 episode |
|---|---|---|---|---|---|---|---|
| 160805 / 164633 / 182453（10-20） | 20 / 20 / 21 | — | — | — | — | 2 | 2 / 18 / 3 |
| 095349 | 22 | 2 | 1 | — | — | 0 | 1（trial 1 是每日自检，日志里记成了 collision） |
| 100247 | 22 | 30 | 30 / 30 | 27 | 6 | 0 | 30 |
| 112231 | 23 | 50 | 49 / 49 | 42 | 14 | 0 | 49 |
| 121258 | 24 | 50 | 50 / 50 | 37 | 15 | 0 | 50 |

- **seed：** 已用于录制的 seed 是 0、7、20、21、22、23、24。评估只用 ≥ 1000 的 seed。
- **重复位置：** 有 3 个 trial 撞了位置：160805 和 164633 都用了 seed 20，095349 和 100247 都用了 seed 22。它们是不同的执行，保留。
- **丢包验收通过：** 开 kEnforce（控制线程 SCHED_FIFO 99）后，130 次写入 0 次丢包。修之前是 23 次写入 2 次丢包。
  - 混杂因素：NUC 上的 Edge 同时被关掉了，分不清是哪个改动起的作用。
- **B 臂自动判定可信：** 10-20 自动判定和人工评分 36/36 一致，所以这些 episode 都按“进了”采信。

## 二、决定：悬停时漏斗接手，不做

### 事实

- **唯一一例：** trial 22（112231）是 52 个晚接管 trial 里唯一一次因为没接管而失败。全部 164 个纠偏 trial 里也只有这一次。
  - 另外 3 个晚接管失败与此无关：1 次丢包，1 次自检，1 次是 contact_low 保护上线前的冒烟测试。
- **trial 22 不是“慢”，是“偏了之后停住、漂走”：**
  - 这个 peg 在 y = −0.253 的边缘位置。策略在 100/150 mm 高度的瞄准误差约 80 mm，是 B 臂中位数（约 25 mm）的 3 倍。
  - 第 450 步降到约 70 mm 后就不再下降。
  - 之后 150 步一直停在 70 mm 高度，横向往外漂，xy 误差从 80 mm 增大到 205 mm。
  - 这次抽到的接管高度是 15.4 mm，策略永远到不了，600 步后 grasp_timeout。
- **其余 51 个晚接管都正常：** 策略自己降到抽中的高度，接管步中位 425、最大 507（上限 600）。

### 第一性原理

1. **录纠偏数据是为了在策略真会到达的状态上拿到专家标签。**
   - 如果加一个“在高位停住就接手”的触发，接手发生在约 70 mm。这个高度本来就在正常接管 60–300 mm 的范围里，正常 trial 已经覆盖了。
   - 所以它带不来新的状态覆盖，反而会把一个“晚接管、低位抬起重对”的 trial 变成一个普通 trial，晚接管的比例被稀释。
   - 晚接管存在的意义恰恰是录“低位偏了 → 抬起 → 重对”。
2. **收益很小：** 约 0.6% 的 trial 白跑，每次约 45 s。
   - no_close 不会让循环停下，机械臂会自己回原位接着跑，所以也没有安全或无人值守上的代价。
3. **成本和风险：** 要新加一个“高位停住”的阈值（速度多低、停多少步）。
   - 策略正常接近时本来就会放慢：它模仿了漏斗的下降节奏，最后 30 mm 大约要 45 步。
   - 阈值定紧了会误触发，悄悄改掉接管高度的分布。代码、测试、分布语义都会变复杂。
4. **真正值得记下的是失败方式本身：** 大偏差 → 在高位停住/漂走。这是 A 臂可能出现的失败方式，DAgger 里最该补标签的就是这类状态。
   - 但它要在 A 臂评估里被测到频率才值得针对：现在只有 B 臂的 1 例。

### 什么时候重新考虑

在 A 臂评估中，如果“SEARCH 阶段高于 45 mm 停住或漂走、最后超时”占失败的 ≥ 20%，或占全部 trial 的 ≥ 10%，再做一个针对性的触发：
- 新的进入原因 `stalled_high`：已经降到 100 mm 以下，之后 z 方向连续 N 步几乎不动。
- 只在这类 trial 上启用，并单独计数。

## 三、数据评估

### 新数据集

`outputs/datasets/trial_<071246>_030000`

- **完整性：**
  - 153 个 episode，67799 帧；索引连续，没有 NaN。
  - 时间戳是 30 fps。
  - 两路视频（ee/side）各 7 个文件，解码出的帧数和 parquet 完全一致。
- **gateway QC：** PASS（7 pass，0 warn，0 fail），10-21 写入 `meta/processing.json`。
- **坐标系一致：**
  - `episode_frames.jsonl` 每个 episode 都有 T。
  - 和 10-19 trial_030000 的 T 相比，平移最多差 8.9 mm、旋转约 0.9°。差异全部来自每个 trial 回原位时的姿态散布；旧数据集内部也有一样的散布（y 147.5–156.0 mm）。
  - 所有 episode 的首帧状态完全相同，(0.3811, −0.1353, 0.0970)，std 为 0。它等于 funnel105 stub 的起始位姿约定，所以合并后的约定仍然等于 030000 的。
- **38 个 episode 里有 > 10 mm 的状态跳变：**
  - 逐个对过日志：全部发生在“低位策略步跳过”的位置，没有一个是来历不明的跳变。
  - 共有 45 个 episode 跳过了低位步；其中 7 个跳过的步数少，跳变 < 10 mm。
  - 录下的动作是逐步相对 runtime 真实的上一条指令（prev_cmd）编码的 delta，所以跳变处没有异常大的标签。
  - 唯一的副作用：跨过跳变的 50 步 action chunk 里少了被跳过的那段下降。这正是 10-20 批准“跳过”时想要的效果：不监督低位偏离的那段下降。
- **帧的构成：**
  - 策略帧（is_intervention 0）占 64%，漏斗帧占 36%。
  - 每个 episode 中位数是策略 293 帧、漏斗 140 帧；长度 268–595 帧。
- **晚接管 episode：** 45 个，大多数都“抬起 → 重对 → 下降”。这是 10-19 的 trial150 里完全没有的状态。

### 三个数据源

| 源 | episode | 帧 | 录制时的策略 | 漏斗 | 位置 |
|---|---|---|---|---|---|
| `trial_<030000>` | 150 | 70169 | 030000 | v14：在高处一次对准，然后直下 | seed 0，只有 58 个不同位置 |
| `funnel_<030000>` | 163 | 40518 | 030000 | v14，只录接管到合手 | — |
| `trial_<071246>`（新） | 153 | 67799 | 071246 | 纠偏：留残差、边降边纠；晚接管时抬起重对 | seed 20–24，约 150 个不同位置 |
| 合计 | 466 | 178486 | | | |

### 为什么三个源全用，不剔除旧的“高处一次对准”数据

- **一个变量：** 新模型和 071246 只差“加了纠偏 153”。A/B 的结果能直接回答纠偏数据有没有用。
- **标签冲突不大：** 设某个高度为 h。抽到的接管高度高于 h 的 trial，在 h 处的标签是漏斗在纠偏；接管高度低于 h 的 trial，在 h 处的标签是策略自己在直降。
  - 新数据里，h = 100 mm 时约 58% 的标签在纠偏，h = 30 mm 时约 82%。
  - 闭环执行时，每步都部分纠偏，误差会逐步收敛。所以混合标签学到的是“部分纠偏”，不是“不纠偏”。
- **剔除留作后手：** 如果这一版 A 臂没有提升，再做一个剔除 funnel163 的对照。不先做，因为它会同时改变数据量和分布，两个变量。

### 已知风险（训练时不处理，评估时要看）

- **策略帧会把 071246 的节奏和偏差一起教给新模型：**
  - 071246 的策略帧慢：在 B 臂里降到 100 mm 以下的中位步数是 320–340。
  - 它也偏：到 60 mm 时的瞄准误差中位约 27 mm（112231 summary）。
  - 新模型会照学这两点，所以评估要用 600 步的上限。
- **pi05 只能对干预帧屏蔽某些动作维度**（`intervention_unsupervised_action_dims`），没有“不监督策略帧”的选项。这次不加。

## 四、下一轮训练

### 配置

照 071246 的配置，只改视图名和步数。

| 项 | 值 | 和 071246 比 |
|---|---|---|
| 基座 | pi05_base snapshot a538eb2（离线缓存） | 相同 |
| LoRA | r64 / alpha64，目标模块是 gemma_expert 的 attn+mlp，加上 state/action/time 投影 | 相同 |
| batch / workers | 8 / 4 | 相同 |
| lr | 5e-5，warmup 200，cosine 衰减到 5e-6 | 相同 |
| 衰减步数 | 45000 | 30000 → 45000 |
| 总步数 | **45000**，每 5k 存一次 | 30000 → 45000 |
| 精度 | bf16，gradient checkpointing，use_amp false（训完再对齐） | 相同 |
| loss 权重 | [1, 1, 1, 0.2, 2] | 相同 |
| 训练 seed | 1000 | 相同 |
| W&B | 关；不上传 checkpoint | 相同 |

**为什么是 45k 步：**
- 071246 训了 313 个 episode、110687 帧，30k × 8 ≈ 2.17 个 epoch。
- 它的 loss 在第 1 到第 2 个 epoch 之间还在下降（约 0.073 → 0.058），没有饱和。
- 新视图 178486 帧，30k 步只有 1.34 个 epoch；45k 步是 2.02 个 epoch，和 071246 持平。
- 用时按 071246 的 1.04 s/步估算，约 13 小时。

### 新视图名

`fr3_spacemouse-insert__delta_ee_from_prev_cmd__trial150_plus_funnel163_plus_corrective153_from030000`

### 启动步骤

1. **建议先把工作树提交：** 3a3d420a 之后的改动都还没提交，提交后 checkpoint 才能对应到一个 commit。
2. **在 NUC 上合并：**

   ```bash
   C=fr3_spacemouse-insert__delta_ee_from_prev_cmd__L4_full48_plus_dagger108_plus_funnel105_030000__pi05__20260930_053656_030000
   K=fr3_spacemouse-insert__delta_ee_from_prev_cmd__trial150_plus_funnel163_from030000__pi05__20261019_071246_030000
   N=fr3_spacemouse-insert__delta_ee_from_prev_cmd__trial150_plus_funnel163_plus_corrective153_from030000
   PYTHONPATH=src:. .venv-fr3/bin/python tools/fr3/fr3_merge_policy_ready_datasets.py \
     --base-view outputs/exports/training_views/fr3_spacemouse-insert__delta_ee_from_prev_cmd__L4_full48_plus_dagger108_plus_funnel105_030000 \
     --dagger-roots outputs/datasets/trial_$C outputs/datasets/funnel_$C outputs/datasets/trial_$K \
     --output-root outputs/exports/training_views/$N --repo-id local/$N --no-base-episodes
   ```

   合并后检查三件事：
   - `start_pose_episodes` 是 303 个（150 + 153 个 whole-trial episode）；
   - 起始位姿约定等于 030000 的；
   - LeRobotDataset 能打开。
3. **拷到 dev box：** 用 `rsync --copy-links` 拷到 `/home/hanyu/Codes/lerobot_train/outputs/exports/training_views/$N`。前端要求两边的相对路径一致。
4. **停 policy server 腾显存：** 在 dev box 上跑 `bash run/policy_server.sh stop`。训练期间不能 rollout。
5. **启动训练：** `POST http://192.168.1.156:8765/api/training/start`，payload 如下：

   ```json
   {
     "hostId": "hanyu@192.168.1.179:/home/hanyu/Codes/lerobot_train",
     "viewName": "fr3_spacemouse-insert__delta_ee_from_prev_cmd__trial150_plus_funnel163_plus_corrective153_from030000",
     "policy": "pi05",
     "pretrainedPath": "/home/hanyu/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/a538eb273274eb30f126a118f39dbc0ee212c883",
     "loraEnabled": true, "loraR": 64, "loraAlpha": 64,
     "loraTargetModules": "(.*\\.gemma_expert\\..*\\.(self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj)|model\\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))",
     "steps": 45000, "batchSize": 8, "numWorkers": 4, "saveFreq": 5000, "logFreq": 50,
     "device": "cuda", "useAmp": false,
     "policyConfig": "{\"dtype\": \"bfloat16\", \"gradient_checkpointing\": true, \"compile_model\": false, \"optimizer_lr\": 5e-05, \"scheduler_warmup_steps\": 200, \"scheduler_decay_steps\": 45000, \"scheduler_decay_lr\": 5e-06, \"action_loss_weights\": [1.0, 1.0, 1.0, 0.2, 2.0]}",
     "wandbEnabled": false
   }
   ```

6. **训完之后：**
   - 用 gateway 把 checkpoint 拉到 NUC（会自动设 use_amp=true）。
   - 两份 dev box 拷贝都跑一次 `fr3_align_checkpoint_use_amp.py`。
   - 手工拷进 policy server 的 repo，然后 `ensure --warm-last`。

## 五、A 臂对比评估

- **比哪几个：** 新 ckpt（45000）、071246、030000。
- **怎么跑：** 同一天交替跑，seed ≥ 1000（例如 1000、1001），每对共用一个 seed，每个 ckpt 10–20 次，步数上限 600。
- **主要指标：** 人评的插入率。
  - 是否先做“拿不准才问人”的门，到时再定：判 seated 且搜索落点 ≥ 9 次才问人。
- **辅助指标：**
  - 到 60 mm 时的瞄准误差（`aimXyAboveMm`），071246 在 B 臂里中位约 27 mm；
  - 合手高度和横向误差；
  - 失败方式的分布，特别是“高位停住/漂走”的占比，它决定第二节那个触发要不要做。
- **为什么要重测：** 071246 之前的 4/10 是在 seed 0、也就是训练见过的位置上测的，偏乐观。030000 也要在 600 步下重测。
