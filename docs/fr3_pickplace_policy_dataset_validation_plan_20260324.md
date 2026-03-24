# FR3 Pick-Place Policy Dataset 验证计划（2026-03-24）

## 当前状态

- FR3 ACT 真机推理现在已经可以按正常 runtime 路径启动并运行。
- 但系统仍然无法完成完整的 pick-place 行为。
- 因此，当前最高价值的问题已经不再是“推理能不能跑起来”，而是“当输入是已知正确的数据时，policy 输出是否符合任务预期”。

## 核心思路

直接使用训练集中的 observation 喂给 policy，然后把 policy 输出与数据集 action 目标、以及 pick-place 任务语义进行对比。

这个方案比继续直接上真机试错更容易定位问题：

- 如果 policy 在训练集 observation 上就已经输出不对，问题更可能在 checkpoint 质量、dataset contract、预处理、或 action 解码。
- 如果 policy 在训练集 observation 上表现正常，但在真机 live observation 上表现不对，问题更可能在 live observation 构造、tactile / image / state 语义、或 runtime 安全门与 clipping。
- 如果 policy 离线输出合理，但真机仍然做不成 pick-place，问题更可能在执行保真度，而不是 policy 意图本身。

## 为什么这是当前最合适的下一步

目前比较大的、显式的 runtime 阻塞项已经基本被缩小了：

- inference entrypoint 已经能跑
- image semantics 已经做过检查
- gripper observation 语义已经对齐到 dataset contract
- move-to-start 与启动路径已经并入 runtime

所以接下来真正要验证的是：当 policy 看到一个合法 observation 时，它到底有没有输出我们以为它应该输出的动作。

## 主要假设

1. policy 输入合同仍然在某些模态上存在不一致。

可能的例子：
- state frame 不一致
- tactile value-space 不一致
- image preprocessing 不一致
- dataset feature ordering 不一致

2. policy 输入是正确的，但 checkpoint 本身并没有足够好地复现 pick-place 的动作分布。

3. policy 离线输出本来是合理的，但 runtime 执行链路把它变坏了。

可能的例子：
- action frame 转换
- gripper 单位映射
- first-frame gate
- per-step clipping
- OTG / 控制器延迟破坏了任务时序

## 验证计划

### Phase 1: 离线 Dataset-Fed Policy Sanity Check

目标：
验证 checkpoint 在输入“精确数据集 observation”时，能否复现预期动作。

方法：
- 用与真机推理完全一致的 preprocessing stack 加载 checkpoint。
- 先选取一小批已知成功的 pick-place 训练 episode。
- 将精确的数据集 observation 喂给 policy。
- 逐帧比较 policy 预测 action 与 dataset action。

要记录的指标：
- xyz action error
- rotation error
- gripper error
- first-frame error
- per-episode mean / p95 error
- grasp / lift / place 等关键阶段的 phase-specific error

结果解释：
- 如果离线误差很低，说明模型本身大概率还“会做这个任务”，剩余问题更偏部署侧。
- 如果离线误差已经很高，说明失败在进入硬件执行前就已经存在。

仓库内现有支撑：
- `tools/fr3/fr3_check_policy_dataset_frame_runtime.py`
- `tools/fr3/fr3_compare_infer_replay_decode.py`

### Phase 2: Pick-Place 阶段切片分析

目标：
避免被全局平均误差掩盖真正的失败点。

方法：
- 选取少量成功 pick-place 训练 episode。
- 按任务粗分阶段：
  - approach
  - pre-grasp alignment
  - close gripper
  - lift
  - transport
  - place
  - open gripper
- 检查 policy 输出是否保留了这些阶段转换。

重点观察：
- gripper 的 close / open 时序是否与 dataset 对齐
- lift 动作是否在该出现时出现
- 模型是否退化成“悬停 / 小幅安全动作”
- policy 是否始终输出稳定 pose，但从不真正 commit 到 grasp

### Phase 3: Live-vs-Dataset Contract Check

目标：
如果 Phase 1 表现良好，就继续找 live 输入到底是哪一类模态偏离了 dataset 预期。

方法：
- 在 runtime 起始时抓取一份 live observation bundle。
- 将其与最近邻 dataset start states / frames 做对比。
- 必要时对模态做单独替换或消融：
  - 只看 state
  - 只看 images
  - 只看 tactile
- 观察哪一种模态修正后，policy 输出会向 dataset 预期靠拢。

仓库内现有支撑：
- `tools/fr3/fr3_compare_live_capture_to_dataset.py`
- `tools/fr3/fr3_compare_live_capture_to_dataset_runtime.py`
- `tools/fr3/fr3_validate_infer_image_semantics.py`

### Phase 4: 回到真机前先做 Runtime Preview

目标：
只有在离线信号足够强之后，才继续真机 pick-place 尝试。

方法：
- 先使用 `--preview` 模式。
- 从一个已知接近 dataset 的初始化状态开始，跑少量 step。
- 将 preview target 与 dataset action 趋势做对比。
- 只有当 preview 输出在方向上明显合理时，才恢复真实 actuation。

## 近期执行顺序

1. 对少量已知 pick-place episode 跑 offline checkpoint-vs-dataset 对比。
2. 按模态与任务阶段做误差排名，而不是只看全局平均值。
3. 重点确认 gripper prediction 是否是最早开始发散的部分。
4. 如果离线输出正常，抓一份 live step0 observation bundle，再和最近邻 dataset starts 对比。
5. 明确 state、tactile、image 三类输入里哪一类是主要偏差源。
6. 带着收敛后的假设重新跑 preview，再决定是否继续真机 pick-place。

## 成功标准

这份计划如果能明确回答下面至少一个问题，就算成功：

- checkpoint 在 dataset observation 上是否还能复现 pick-place action？
- 如果不能，最先坏的是哪个阶段？
- 如果能，是哪一种 live 模态或 runtime 变换破坏了行为？
- 当前主要 blocker 到底是 policy 质量、input contract，还是 execution fidelity？

## 调查结束后的决策规则

- 如果 dataset-fed policy output 一开始就不对：优先查 checkpoint / dataset / preprocessing。
- 如果 dataset-fed output 正常，但 live-fed output 不对：优先查 live observation contract。
- 如果二者都正常，但真机仍失败：优先查执行跟踪、gripper 时序与控制侧保真度。

## 建议沉淀的产物

每次验证建议至少保存：
- 一份简短 markdown 结论
- 一份 per-frame action error csv
- 一份 top-k 最差帧表，并带 phase label
- 一句当前 dominant blocker 的结论句

这样后续这条线会一直是 evidence-driven 的，而不是继续在没有局部化失败点的情况下消耗真机试验次数。
