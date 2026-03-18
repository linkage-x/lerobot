# 代码审查报告：fr3_das_replay_real（真机重播）

**日期：** 2026-03-18
**审查文件：**
- `tools/fr3/fr3_das_replay_real_runtime.py`（新增）
- `tools/fr3/fr3_das_replay_real.py`（新增）

**模式：** no-spec | **审查层：** Blind Hunter + Edge Case Hunter
**结果：** 7 patch，4 defer，5 reject

---

## Patch（需修复，按优先级排序）

### P1. panda_py 连接未显式释放，可能引发双重控制冲突
- **严重性：** CRITICAL
- **位置：** `fr3_das_replay_real_runtime.py` → `move_to_das_start()`
- **问题：** `panda_py.Panda` 对象离开作用域后依赖 GC 析构，不保证在 `FrankaResearch3.connect()` 前完成。Franka 控制器可能仍持有第一个连接，导致第二次连接报错或 robot fault（概率性）。
- **修复：** 在 `move_to_joint_position()` 后显式 `del panda`，并加短暂延迟（如 `time.sleep(0.5)`），确保控制器释放。

```python
# 当前：
panda.move_to_joint_position(_IK_SEED_JOINTS_RAD.tolist())
print("[INFO] 已到达 DAS 起始关节角")

# 修复后：
panda.move_to_joint_position(_IK_SEED_JOINTS_RAD.tolist())
del panda
time.sleep(0.5)   # 等待 Franka 控制器释放控制权
print("[INFO] 已到达 DAS 起始关节角")
```

---

### P2. `actions[fi][7]` 无 schema 校验
- **严重性：** HIGH
- **位置：** `replay_real()` 循环体（两处）
- **问题：** 硬编码 action 向量宽度 ≥ 8。dataset 格式不匹配时 `IndexError` 崩溃（已发过命令），或静默读错列作为夹爪目标。
- **修复：** 循环前加断言：

```python
assert actions.shape[1] >= 8, (
    f"action 列数 {actions.shape[1]} < 8，期望 [x,y,z,qx,qy,qz,qw,gripper]"
)
```

---

### P3. shell 命令注入风险（`args.dataset` 拼入 `bash -lc`）
- **严重性：** HIGH
- **位置：** `fr3_das_replay_real.py` → `build_docker_command()`
- **问题：** `f"--dataset=/lerobot/{args.dataset}"` 拼入 `" ".join(runtime_args)` 后整体传给 `bash -lc`，含特殊字符的路径被 shell 解释执行。
- **修复：** 对用户输入参数加 `shlex.quote()`：

```python
import shlex
f"--dataset={shlex.quote(f'/lerobot/{args.dataset}')}",
f"--robot-ip={shlex.quote(args.robot_ip)}",
f"--gripper-port={shlex.quote(args.gripper_port)}",
```

---

### P4. 所有帧跳过时 `np.percentile([], 95)` 抛 ValueError
- **严重性：** HIGH
- **位置：** `replay_real()` → 统计汇总段
- **问题：** 全部帧 `target_z < 0.15m` 时 `pos_errors_mm` 为空，`np.percentile` 抛异常。
- **修复：** 统计前加空数组保护：

```python
if len(pos_arr) == 0:
    print("[WARN] 无有效帧（全部被跳过），无统计数据")
    return 0
```

---

### P5. `load_episode` mask 为空时给出无诊断的 IndexError
- **严重性：** MEDIUM
- **位置：** `load_episode()` 末尾
- **问题：** metadata 找到 chunk/file 但数据文件无匹配行，`mask=[]`，后续 `states[0]` 抛无诊断 IndexError。
- **修复：**

```python
if not mask:
    raise ValueError(
        f"Episode {episode_idx} found in metadata but no rows in {data_file}"
    )
```

---

### P6. Docker 服务名 `lerobot-fr3-sim` 用于真机有误导性
- **严重性：** MEDIUM
- **位置：** `fr3_das_replay_real.py`
- **问题：** 若该服务名指向仿真容器，`--robot-ip` 被传入但 sim 控制器可能忽略，操作者以为在操控真机。
- **修复：** 更新服务名或添加明确注释，确认 docker-compose.yml 对应的服务定义。

---

### P7. `subprocess.run(check=False)` Docker 失败时无诊断输出
- **严重性：** MEDIUM
- **位置：** `fr3_das_replay_real.py` → `main()`
- **问题：** Docker 启动失败只返回非零码，无任何错误提示。
- **修复：**

```python
import sys
result = subprocess.run(cmd, check=False)
if result.returncode != 0:
    print(f"[ERROR] docker compose run 失败，退出码 {result.returncode}", file=sys.stderr)
return result.returncode
```

---

## Defer（暂不处理）

| # | 描述 | 原因 |
|---|---|---|
| D1 | `_WARMUP` 语义：短 episode 无稳定期统计 | 与仿真版一致，可接受 |
| D2 | 固定 fps pacing 而非 timestamp 驱动 | 与仿真版相同设计决策 |
| D3 | Parquet chunk 整体读入内存 | 非当前变更引入，同仿真版 |
| D4 | T_B_Ws anchor 基于录制 states[0]，真机到位有微小偏差 | 根本设计取舍 |

---

## Reject（已排除，共 5 项）

- `_IK_SEED_JOINTS_RAD.tolist()` 崩溃：假阳性，常量已是 `np.array`
- 无 abort gate：用户明确要求 warn+skip
- `rotation_angle_error_deg` clip 范围：实际正确
- quat 归一化缺失：scipy 内部已处理
- `T_B_Ws` 双重计算有害：审查者自述"harmless"
