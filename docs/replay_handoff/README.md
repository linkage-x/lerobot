# P0 / BOX 2.0 replay 接手入口

该分支保存 2026-09-09 本地 replay 开发快照。主仓库和私有子模块均使用 `yzy/error_study`，从原有本地提交建立，没有合并 `hph/error_study` 后续更新，也没有改动 Thor 已部署控制代码。

首先阅读 [标定、模型与重播交接](../../exports/BOX_Franka_标定与重播交接_20260909.md)。当前是单臂原生 panda-py，只动机械臂、夹爪固定开口；不是完整双臂或夹爪联合重播交付。

## 获取代码

需要主仓库及私有 `linkage-x/opencv_kalibr`、`linkage-x/box-sdk` 的访问权限。不要仅下载网页 ZIP。

```bash
git clone --branch yzy/error_study https://github.com/linkage-x/lerobot.git
cd lerobot
git submodule update --init --recursive
git lfs pull
```

子模块 URL 使用 SSH，需在自己的 GitHub 账号配置 SSH 认证；若使用 HTTPS，应在本机配置对应 URL，不提交个人凭据。子模块以主仓库记录的 gitlink 为准，接手时不要直接 `git submodule update --remote` 升到其他提交。

## 代码范围

- `tools/fr3/native_arm_only_20260908/`：当前入口、原生 C++ 构建源码、Python 绑定、日志分析与离线测试。
- `tools/fr3/native_reuse_20260908/`：原生规划器的独立导数审计代码，不是另一个真机启动入口。
- `tools/fr3/staged_replay/`、`arm_only_replay_20260908/` 与根目录历史 replay 脚本：保留实现及故障排查参考，当前入口仍从 Thor 这些独立目录读取部分依赖。不要因其被提交而认为已重新放行全部历史入口。
- `outputs/p0_recovery_replay_20260907/` 中显式跟踪的脚本及模型：当前 P0 修正的模型生成、接触 IK、渲染和诊断代码；运行依赖输入另行传递。部分脚本拒绝覆盖已有输出，应先规划新输出目录。
- `third_party/opencv_kalibr`：sidecar 开口读取/补齐、MuJoCo 双臂离线检查等本地修改，源码仍留在私有仓库。
- `src/lerobot/robots/franka_research3/assets/franka_fr3/`：V2 单/双臂模型、拼接脚本、夹爪网格。旧 P0/P1/P2 仅保留为历史模型，当前修正 P0 以交接文档为准。
- 前端修改：macOS 大小写文件解析修复，以及双臂碰撞结果接入检查。

未提交：密钥/凭据、原始采集视频和数据、运行日志、虚拟环境、编译二进制、旧 MuJoCo 数据导出包、`arm_settle_fix_20260908.abandoned` 和未完成的 `replay_v2_20260908`。不要将它们从其他目录批量补交到公开主仓库。

## 运行资产不是仅靠 Git 就能恢复

当前 Thor 部署依赖四个完整目录及 Python / libfranka 环境，路径见交接文档。最新标定快照、991 帧关节/开口输入、`timed_plan.json`、模型 meshes、各 `manifest.json` 与独立编译库仍在 Thor，需经负责人授权另行同步和校验。主仓库公开，不把私有子模块整份源码或采集数据打包到公开 exports。

完整 TCP 工具配置实际读取 `/home/nvidia/box_api/replay_p0_arm_only_20260908/tool_reference.json`，不是新入口目录。入口中的 TCP 误差检查在分段结束时执行，不等同于旧自定义播放器的连续跟踪超限停止逻辑；原生控制及硬件故障保护仍在。

`native_arm_only_20260908/native` 是已部署 C++ 源码快照；`core_binding.reference.cpp` 是绑定文件的同版本副本。AArch64 的 `.so` 不提交也不能在 x86 直接运行。`seal_and_test.py` 是新部署包的离线核对/建清单程序，不是用于覆盖现有 manifest 的修复按钮。换机器须重新构建、核验依赖和场景，不要将编译成功视为真机放行。

该分支没有启动任何硬件。最近真机仅完成 Episode 0 部分段，独立物理标定精度、右 BOX 的独立 Cube→TCP 实测、夹爪网络与同步仍未关闭。

本次发布前验证：主仓库待提交的 61 个 Python 文件语法通过；`test_guarded_replay_offline` 与 `test_timed_candidate_offline` 共 6 项测试通过；子模块的两个 sidecar 开口补齐测试通过。未重新跑全部前端构建、所有双臂仿真或真机；原始 URDF/C++ 快照的换行及部分行尾空格保留，以免无关格式化影响已记录的源码/模型哈希。
