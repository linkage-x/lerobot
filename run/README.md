# 部署：生产（maintainer）与沙箱

谁能部署到哪里（规则见 [AGENTS.md](../AGENTS.md) 第 2–4 节）：

| 谁 | 部署到哪里 | 怎么做 |
|---|---|---|
| 学生、agent | 自己的沙箱（无硬件） | `bash run/deploy.sh thor1 --sandbox <名字>`，见 AGENTS.md 第 3 节 |
| maintainer（`.github/CODEOWNERS` 里的 `@BigJohnn`） | 生产 | 本文 |

机器的地址、主机名和生产目录写在 [`run/targets/`](targets/README.md) 里，脚本里不写地址。部署时**必须写明目标**，
没有默认目标。

## 1. 生产由哪几部分组成（thor1）

| 部分 | 在哪 | 说明 |
|---|---|---|
| 网关 | thor1 `/home/nvidia/lerobot`，端口 8765 | 相机、BOX、机械臂、激光跟踪仪都接在 Thor 上，所以网关只能跑在 Thor |
| 部署副本 | corenetic `hph@192.168.11.44:~/lerobot-box-deploy` | 一份只用于部署的 clone（`box` 分支，含子模块和 LFS 文件）。**不要在里面开发**：守卫要求它干净并且在 `origin/box` 上 |
| 学生前端 | corenetic 端口 **5180**，systemd 用户服务 `lerobot-frontend-thor1` | 学生打开 <http://192.168.11.44:5180/>。提供的是部署副本里**构建好的 `dist/`**（`vite preview`），只有重新构建并重启服务时页面才会变（第 5 节） |

corenetic 上的 5173/5174 是 corenetic 账号自己的开发前端，不要占用。

## 2. 常规生产部署（PR 合入 `box` 之后）

在 corenetic 上执行（`ssh hph@192.168.11.44`）：

```bash
# 1. Thor1 是否空闲：recording.state 必须是 idle。connecting / armed 也说明录制器进程还在，
#    部署会重启网关并杀掉它；标定会话、跟踪仪会话也必须是 false
curl -s http://192.168.111.122:8765/api/snapshot | python3 -c "import json,sys; d=json.load(sys.stdin); \
print(d['recording']['state'], d['calibrationSession']['active'], d['trackerMountSession']['active'], \
d['deployment']['release']['commit'][:8])"

# 2. 部署副本更新到 box 最新（页面不受影响：它提供的是上次构建的 dist/）
cd ~/lerobot-box-deploy
git fetch origin box && git merge --ff-only origin/box && git submodule update --init --recursive
git log -1 --oneline && git status --short          # status 必须为空

# 3. 先 dry-run：看守卫结论，以及 rsync 要删 Thor 上的哪些文件（*deleting 开头的行）
bash run/sync_to_target.sh thor1 -n 2>&1 | grep -E '^\*deleting|WARNING|ERROR|==>'

# 4. 部署：同步文件并重启 thor1 的网关（前端单独重启，见第 5 步）
bash run/deploy.sh thor1 --no-frontend

# 5. 网关部署成功后再构建前端并重启，前后端就是同一个提交。第 4 步失败时不要做这一步：
#    页面继续停在旧版本，正好和旧网关一致
(cd tools/data_collection_gui/frontend && npm run build) && systemctl --user restart lerobot-frontend-thor1

# 6. 核对：通过前端读到的网关版本 == 部署副本的 HEAD
curl -s http://127.0.0.1:5180/api/snapshot | python3 -c "import json,sys; \
print(json.load(sys.stdin)['deployment']['release'])"; git rev-parse HEAD
```

最后告诉现场的人：录制器要重新 **Connect**。

`deploy.sh` 依次做这些事：
- 生产守卫：工作区必须干净，HEAD 必须在 `origin/box` 上；
- 主机名核对：通过 ssh 确认这个地址真的是 thor1（`upai-pro03`）；
- 版本单调检查：sidecar schema 不能比 Thor 上的旧，标定指针和 world reference 必须一致；
- 用 `rsync --delete-delay` 把 Thor 的生产目录镜像成部署副本的样子（排除 `outputs/`、`.venv*`、`node_modules/` 等）；
- 同步 EE 轨迹标定输入（见下面第一条）；
- 重启网关，并把提交、部署人和时间写进 `DEPLOYED.json`。GUI 标题和 `/api/snapshot` 的
  `deployment.release` 显示的就是它。

dry-run 和部署时会看到的正常现象：
- `WARN: calibration summary not found: .../outputs/calibration/thor_gmsl2_*_0720/summary.json`：`outputs/`
  不进 git，部署副本里没有这些文件，所以这一步被跳过，Thor 上已有的标定不受影响。
  需要强制同步时，先把这两个目录拷进部署副本，再加 `REQUIRE_EE_CALIBRATION=1`。
- `cannot delete non-empty directory: ...`：Thor 上有本地生成的文件，rsync 不删，没关系。
- `*deleting` 列出的应当只有运行时文件（如 `box_sdk.log`）。如果列出了源码、标定或配置文件，**先停下**，
  查清楚是谁把它们放到了 Thor 上（AGENTS.md 第 4 节：标定数据不经 maintainer 不许改）。

建议在部署副本里设一次 `git config user.name <你的名字>`，`DEPLOYED.json` 的 `by` 字段才会写明是谁部署的。

## 3. maintainer 部署未合入的提交（`--maintainer`）

只在必须先上线、来不及合入的情况下用（例如现场的紧急修复）。`--maintainer` 只放宽「HEAD 必须在 `origin/box` 上」
这一条，其他检查不变：工作区仍然必须干净，主机名、版本单调检查照做。

```bash
cd ~/lerobot-box-deploy
git fetch origin <分支> && git switch --detach origin/<分支>     # 只部署已经 push 的提交
bash run/deploy.sh thor1 --no-frontend --maintainer
(cd tools/data_collection_gui/frontend && npm run build) && systemctl --user restart lerobot-frontend-thor1
```

- `DEPLOYED.json` 会记下 `"onBox": false, "maintainerOverride": true`，GUI 上大家都能看到生产跑的不是 `box`。
- PR 合入后，**按第 2 节从 `box` 重新部署一次**，再 `git switch box` 把部署副本切回去。
  不要让生产长期停在一个 `box` 里没有的提交上，下一个人从 `box` 部署时会把它覆盖掉。
- agent 永远不用 `--maintainer`（AGENTS.md 第 3 节）。

## 4. 回滚到 `box` 上更早的提交

```bash
cd ~/lerobot-box-deploy
git switch --detach <提交>                 # 必须是 origin/box 历史上的提交，守卫才放行
bash run/deploy.sh thor1 --no-frontend && (cd tools/data_collection_gui/frontend && npm run build) \
  && systemctl --user restart lerobot-frontend-thor1
git switch box                              # 修好后按第 2 节重新部署
```

## 5. 学生前端服务（corenetic）

服务文件 `~/.config/systemd/user/lerobot-frontend-thor1.service`：

```ini
[Unit]
Description=LeRobot GUI frontend for thor1 (students open http://<this machine>:5180/)
After=network-online.target

[Service]
WorkingDirectory=%h/lerobot-box-deploy/tools/data_collection_gui/frontend
Environment=PATH=%h/.nvm/versions/node/v24.15.0/bin:/usr/bin:/bin
Environment=GUI_API_TARGET=http://192.168.111.122:8765
ExecStart=%h/.nvm/versions/node/v24.15.0/bin/npm run preview -- --port 5180 --strictPort
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
```

```bash
(cd ~/lerobot-box-deploy/tools/data_collection_gui/frontend && npm run build)           # preview 要有 dist/
systemctl --user daemon-reload && systemctl --user enable --now lerobot-frontend-thor1   # 安装或改动后
sudo loginctl enable-linger hph      # 一次性：开机就启动，hph 全部退出登录后也不停
systemctl --user status lerobot-frontend-thor1
journalctl --user -u lerobot-frontend-thor1 -n 50
```

- **生产不要用 `npm run dev`。** vite dev 会监听源码并热更新：部署副本里 `git merge` 一下、或者有人顺手改个文件，
  学生页面立刻就变，而网关要到 `deploy.sh` 跑完才更新，中间前后端版本不一致会出假报错；如果部署被拒，
  这种错位会一直持续。`vite preview` 只提供构建好的 `dist/`，`/api` 照样代理到 `GUI_API_TARGET`。
- `GUI_API_TARGET` 必须写明。没设时 vite 默认连 `127.0.0.1:8765`（本机的网关），不会连生产，这样开发时忘设变量
  也不会让未合入的页面去操作生产。
- `ExecStart` 必须写 npm 的绝对路径：systemd 找可执行文件时不看 `Environment=PATH`。升级 node 后两处路径都要改。
- 如果没开 linger，corenetic 重启后、或 hph 全部退出登录后页面就打不开了。学生看到的现象很像网关坏了。
- 前端依赖装在部署副本里（`tools/data_collection_gui/frontend/node_modules`）。`package-lock.json` 变了之后，
  在那个目录里先执行 `npm ci`，再构建、重启服务。

## 6. 重建部署副本（换机器或副本损坏时）

corenetic 直接从 GitHub 下载很慢，git-lfs 又不走代理，checkout 会卡很久。所以在网络好的机器上（有代理的话设好
`http_proxy` / `https_proxy`）准备好，再用 rsync 拷过去，局域网大约 100 MB/s：

```bash
# 在一台已有本仓库的机器上（$REPO 是那份仓库，LFS 对象和子模块都直接从它拷，不用下载）
L=/tmp/lerobot-box-deploy
git clone --no-checkout --no-hardlinks "$REPO" "$L" && cd "$L"
git fetch "$REPO" +refs/remotes/origin/box:refs/remotes/origin/box
git remote set-url origin git@github.com:linkage-x/lerobot.git
mkdir -p .git/lfs && cp -a "$REPO/.git/lfs/objects" .git/lfs/
git checkout -B box origin/box && git branch -u origin/box
git config submodule.third_party/opencv_kalibr.url "$REPO/.git/modules/third_party/opencv_kalibr"
git config submodule.tools/thor/box_sdk.url "$REPO/.git/modules/tools/thor/box_sdk"
git -c protocol.file.allow=always submodule update --init --recursive && git submodule sync --recursive
(cd tools/data_collection_gui/frontend && npm ci)
git status --short                          # 必须为空
rsync -a "$L/" hph@192.168.11.44:/home/hph/lerobot-box-deploy/
```
