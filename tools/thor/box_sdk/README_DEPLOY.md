## box_collection_sdk Python 交付包

本目录用于对外交付（Python 调用）。

### 目录内容

- `box_collection_sdk-*.whl`：安装包（wheel）
- `demo.py`：最小示例（仅演示 start/set_mode/读缓存）

完整 `release_bundle` 打包后，`demo.py` 与 `README_DEPLOY.md` 位于 **release_bundle 根目录**（与 `setup_env.sh` 同级）；`python/` 下仅放 wheel。

### 1) 安装

建议在虚拟环境中安装：

```bash
python3 -m venv venv
. venv/bin/activate
pip install ./box_collection_sdk-*.whl
```

如果目标机器没有 venv（缺少 `python3-venv`），也可以：

```bash
pip install --user ./box_collection_sdk-*.whl
```

### 2) 运行示例

在交付目录或已解压的 `release_bundle` 根目录下（已 `source setup_env.sh` 时）：

```bash
python3 demo.py 5000 5000 192.168.2.60
```

参数：
- 第 1 个：本地 bind 端口
- 第 2 个：远端端口
- 第 3 个：远端 IP

### 3) 常见问题

- **ImportError / OSError 加载 .so 失败**：请确认机器架构一致（例如都是 x86_64 Linux），以及 wheel 文件完整未损坏。
- **端口占用**：换一个 bind 端口，比如 `55000`。
