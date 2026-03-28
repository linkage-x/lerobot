把 Hikrobot MVS Linux runtime 的 `.deb` 包放在这个目录。

当前已放入：

- `MvCamCtrlSDK_Runtime-4.7.0_x86_64_20251113.deb`
- `Samples/`，来自 host 的 `/opt/MVS/Samples`，用于补齐 runtime `.deb` 不包含的 Python 绑定

Git 管理建议：

- 这个仓库已经启用了 Git LFS。
- `docker/hik-sdk/*.deb` 已加入 `.gitattributes` 的 LFS 规则。
- 也就是说你可以直接 `git add docker/hik-sdk/MvCamCtrlSDK_Runtime-4.7.0_x86_64_20251113.deb`，提交的将是 LFS 指针，不是普通 Git 大对象。

构建示例：

```bash
cd /home/hanyu/Codes/lerobot

INSTALL_HIKROBOT_SDK=true \
docker compose -f docker/docker-compose.yml build lerobot-user
```

如果你需要 CUDA 镜像：

```bash
cd /home/hanyu/Codes/lerobot

INSTALL_HIKROBOT_SDK=true \
docker compose --profile sim --profile teleop --profile gpu -f docker/docker-compose.yml build lerobot-fr3-sim-teleop
```

容器内验证 MVS Python 绑定：

```bash
cd /home/hanyu/Codes/lerobot

docker compose -f docker/docker-compose.yml run --rm lerobot-user \
  bash -lc 'python -c "from MvImport import MvCameraControl_class as mvs; print(\"hik_mvs_ok\", hasattr(mvs, \"MvCamera\"))"'
```

遥操使用说明：

- 这次改动只集成了 Docker 层和 USB/udev 运行时权限。
- 如果宿主机上的 Hik 相机已经被暴露成 `/dev/video*`，可以继续按仓库现有的 `opencv` 相机配置方式使用。
- 如果你要像 `sensor_proto` 那样直接走 MVS Python API 采图，还需要再补一层 LeRobot 的 `hikrobot` camera backend。
