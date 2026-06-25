"""
box_collection_sdk 多设备交互示例（对应 host_integration_guide.md）。

使用流程：
  1. 扫描设备：C++ 广播发现（UDP 15001），列出当前在线设备及数量；
  2. 按序号选择要连接（控制）的设备；
  3. 之后一切下发与读取都以 device_id 为准（set_*/get_sensor_cache 按 id）。

用法：
  python3 demo.py [bind_port]
"""

import readline
import sys
import time

from box_sdk import (
    DATA_PORT,
    DISCOVERY_PORT,
    Box,
    DiscoveredDevice,
    DiscoveryKeepAlive,
    discover,
)

KEEPALIVE_INTERVAL_S = 3.0
SCAN_TIMEOUT_S = 2.5
WATCH_RATE_HZ = 20
WATCH_PERIOD_S = 1.0 / WATCH_RATE_HZ

COMMANDS = (
    "list",
    "ids",
    "watch",
    "get_mode",
    "set_mode",
    "set_clamp_position",
    "set_trigger_zero",
    "cali6d",
    "calitouch",
    "help",
    "quit",
    "exit",
    "q",
)

COMMANDS_WITH_DEVICE_ID = frozenset(
    {
        "watch",
        "get_mode",
        "set_mode",
        "set_clamp_position",
        "set_trigger_zero",
        "cali6d",
        "calitouch",
    }
)


class _CompletionState:
    def __init__(self) -> None:
        self.box: Box | None = None
        self.connected: list[DiscoveredDevice] = []
        self.devices: list[DiscoveredDevice] = []
        self.matches: list[str] = []

    def device_id_strings(self) -> list[str]:
        ids: set[int] = set()
        for d in self.connected:
            ids.add(d.device_id)
        for d in self.devices:
            ids.add(d.device_id)
        if self.box is not None:
            try:
                ids.update(self.box.get_device_ids())
                ids.update(self.box.get_known_device_ids())
            except Exception:
                pass
        return [str(i) for i in sorted(ids)]


_completion_state = _CompletionState()


def _match_candidates(text: str) -> list[str]:
    buffer = readline.get_line_buffer()
    begidx = readline.get_begidx()
    line_before = buffer[:begidx]
    stripped = line_before.rstrip()
    parts = stripped.split()
    ends_with_space = len(line_before) > len(stripped)

    if not parts or (len(parts) == 1 and not ends_with_space):
        return [cmd for cmd in COMMANDS if cmd.startswith(text)]

    cmd = parts[0].lower()
    device_ids = _completion_state.device_id_strings()

    if cmd in COMMANDS_WITH_DEVICE_ID:
        if (len(parts) == 1 and ends_with_space) or (len(parts) == 2 and not ends_with_space):
            return [did for did in device_ids if did.startswith(text)]

    if cmd == "set_mode":
        if (len(parts) == 2 and ends_with_space) or (len(parts) == 3 and not ends_with_space):
            return [v for v in ("0", "1") if v.startswith(text)]

    return []


def _demo_completer(text: str, state: int) -> str | None:
    if state == 0:
        _completion_state.matches = _match_candidates(text)
    if state < len(_completion_state.matches):
        return _completion_state.matches[state]
    return None


def setup_command_completion(
    box: Box,
    connected: list[DiscoveredDevice],
    devices: list[DiscoveredDevice],
) -> None:
    """启用命令行 Tab 补全（命令名、device_id；set_mode 第三参数可补 0/1）。"""
    _completion_state.box = box
    _completion_state.connected = connected
    _completion_state.devices = devices
    try:
        readline.set_completer(_demo_completer)
        readline.set_completer_delims(" \t\n")
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass


def scan_devices(timeout: float = SCAN_TIMEOUT_S, broadcast_addr: str = "255.255.255.255") -> list[DiscoveredDevice]:
    print(f"[scan] 广播 REQ → {broadcast_addr}:{DISCOVERY_PORT}（超时 {timeout}s）")
    return discover(timeout=timeout, broadcast_addr=broadcast_addr)


def print_devices(devices: list[DiscoveredDevice]):
    print(f"在线设备数量: {len(devices)}")
    for i, d in enumerate(devices):
        print(
            f"  [{i}] device_id={d.device_id}  sn={d.sn}  ip={d.ip}:{d.data_port}  "
            f"fw=0x{d.fw_version:04X}  uptime={d.uptime_ms}ms  caps={d.capability_names}"
        )


def ask_selection(devices: list[DiscoveredDevice]) -> list[DiscoveredDevice]:
    prompt = "请输入要连接的设备序号（逗号分隔，回车=全部）: "
    try:
        raw = input(prompt).strip()
    except EOFError:
        raw = ""
    if not raw:
        return list(devices)
    try:
        indices = [int(x) for x in raw.replace("，", ",").split(",") if x.strip()]
    except ValueError:
        print("输入无效，默认连接全部")
        return list(devices)
    valid: list[DiscoveredDevice] = []
    missing: list[int] = []
    for idx in indices:
        if 0 <= idx < len(devices):
            valid.append(devices[idx])
        else:
            missing.append(idx)
    if missing:
        print(f"忽略无效序号: {missing}")
    if not valid:
        print("未选中有效设备，默认连接全部")
        return list(devices)
    return valid


def register_devices(box: Box, selected: list[DiscoveredDevice]) -> None:
    for d in selected:
        rc = box.register_device(d.device_id, d.ip, d.data_port)
        if rc != 0:
            print(f"[warn] register_device({d.device_id}, {d.ip}:{d.data_port}) -> {rc} {box.err_str(rc)}")


MENU = """
可用命令（均以 device_id 为准）:
  list                      重新扫描并列出在线设备
  ids                       显示已上报(有缓存)/已登记(可下发)的设备 id
  watch      <id>            以 20Hz 持续打印传感器缓存（含触觉 total_force，Ctrl+C 返回菜单）
  get_mode   <id>            查询工作模式（0=默认控制 1=自定义控制）
  set_mode   <id> <0|1>     设置工作模式
  set_clamp_position <id> <pos_m>  夹爪开口位置(米)
  set_trigger_zero <id>     设置触发零点
  cali6d     <id>           下发六维力校准（单次）
  calitouch  <id>           下发触觉校准（单次）
  help                      显示本菜单
  quit                      退出

提示：命令与 device_id 支持 Tab 补全（如 watch + Tab + 107… + Tab）。
"""


def _fmt_vec3(values, fmt: str = ".4f") -> str:
    return ",".join(f"{v:{fmt}}" for v in values)


def format_sensor_snapshot(snap) -> str:
    d = snap.data
    g = d.gripper_data
    imu = d.imu_data
    tr = d.trigger_data
    raw6 = d.six_d_force_data
    filt6 = d.six_d_force_data_filter
    t0 = snap.touch_sensor_data[0].total_force
    t1 = snap.touch_sensor_data[1].total_force
    return (
        f"dev={snap.device_id} idx={snap.liwp_index} ts={snap.liwp_timestemp} | "
        f"grip(ts={g.timestamp},dist={g.distance:.4f}) "
        f"trig(ts={tr.timestamp},dist={tr.distance:.4f}) "
        f"imu(ts={imu.timestamp},acc=[{_fmt_vec3(imu.acc)}],gyr=[{_fmt_vec3(imu.gyr)}],"
        f"rpy=[{imu.roll:.3f},{imu.pitch:.3f},{imu.yaw:.3f}],"
        f"quat=[{_fmt_vec3(imu.quat)}]) "
        f"6d(ts={raw6.timestamp},raw=[{_fmt_vec3(raw6.data)}],"
        f"filt=[{_fmt_vec3(filt6.data)}]) "
        f"touch0_total=[{t0.fx},{t0.fy},{t0.fz}] "
        f"touch1_total=[{t1.fx},{t1.fy},{t1.fz}]"
    )


def watch_loop(box: Box, device_id: int):
    print(f"dev {device_id} 以 {WATCH_RATE_HZ}Hz 持续打印传感器缓存（Ctrl+C 返回菜单）...")
    next_tick = time.monotonic()
    try:
        while True:
            rc, snap = box.get_sensor_cache(device_id)
            if rc == 0 and snap.valid:
                print(format_sensor_snapshot(snap))
            else:
                print(f"dev {device_id}: 无数据 rc={rc} {box.err_str(rc)}")
            next_tick += WATCH_PERIOD_S
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            elif sleep_s < -WATCH_PERIOD_S:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        print("\n返回菜单")


def _do_set_trigger_zero(box: Box, device_id: int) -> None:
    rc = box.set_trigger_zero(device_id=device_id)
    print(f"set_trigger_zero dev={device_id} -> {rc} {box.err_str(rc)}")


def handle_command(
    box: Box,
    connected: list[DiscoveredDevice],
    devices: list[DiscoveredDevice],
    line: str,
) -> bool:
    parts = line.split()
    if not parts:
        return True
    cmd = parts[0].lower()

    def need_id(n=2):
        if len(parts) < n:
            print("用法错误，缺少 device_id。输入 help 查看")
            return None
        try:
            return int(parts[1])
        except ValueError:
            print("device_id 必须是整数")
            return None

    if cmd in ("quit", "exit", "q"):
        return False
    if cmd == "help":
        print(MENU)
    elif cmd == "list":
        devices[:] = scan_devices()
        print_devices(devices)
    elif cmd == "ids":
        print("已上报(有缓存):", box.get_device_ids())
        print("已登记(可下发):", box.get_known_device_ids())
    elif cmd == "watch":
        did = need_id()
        if did is not None:
            watch_loop(box, did)
    elif cmd == "get_mode":
        did = need_id()
        if did is not None:
            rc, mode = box.get_mode(did)
            mode_name = "默认控制" if mode == 0 else "自定义控制" if mode == 1 else str(mode)
            print(f"get_mode dev={did} -> rc={rc} {box.err_str(rc)} mode={mode} ({mode_name})")
    elif cmd == "set_mode":
        did = need_id(3)
        if did is not None:
            rc = box.set_mode(int(parts[2]), did)
            print(f"set_mode({parts[2]}) dev={did} -> {rc} {box.err_str(rc)}")
    elif cmd == "set_clamp_position":
        did = need_id(3)
        if did is not None:
            rc = box.set_clamp_pos(float(parts[2]), device_id=did)
            print(f"set_clamp_position({parts[2]}) dev={did} -> {rc} {box.err_str(rc)}")
    elif cmd == "set_trigger_zero":
        did = need_id()
        if did is not None:
            _do_set_trigger_zero(box, did)
    elif cmd == "calitouch":
        did = need_id()
        if did is not None:
            rc = box.cali_touch_sensor(device_id=did)
            print(f"cali_touch_sensor dev={did} -> {rc} {box.err_str(rc)}")
    elif cmd == "cali6d":
        did = need_id()
        if did is not None:
            rc = box.cali_6d_force_sensor(device_id=did)
            print(f"cali_6d_force_sensor dev={did} -> {rc} {box.err_str(rc)}")
    else:
        print("未知命令，输入 help 查看")
    return True


def main():
    bind_port = int(sys.argv[1]) if len(sys.argv) >= 2 else DATA_PORT

    devices = scan_devices()
    print_devices(devices)
    if not devices:
        print("未发现任何设备，退出。请确认与设备同一子网、广播未被拦截。")
        return

    connected = ask_selection(devices)
    if not connected:
        print("未选择任何设备，退出。")
        return
    print("已选择连接:")
    for d in connected:
        print(f"  device_id={d.device_id}  ip={d.ip}:{d.data_port}  sn={d.sn}")

    box = Box()
    first = connected[0]
    box.start(
        bind_ip="0.0.0.0",
        bind_port=bind_port,
        remote_ip=first.ip,
        remote_port=first.data_port,
    )
    register_devices(box, connected)

    keepalive = None
    try:
        keepalive = DiscoveryKeepAlive(
            bind_port=DISCOVERY_PORT,
            interval_ms=int(KEEPALIVE_INTERVAL_S * 1000),
        )
        print(f"[keepalive] 已启动，每 {KEEPALIVE_INTERVAL_S}s 广播 REQ")
    except RuntimeError as e:
        print(f"[warn] 无法启动发现保活: {e}")

    print(MENU)
    setup_command_completion(box, connected, devices)
    try:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if not handle_command(box, connected, devices, line):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if keepalive is not None:
            keepalive.close()
        box.stop()
        box.close()


if __name__ == "__main__":
    main()
