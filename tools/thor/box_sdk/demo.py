import os
import sys
import time

from box_sdk import Box


def _local_controller_so() -> str | None:
    root = os.path.dirname(os.path.abspath(__file__))
    so = os.path.join(root, "lib", "libbox_controller.so")
    return so if os.path.isfile(so) else None


def main():
    bind_port = int(sys.argv[1]) if len(sys.argv) >= 2 else 15000
    remote_port = int(sys.argv[2]) if len(sys.argv) >= 3 else 15000

    box = Box(so_path=_local_controller_so())
    box.start(bind_ip="0.0.0.0", bind_port=bind_port, remote_ip="192.168.2.60", remote_port=remote_port)

    rc = box.set_mode(1)
    print("set_mode:", rc, box.err_str(rc))
    time.sleep(1)

    rc = box.set_clamp_pos(0.004)
    print("set_clamp_pos:", rc, box.err_str(rc))
    time.sleep(1)

    rc = box.set_mode(0)
    print("set_mode:", rc, box.err_str(rc))
    time.sleep(1)

    print("running... Ctrl+C to exit")
    try:
        while True:
            rc, snap = box.get_sensor_cache()
            if rc == 0 and snap.valid:
                print("gripper_data.distance:", snap.data.gripper_data.distance)
            else:
                print("get_sensor_cache failed:", rc, box.err_str(rc))
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    box.stop()
    box.close()


if __name__ == "__main__":
    main()

