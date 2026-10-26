"""A stand-in for panda_py in the dual-FR3 runtime tests: a 1 kHz control loop round nominal joints.

What ``RealFr3Driver`` uses of panda_py, and no more: ``Panda(hostname)`` with ``get_robot().read_once()``,
``get_state()`` (a copy the control loop refreshes, frozen while no controller runs, as panda_py's),
``move_to_joint_position``, ``start_controller`` / ``stop_controller``, ``raise_error``; and
``controllers.JointPosition`` with panda_py's per-millisecond set-point filter.

The hostname's suffix sets the arm's behaviour: ``"10.0.0.1"`` is a healthy arm; ``#reflex@S`` makes the
control loop die on libfranka's error S seconds after the controller started, ``#stall@S`` makes it stop
without one, ``#m_ee=KG`` is what Desk holds, ``#q=J:RAD`` starts joint J (1-based) there.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

Q_READY = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
#: the joints follow the filtered set point with this time constant (an arm under PD control)
TRACK_TAU_S = 0.01


class Duration:
    def __init__(self, seconds: float):
        self._s = seconds

    def to_sec(self) -> float:
        return self._s


@dataclass
class RobotState:
    q: list[float]
    time: Duration
    m_ee: float = 0.0
    F_x_Cee: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # noqa: N815


class JointPosition:
    def __init__(self, stiffness=None, damping=None, filter_coeff: float = 1.0):
        del stiffness, damping
        self.filter_coeff = float(filter_coeff)
        self.lock = threading.Lock()
        self.target: np.ndarray | None = None
        self.velocity = np.zeros(7)
        self.calls = 0

    def set_control(self, position, velocity=None) -> None:
        with self.lock:
            self.target = np.asarray(position, dtype=float).copy()
            self.velocity = np.zeros(7) if velocity is None else np.asarray(velocity, dtype=float).copy()
            self.calls += 1


controllers = SimpleNamespace(JointPosition=JointPosition)


class Panda:
    def __init__(self, hostname: str, name: str = "panda"):
        del name
        self.hostname = hostname
        self.behaviour: dict[str, float] = {}
        self.q = Q_READY.copy()
        self.m_ee = 0.0
        for item in hostname.split("#")[1:]:
            key, _, value = item.replace("=", "@").partition("@")
            if key == "q":
                j, _, rad = value.partition(":")
                self.q[int(j) - 1] = float(rad)
            elif key == "m_ee":
                self.m_ee = float(value)
            else:
                self.behaviour[key] = float(value)
        # a robot clock that started a while ago, different per arm
        self._boot = time.monotonic() - 500.0 - (sum(map(ord, hostname)) % 97)
        self._error: str | None = None
        self._ctrl: JointPosition | None = None
        self._run = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = self._read()

    def _read(self) -> RobotState:
        t = np.floor((time.monotonic() - self._boot) * 1e3) / 1e3
        return RobotState(q=self.q.tolist(), time=Duration(t), m_ee=self.m_ee)

    def get_robot(self):
        return SimpleNamespace(read_once=self._read)

    def get_state(self) -> RobotState:
        return self._state

    def move_to_joint_position(self, positions, speed_factor: float = 0.2, **_) -> bool:
        del speed_factor
        self.q = np.asarray(positions, dtype=float).copy()
        self._state = self._read()
        return True

    def start_controller(self, ctrl: JointPosition) -> None:
        self.stop_controller()
        ctrl.set_control(self.q, np.zeros(7))
        self._ctrl = ctrl
        self._run.set()
        self._thread = threading.Thread(target=self._loop, args=(ctrl,), daemon=True)
        self._thread.start()

    def _loop(self, ctrl: JointPosition) -> None:
        started = time.monotonic()
        q_d = self.q.copy()
        last = started
        while self._run.is_set():
            now = time.monotonic()
            if now - started > self.behaviour.get("reflex", np.inf):
                self._error = "Control loop interrupted: cartesian_reflex (fake)"
                return
            if now - started > self.behaviour.get("stall", np.inf):
                return
            with ctrl.lock:
                target = ctrl.target.copy()
            ticks = max(1, round((now - last) * 1e3))
            last = now
            for _ in range(ticks):
                q_d = q_d + ctrl.filter_coeff * (target - q_d)
            self.q = self.q + (1.0 - np.exp(-(ticks * 1e-3) / TRACK_TAU_S)) * (q_d - self.q)
            self._state = self._read()
            time.sleep(0.001)

    def raise_error(self) -> None:
        if self._error:
            error, self._error = self._error, None
            raise RuntimeError(error)

    def stop_controller(self) -> None:
        self._run.clear()
        if self._thread is not None:
            self._thread.join()
        self._thread = None
        self._ctrl = None
