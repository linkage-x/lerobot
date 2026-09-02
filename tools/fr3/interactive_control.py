"""The control channel a long-running rig process exposes while it is moving the arm.

One implementation, three callers-to-be: interactive inference, the MuJoCo DAgger rehearsal,
and whatever comes next. It lives apart from the inference runtime because the rehearsal must
be startable without the policy stack -- and because a second copy of "what does pressing t
mean" is a second thing that can disagree with the GUI about it.

Three backends, in the order they are tried: sshkeyboard, a raw-mode TTY, and stdin-as-a-pipe.
The last one is how the browser reaches this process: the gateway holds the runtime's stdin and
writes words, not keystrokes, so the page and the terminal drive the same session with the same
alphabet.
"""

from __future__ import annotations

import json
import os
import select
import stat as stat_module
import sys
import termios
import threading
import tty
from typing import Any


class InteractiveRolloutKeyboard:
    def __init__(
        self,
        *,
        start_key: str,
        stop_key: str,
        home_key: str,
        quit_key: str,
        takeover_key: str | None = None,
    ):
        self.start_key = self._normalize_key(start_key)
        self.stop_key = self._normalize_key(stop_key)
        self.home_key = self._normalize_key(home_key)
        self.quit_key = self._normalize_key(quit_key)
        # None when DAgger takeover is off, so the key stays free for whatever else the
        # operator has bound and a stray press cannot hand a live rollout to a device that
        # was never connected.
        self.takeover_key = self._normalize_key(takeover_key) if takeover_key else None
        self.start_requested = threading.Event()
        self.stop_requested = threading.Event()
        self.home_requested = threading.Event()
        self.scene_reset_requested = threading.Event()
        # A calibration probe: one commanded point, taken so a still can be tied to the base
        # frame. Separate from scene reset because they are different motions with different
        # QC, and because an operator watching the arm should be told which one it is doing.
        self.probe_pose_requested = threading.Event()
        self.quit_requested = threading.Event()
        self._json_requests = {
            'scene_reset': self.scene_reset_requested,
            'probe_pose': self.probe_pose_requested,
        }
        self._json_payloads: dict[str, dict[str, Any]] = {}
        self._json_lock = threading.Lock()
        # The manual override, not the ordinary way in: takeover normally engages itself when
        # the SpaceMouse moves (see tools/fr3/dagger_takeover.py). This latch is for an operator
        # who wants the arm held still without touching the device.
        #
        # A toggle, not a hold. sshkeyboard and the cbreak fallback both report presses, not
        # releases, so a dead-man's switch cannot be built on this channel -- and one built on
        # key repeat would drop control every time the operator's hand paused.
        self.takeover_engaged = threading.Event()
        self._thread: threading.Thread | None = None
        self._stop_listening = None

    @staticmethod
    def _normalize_key(key: str) -> str:
        return str(key).strip().lower()

    def _on_press(self, key: str) -> None:
        key_name = self._normalize_key(key)
        if key_name == self.start_key:
            print('[INFO] interactive_key=start')
            self.start_requested.set()
            self.stop_requested.clear()
        elif key_name == self.stop_key:
            print('[INFO] interactive_key=stop_current_rollout')
            self.stop_requested.set()
        elif key_name == self.home_key:
            print('[INFO] interactive_key=move_to_start')
            self.home_requested.set()
        elif self.takeover_key is not None and key_name == self.takeover_key:
            if self.takeover_engaged.is_set():
                self.takeover_engaged.clear()
                print('[INFO] interactive_key=takeover_release')
            else:
                self.takeover_engaged.set()
                print('[INFO] interactive_key=takeover_engage')
        elif key_name == self.quit_key:
            print('[INFO] interactive_key=quit')
            self.quit_requested.set()
            self.stop_requested.set()
            self.start_requested.set()
            # Quitting drops the takeover with it. Leaving it engaged would hand the next
            # process state, and there is no next process to hand it to.
            self.takeover_engaged.clear()

    def _listen_keyboard_loop(self, listen_keyboard: Any) -> None:
        try:
            listen_keyboard(on_press=self._on_press, sequential=False)
        except TypeError:
            listen_keyboard(on_press=self._on_press)

    # Words accepted alongside the bare keys on the pipe channel, so a caller that is not a
    # keyboard can say what it means. The keys stay valid because the GUI and the terminal
    # then speak the same alphabet, which is one less thing to keep in step.
    _PIPE_COMMAND_WORDS = {
        'start': 'start',
        'stop': 'stop',
        'home': 'home',
        'quit': 'quit',
        'takeover': 'takeover',
        'scene_reset': 'scene_reset',
        'probe_pose': 'probe_pose',
    }

    def _listen_pipe_loop(self) -> None:
        """One command per line, for a caller that is a program rather than a terminal.

        The GUI reaches this path: it holds the runtime's stdin as a pipe, which no keyboard
        backend can read -- sshkeyboard and the cbreak fallback both want a terminal. This is
        the same control shape the recorder already exposes to the gateway (see
        _write_recorder_stdin), so the two long-running rig processes are driven the same way.

        Reading returns '' only at EOF, which is the writer closing the pipe. That is a quit:
        whoever was steering this rollout is gone, and continuing to move the arm with no
        controller is the one outcome worth avoiding.
        """
        for raw_line in sys.stdin:
            if self.quit_requested.is_set():
                return
            raw_command = raw_line.strip()
            command = raw_command.lower()
            if not command:
                continue
            resolved = self._PIPE_COMMAND_WORDS.get(command.split(maxsplit=1)[0])
            if resolved == 'start':
                self._on_press(self.start_key)
            elif resolved == 'stop':
                self._on_press(self.stop_key)
            elif resolved == 'home':
                self._on_press(self.home_key)
            elif resolved == 'quit':
                self._on_press(self.quit_key)
            elif resolved == 'takeover':
                if self.takeover_key is None:
                    print('[WARN] interactive_pipe_takeover_ignored reason=takeover_disabled')
                else:
                    self._on_press(self.takeover_key)
            elif resolved in self._json_requests:
                self._queue_json_command(resolved, raw_command)
            elif len(command) == 1:
                self._on_press(command)
            else:
                print(f'[WARN] interactive_pipe_command_ignored={command!r}')
        print('[INFO] interactive_pipe=closed_by_peer')
        self._on_press(self.quit_key)

    def _queue_json_command(self, name: str, raw_command: str) -> None:
        """Accept one command that carries coordinates rather than being a bare word.

        The payload replaces any earlier one instead of queueing behind it: these are requests
        to move the arm, and the operator's most recent instruction is the only one they still
        mean.
        """

        parts = raw_command.split(maxsplit=1)
        payload_text = parts[1] if len(parts) == 2 else '{}'
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            print(f'[WARN] interactive_pipe_{name}_ignored reason=bad_json details={exc}')
            return
        if not isinstance(payload, dict):
            print(f'[WARN] interactive_pipe_{name}_ignored reason=payload_not_object')
            return
        with self._json_lock:
            self._json_payloads[name] = payload
        print(f'[INFO] interactive_key={name}')
        self._json_requests[name].set()

    @staticmethod
    def _stdin_is_pipe() -> bool:
        """Whether stdin can actually deliver commands without being a terminal.

        A pipe or a socket has a writer on the other end; /dev/null (which is what
        subprocess.DEVNULL gives) is a character device that returns EOF immediately, and
        treating that as a control channel would quit the rollout the moment it started.
        """
        try:
            mode = os.fstat(sys.stdin.fileno()).st_mode
        except (OSError, ValueError):
            return False
        return stat_module.S_ISFIFO(mode) or stat_module.S_ISSOCK(mode)

    def _listen_stdin_loop(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError('Interactive rollout fallback requires a TTY stdin.')

        fd = sys.stdin.fileno()
        original_termios = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self.quit_requested.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not readable:
                    continue
                char = sys.stdin.read(1)
                if char == '\x03':
                    self._on_press(self.quit_key)
                    break
                if char == '\x1b':
                    self._on_press(self.quit_key)
                    continue
                self._on_press(char)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, original_termios)

    def start(self) -> None:
        # Checked before sshkeyboard, not after: sshkeyboard grabs the *terminal*, so on a
        # process whose stdin is a pipe it would install a backend that can never receive a
        # key while the channel that can sits unused.
        if not sys.stdin.isatty() and self._stdin_is_pipe():
            self._thread = threading.Thread(target=self._listen_pipe_loop, daemon=True)
            self._thread.start()
            backend = 'pipe'
            print(
                '[INFO] interactive_rollouts=enabled '
                f'keyboard_backend={backend} '
                f"start_key='{self.start_key}' stop_key='{self.stop_key}' "
                f"home_key='{self.home_key}' quit_key='{self.quit_key}'"
                + ('' if self.takeover_key is None else f" takeover_key='{self.takeover_key}'")
            )
            return

        try:
            from sshkeyboard import listen_keyboard, stop_listening
        except ImportError:
            if not sys.stdin.isatty():
                raise RuntimeError(
                    'Interactive rollout mode requires `sshkeyboard`, a TTY stdin, or a pipe '
                    'on stdin. Install sshkeyboard in the runtime, run with an interactive '
                    'TTY, or drive it from a parent process that holds stdin open.'
                )
            self._thread = threading.Thread(target=self._listen_stdin_loop, daemon=True)
            self._thread.start()
            backend = 'stdin'
        else:
            self._stop_listening = stop_listening
            self._thread = threading.Thread(
                target=self._listen_keyboard_loop,
                args=(listen_keyboard,),
                daemon=True,
            )
            self._thread.start()
            backend = 'sshkeyboard'

        print(
            '[INFO] interactive_rollouts=enabled '
            f'keyboard_backend={backend} '
            f"start_key='{self.start_key}' stop_key='{self.stop_key}' "
            f"home_key='{self.home_key}' quit_key='{self.quit_key}'"
            + ('' if self.takeover_key is None else f" takeover_key='{self.takeover_key}'")
        )

    def wait_for_command(self, *, arm_at_start: bool) -> str:
        """Block until the operator asks for the next thing: 'start', 'home' or 'quit'.

        Every request flag is cleared on entry, so a key pressed *during* a rollout is dropped
        rather than acted on the instant that rollout ends. That matters most for home: an arm
        that begins moving on its own seconds after the operator stopped watching is the exact
        outcome an interactive mode exists to prevent.

        `arm_at_start` is printed rather than assumed. The launcher homes the arm once, before
        this process exists; from the second rollout onwards the arm is wherever the last one
        left it, and a banner that keeps claiming otherwise is how an operator ends up starting
        a rollout from a pose the dataset frame was never anchored to.
        """
        self.start_requested.clear()
        self.stop_requested.clear()
        self.home_requested.clear()
        self.scene_reset_requested.clear()
        self.probe_pose_requested.clear()
        # Every rollout begins under the policy. A latch left set from the last one would hand
        # the arm to a SpaceMouse nobody is holding, at the moment a fresh rollout starts
        # moving. (Automatic takeover cannot be left behind this way: it is a property of what
        # the device is doing right now, not a flag.)
        self.takeover_engaged.clear()
        print(
            '[INFO] interactive_waiting_for_start '
            f'arm_at_start={1 if arm_at_start else 0} '
            f"press '{self.start_key}' to start, '{self.home_key}' to move to start, "
            f"'{self.quit_key}' to quit."
        )
        while not self.quit_requested.is_set():
            if self.start_requested.wait(timeout=0.1):
                self.start_requested.clear()
                self.stop_requested.clear()
                # Quit sets start_requested too, to break exactly this wait. Without the
                # re-check it reads as a start, and the session announces a rollout index it
                # then immediately abandons.
                if self.quit_requested.is_set():
                    break
                return 'start'
            if self.home_requested.is_set():
                self.home_requested.clear()
                return 'home'
            for name, event in self._json_requests.items():
                if event.is_set():
                    event.clear()
                    return name
        return 'quit'

    def _pop_json_payload(self, name: str) -> dict[str, Any] | None:
        with self._json_lock:
            return self._json_payloads.pop(name, None)

    def pop_scene_reset_payload(self) -> dict[str, Any] | None:
        return self._pop_json_payload('scene_reset')

    def pop_probe_pose_payload(self) -> dict[str, Any] | None:
        return self._pop_json_payload('probe_pose')

    def takeover_is_engaged(self) -> bool:
        return self.takeover_key is not None and self.takeover_engaged.is_set()

    def should_stop_rollout(self) -> bool:
        return self.stop_requested.is_set() or self.quit_requested.is_set()

    def close(self) -> None:
        self.quit_requested.set()
        if callable(self._stop_listening):
            try:
                self._stop_listening()
            except Exception:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
