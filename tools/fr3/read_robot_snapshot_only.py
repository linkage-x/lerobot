#!/usr/bin/env python3
"""Read FR3 state over SSH. No Panda controller, setters, recovery or motion."""
import argparse
import json
import subprocess
from pathlib import Path

REMOTE = '''
import json
from datetime import datetime, timezone
from panda_py import libfranka
r = libfranka.Robot("192.168.11.102", libfranka.RealtimeConfig.kIgnore)
s = r.read_once()
o = {k: list(getattr(s, k)) for k in ["q", "dq", "O_T_EE", "F_T_EE", "F_x_Cee", "I_ee", "tau_ext_hat_filtered", "O_F_ext_hat_K"]}
o.update(timestamp_utc=datetime.now(timezone.utc).isoformat(), read_only=True,
         motion_commands_sent=0, server_version=r.server_version(),
         robot_mode=str(s.robot_mode), current_errors=str(s.current_errors),
         last_motion_errors=str(s.last_motion_errors), m_ee=s.m_ee, m_load=s.m_load)
print(json.dumps(o, allow_nan=False))
'''

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    command = ('env LD_LIBRARY_PATH=/home/nvidia/lerobot/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib:'
               '/home/nvidia/Code/infer/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib '
               '/home/nvidia/Code/infer/.venv-fr3/bin/python -')
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                             'nvidia@192.168.111.122', command], input=REMOTE, text=True,
                            capture_output=True, timeout=20, check=True)
    state = json.loads(result.stdout)
    assert state['read_only'] and state['motion_commands_sent'] == 0
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(state, f, indent=2, allow_nan=False)
    print(json.dumps(state, indent=2))

if __name__ == '__main__':
    main()
