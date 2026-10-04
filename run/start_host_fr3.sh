#!/usr/bin/env bash
# Called by deploy after Thor's old recorder has stopped. No USB/FCI is opened.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
remote="nvidia@192.168.111.122"
remote_dir="/home/nvidia/lerobot"
python_bin="$repo_root/.venv-fr3/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "ERROR: host .venv-fr3 missing. See docs/thor_fr3_teleoperation.md host setup." >&2
  exit 1
fi
export PYTHONPATH="$repo_root/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
cmeel_lib="$repo_root/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib"
export LD_LIBRARY_PATH="$repo_root/.venv-fr3/lib:/usr/local/lib:$cmeel_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p outputs/logs/fr3_teleop outputs/secrets
chmod 700 outputs/secrets
# Refuse to kill a busy controller. A live connection cannot answer this request.
"$python_bin" -m tools.thor.fr3_host --shutdown-if-idle
sleep 0.2
if "$python_bin" -c 'import yaml; from pathlib import Path; c=yaml.safe_load(Path("tools/thor/gmsl2/thor_fr3_teleop.yaml").read_text()); raise SystemExit(0 if c["fr3_teleop"].get("host_performance_governor", False) else 1)'; then
  if ! bash run/setup_host_fr3_permissions.sh --performance; then
    echo "WARN: could not set host performance governor; fix host scheduling before F." >&2
  fi
fi
umask 077
if [[ ! -s outputs/secrets/fr3_host.token ]]; then
  "$python_bin" -c 'import secrets; from pathlib import Path; Path("outputs/secrets/fr3_host.token").write_text(secrets.token_hex(32))'
fi
chmod 600 outputs/secrets/fr3_host.token
ssh -o BatchMode=yes -o ConnectTimeout=5 "$remote" \
  "umask 077; mkdir -p '$remote_dir/outputs/secrets'; chmod 700 '$remote_dir/outputs/secrets'; cat > '$remote_dir/outputs/secrets/fr3_host.token'; chmod 600 '$remote_dir/outputs/secrets/fr3_host.token'" \
  < outputs/secrets/fr3_host.token
# Only our dedicated control socket is closed; other SSH sessions are untouched.
control_socket="$repo_root/outputs/secrets/fr3_ssh.sock"
if ssh -S "$control_socket" -O check "$remote" >/dev/null 2>&1; then
  ssh -S "$control_socket" -O exit "$remote"
fi
ssh -M -S "$control_socket" -fNT \
  -o BatchMode=yes -o ExitOnForwardFailure=yes -o ConnectTimeout=5 \
  -o ServerAliveInterval=2 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:18766:127.0.0.1:18766 "$remote"
setsid "$python_bin" -u -m tools.thor.fr3_host \
  </dev/null >outputs/logs/fr3_teleop/host.log 2>&1 &
host_pid=$!
disown
if [[ "$(ulimit -r)" -lt 99 ]] && [[ -f outputs/secrets/host_fifo_approved ]]; then
  if ! sudo -n prlimit --pid "$host_pid" --rtprio=99:99; then
    echo "WARN: host service lacks FIFO permission; log out/in after host scheduling setup before F." >&2
  fi
fi
for _ in 1 2 3 4 5; do
  if "$python_bin" -m tools.thor.fr3_host --health 2>/dev/null; then
    echo "==> Host FR3 service ready; SpaceMouse and FCI open only when you press F."
    exit 0
  fi
  sleep 0.3
done
echo "ERROR: host bridge did not start; see outputs/logs/fr3_teleop/host.log" >&2
exit 1
