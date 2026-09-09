#!/usr/bin/env bash
set -euo pipefail

export FRANKA_REPLAY_BACKEND=native
exec /home/nvidia/box_api/scripts/run_franka_non_rt_replay_otg.sh "$@"
