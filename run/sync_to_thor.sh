#!/usr/bin/env bash
# Compatibility wrapper for the target-aware incremental sync script.
# Usage:
#   bash run/sync_to_thor.sh            # full sync
#   bash run/sync_to_thor.sh --dry-run  # preview only
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
exec bash "$script_dir/sync_to_target.sh" thor "$@"
