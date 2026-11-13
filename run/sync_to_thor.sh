#!/usr/bin/env bash
# Compatibility wrapper for the target-aware incremental sync script. There is more than one Thor,
# so it no longer picks one: name it.
# Usage:
#   bash run/sync_to_thor.sh thor1            # full sync
#   bash run/sync_to_thor.sh thor2 --dry-run  # preview only
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
exec bash "$script_dir/sync_to_target.sh" "$@"
