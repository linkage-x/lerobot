#!/usr/bin/env bash
# Manage personal sandboxes on a Thor (deploy into one with: bash run/deploy.sh thor1 --sandbox NAME).
# The first argument is always the Thor (run/targets/; `thor` = thor1); there is no default.
#
#   bash run/sandbox.sh TARGET list                          # every sandbox: port, version, running or not
#   bash run/sandbox.sh TARGET status NAME                   # one sandbox in detail
#   bash run/sandbox.sh TARGET datasets [FILTER]             # production datasets you can copy, with sizes
#   bash run/sandbox.sh TARGET copy-dataset NAME DATASET...  # copy production datasets into the sandbox
#   bash run/sandbox.sh TARGET logs NAME                     # tail the sandbox gateway log
#   bash run/sandbox.sh TARGET stop NAME                     # stop the sandbox gateway
#   bash run/sandbox.sh TARGET remove NAME --yes             # stop it and delete the whole sandbox directory
#
# A sandbox only ever reads production: datasets are copied, never linked, because trajectory
# generation, BOX re-pairing and episode deletion rewrite files in place.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=run/sandbox_common.sh
. "$script_dir/sandbox_common.sh"

usage() {
  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
}

target=""
case "${1:-}" in
  -h|--help|"") ;;
  *) target="$1"; shift ;;
esac

remote() {
  target_ssh "$@"
}

# Remote helper: the pids of the gateway started for one tree.
remote_gateway_pids='
gateway_pids_for() {
  python3 - "$1" <<"PY"
import os, sys
repo = sys.argv[1]
for name in os.listdir("/proc"):
    if not name.isdigit():
        continue
    try:
        args = [a.decode("utf-8", "ignore") for a in open(f"/proc/{name}/cmdline", "rb").read().split(b"\0") if a]
    except OSError:
        continue
    if "tools.data_collection_gui.gateway" in args and any(
        a == "--repo-root" and b == repo for a, b in zip(args, args[1:])
    ):
        print(name)
PY
}
'

cmd="${1:-}"
shift || true

case "$target" in
  "") ;;
  list|status|logs|stop|datasets|copy-dataset|remove)
    # The pre-11-11 form (no machine): say so instead of resolving "list" as a target.
    require_target_name ""
    ;;
  *)
    if [[ -z "$cmd" ]]; then usage >&2; exit 2; fi
    resolve_target "$target"
    if [[ "$TARGET_SANDBOXES" != true ]]; then
      echo "ERROR: ${TARGET_NAME} has no sandboxes (run/targets/${TARGET_NAME}.json)" >&2
      exit 2
    fi
    require_target_identity
    ;;
esac

case "$cmd" in
  list)
    remote "$remote_gateway_pids"'
      shopt -s nullglob
      dirs=('"$THOR_SANDBOX_BASE"'/*/)
      if [[ ${#dirs[@]} -eq 0 ]]; then echo "no sandboxes yet"; exit 0; fi
      printf "%-16s %-6s %-9s %-10s %s\n" NAME PORT GATEWAY COMMIT "BRANCH / DEPLOYED"
      for d in "${dirs[@]}"; do
        d="${d%/}"; name="${d##*/}"
        port="$(cat "$d/.sandbox_port" 2>/dev/null || echo -)"
        if [[ -n "$(gateway_pids_for "$d")" ]]; then running=running; else running=stopped; fi
        info="$(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print((r.get(\"commit\") or \"-\")[:8], (r.get(\"branch\") or \"-\") + (\" (dirty)\" if r.get(\"dirty\") else \"\"), r.get(\"deployedUtc\") or \"-\")" "$d/DEPLOYED.json" 2>/dev/null || echo "- - -")"
        read -r commit branch when <<<"$info"
        printf "%-16s %-6s %-9s %-10s %s %s\n" "$name" "$port" "$running" "$commit" "$branch" "$when"
      done
      echo
      df -h --output=avail,pcent / | tail -1 | awk "{print \"'"$TARGET_NAME"' disk: \" \$1 \" free (\" \$2 \" used)\"}"
    '
    ;;

  status|logs|stop)
    name="${1:-}"
    [[ -n "$name" ]] || { usage >&2; exit 2; }
    validate_sandbox_name "$name"
    dir="$(sandbox_dir "$name")"
    case "$cmd" in
      status)
        remote "$remote_gateway_pids"'
          d='"'$dir'"'
          [[ -d "$d" ]] || { echo "no sandbox '"$name"' (deploy one: bash run/deploy.sh '"$TARGET_NAME"' --sandbox '"$name"')"; exit 1; }
          port="$(cat "$d/.sandbox_port" 2>/dev/null || echo "?")"
          pids="$(gateway_pids_for "$d")"
          echo "sandbox:  '"$name"' ($d)"
          echo "gateway:  ${pids:+running (pid $pids)}${pids:-stopped}, http://'"$TARGET_ADDRESS"':${port}"
          echo "deployed:"; sed "s/^/  /" "$d/DEPLOYED.json" 2>/dev/null || echo "  (no DEPLOYED.json)"
          echo "datasets:"; (cd "$d/outputs/datasets" 2>/dev/null && du -sh -- * 2>/dev/null | sed "s/^/  /") || true
          echo "size:     $(du -sh "$d" 2>/dev/null | cut -f1)"
        '
        ;;
      logs)
        remote 'f="$(ls -t '"'$dir'"'/outputs/logs/data_collection_gui/gateway_*.log 2>/dev/null | head -1)"; if [[ -n "$f" ]]; then echo "--- $f"; tail -n 80 "$f"; else echo "no gateway log yet"; fi'
        ;;
      stop)
        remote "$remote_gateway_pids"'
          pids="$(gateway_pids_for '"'$dir'"')"
          if [[ -z "$pids" ]]; then echo "sandbox '"$name"': gateway not running"; exit 0; fi
          echo "$pids" | xargs -r kill; sleep 1
          pids="$(gateway_pids_for '"'$dir'"')"; [[ -z "$pids" ]] || echo "$pids" | xargs -r kill -9
          echo "sandbox '"$name"': gateway stopped"
        '
        ;;
    esac
    ;;

  datasets)
    filter="${1:-}"
    remote 'cd '"'$TARGET_DIR'"'/outputs/datasets && for d in */; do d="${d%/}"; [[ -z "'"$filter"'" || "$d" == *"'"$filter"'"* ]] || continue; printf "%8s  %s\n" "$(du -sh -- "$d" 2>/dev/null | cut -f1)" "$d"; done'
    ;;

  copy-dataset)
    name="${1:-}"
    shift || true
    [[ -n "$name" && $# -gt 0 ]] || { usage >&2; exit 2; }
    validate_sandbox_name "$name"
    dir="$(sandbox_dir "$name")"
    for dataset in "$@"; do
      if [[ ! "$dataset" =~ ^[A-Za-z0-9._-]+$ || "$dataset" == .* ]]; then
        echo "ERROR: '$dataset' is not a dataset name (see: bash run/sandbox.sh datasets)" >&2
        exit 2
      fi
    done
    remote '
      set -euo pipefail
      src_root='"'$TARGET_DIR'"'/outputs/datasets
      dst_root='"'$dir'"'/outputs/datasets
      [[ -d '"'$dir'"' ]] || { echo "ERROR: no sandbox '"$name"' yet; deploy it first: bash run/deploy.sh '"$TARGET_NAME"' --sandbox '"$name"'" >&2; exit 1; }
      mkdir -p "$dst_root"
      for dataset in '"$*"'; do
        src="$src_root/$dataset"; dst="$dst_root/$dataset"
        [[ -d "$src" ]] || { echo "ERROR: production has no dataset $dataset (see: bash run/sandbox.sh datasets)" >&2; exit 1; }
        if [[ -e "$dst" ]]; then echo "skip $dataset: already in the sandbox (delete $dst first to copy it again)"; continue; fi
        need_kb="$(du -sk "$src" | cut -f1)"
        free_kb="$(df -k --output=avail / | tail -1 | tr -d " ")"
        # Keep 30 GB free for recordings: a full disk stops the production recorder.
        if (( free_kb - need_kb < 30 * 1024 * 1024 )); then
          echo "ERROR: copying $dataset ($((need_kb / 1024)) MB) would leave less than 30 GB free on '"$TARGET_NAME"'; remove datasets you no longer need first" >&2
          exit 1
        fi
        echo "copying $dataset ($((need_kb / 1024)) MB)..."
        cp -a --reflink=auto "$src" "$dst.partial"
        mv "$dst.partial" "$dst"
      done
      echo "done; the sandbox GUI picks the new datasets up on its next scan"
    '
    ;;

  remove)
    name="${1:-}"
    [[ -n "$name" ]] || { usage >&2; exit 2; }
    validate_sandbox_name "$name"
    if [[ "${2:-}" != "--yes" ]]; then
      echo "This stops sandbox '$name' and deletes $(sandbox_dir "$name") on ${TARGET_NAME}, including every dataset copied or recorded into it. Production is not touched." >&2
      echo "Run again with --yes to do it: bash run/sandbox.sh ${TARGET_NAME} remove $name --yes" >&2
      exit 2
    fi
    bash "$0" "$TARGET_NAME" stop "$name"
    remote 'd='"'$(sandbox_dir "$name")'"'; case "$d" in '"'$THOR_SANDBOX_BASE'"'/?*) rm -rf -- "$d"; echo "removed $d" ;; *) echo "refusing to remove $d" >&2; exit 1 ;; esac'
    ;;

  -h|--help|"")
    usage
    ;;
  *)
    echo "ERROR: unknown command '$cmd'" >&2
    usage >&2
    exit 2
    ;;
esac
