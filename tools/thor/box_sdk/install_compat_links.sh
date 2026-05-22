#!/usr/bin/env bash
# Create the compat symlinks the vendored BOX SDK runtime needs on JetPack 6+.
#
# `libbox_controller.so` was built against `libtinyxml2.so.9` /
# `liburdfdom_model.so.3.0`, but Ubuntu 22.04 / 24.04 ship `.so.10` and
# `.so.4.0`. The ABIs we touch (URDF parsing) are compatible, so a name-only
# symlink lets the loader find them.
#
# Idempotent. Run once after `sudo apt install -y liburdfdom-dev` (which is
# what supplies the underlying system libraries).
#
# Usage:
#   bash tools/thor/box_sdk/install_compat_links.sh

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$THIS_DIR/lib"

mkdir -p "$LIB_DIR"

# Detect the multiarch dir (aarch64 on Jetson, x86_64 on workstation).
case "$(uname -m)" in
    aarch64|arm64) MULTIARCH="aarch64-linux-gnu" ;;
    x86_64|amd64)  MULTIARCH="x86_64-linux-gnu" ;;
    *) echo "Unsupported arch: $(uname -m)"; exit 1 ;;
esac
SYS_LIB="/usr/lib/$MULTIARCH"

link_compat() {
    local link_name="$1" candidates_glob="$2"
    if [ -L "$LIB_DIR/$link_name" ] || [ -e "$LIB_DIR/$link_name" ]; then
        echo "already present: $LIB_DIR/$link_name"
        return
    fi
    local target
    target=$(ls -1 $candidates_glob 2>/dev/null | sort -V | tail -n1 || true)
    if [ -z "$target" ]; then
        echo "WARN: no system candidate for $link_name (glob=$candidates_glob)"
        echo "      run: sudo apt install -y liburdfdom-dev"
        return
    fi
    ln -s "$target" "$LIB_DIR/$link_name"
    echo "linked: $LIB_DIR/$link_name -> $target"
}

link_compat libtinyxml2.so.9 "$SYS_LIB/libtinyxml2.so.1*"
link_compat liburdfdom_model.so.3.0 "$SYS_LIB/liburdfdom_model.so.4*"

echo "done. \`source tools/thor/box_sdk/setup_env.sh\` to enable LD_LIBRARY_PATH."
