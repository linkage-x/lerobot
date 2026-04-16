#!/usr/bin/env bash
set -e

adb devices

echo "[1/3] Wake Quest..."
adb shell input keyevent KEYCODE_WAKEUP || true

echo "[2/3] Disable proximity sleep..."
adb shell am broadcast -a com.oculus.vrpowermanager.prox_close

echo "[3/3] Extend screen timeout and stay awake..."
adb shell settings put system screen_off_timeout 999999999
adb shell svc power stayon true

echo "Done. Put Quest 3 on a stand facing the controllers."