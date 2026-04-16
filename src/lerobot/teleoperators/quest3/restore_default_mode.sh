#!/usr/bin/env bash
set -e

adb shell am broadcast -a com.oculus.vrpowermanager.automation_disable
adb shell settings put system screen_off_timeout 30000
adb shell svc power stayon false

echo "Quest sleep/proximity behavior restored."