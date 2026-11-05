"""lerobot_train with the relight augmentation installed (tools/fr3/relight_aug.py; maps dir in RELIGHT_AUG_MAPS).

fr3_train_il_policy.py runs this module instead of lerobot.scripts.lerobot_train when --relight-aug-maps is given.
DataLoader workers are forked after the patch, so they inherit it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import relight_aug  # noqa: E402

if not os.environ.get("RELIGHT_AUG_MAPS"):
    raise SystemExit("relight_train_entry: RELIGHT_AUG_MAPS is not set")
relight_aug.install()

from lerobot.scripts.lerobot_train import main  # noqa: E402

if __name__ == "__main__":
    main()
