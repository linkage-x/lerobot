"""Stable GUI recorder entrypoint for registered Robot / Teleoperator adapters.

Reuses the gateway command channel and dataset lifecycle of the existing recorder.
FR3 uses its Cartesian processors; other robots use their registered/native schema.
"""

from tools.fr3.fr3_gui_record_runtime import main

if __name__ == "__main__":
    main()
