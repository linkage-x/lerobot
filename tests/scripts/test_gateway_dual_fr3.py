"""The gateway serves an episode's two-arm verdicts (roadmap C4) read-only, as the replay page shows them."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tools.data_collection_gui import gateway


def _state(tmp_path: Path, verdicts: dict | None) -> gateway.GatewayState:
    dataset_root = tmp_path / "session"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text("{}")
    if verdicts is not None:
        (dataset_root / gateway.DUAL_FR3_EXECUTABLE_FILE).write_text(json.dumps(verdicts))
    state = gateway.GatewayState(
        repo_root=Path.cwd(),
        config_path=tmp_path / "config.yaml",
        config={},
        recording=gateway.RecordingStatus(),
        replay=gateway.ReplayStatus(episode=0),
    )
    state.selected_replay_root = dataset_root
    return state


VERDICTS = {
    "schema": "dual_fr3_executable/v1",
    "modes": {
        "absolute": {
            "created_utc": "2026-10-26T04:00:00Z",
            "embodiment": "fr3_dual",
            "summary": {"episodes": 2, "executable": 1},
            "episodes": {
                "0": {"dual_fr3_executable": True, "reasons": []},
                "1": {"dual_fr3_executable": False, "reasons": ["arms 10 mm apart"]},
            },
        }
    },
}


def test_an_episode_s_verdicts_are_served_per_mode(tmp_path: Path) -> None:
    state = _state(tmp_path, VERDICTS)
    server = gateway.DataCollectionGuiServer(("127.0.0.1", 0), state)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/api/replay/dual-fr3?episode=1", timeout=5) as response:
            payload = json.loads(response.read())
        absolute = payload["modes"]["absolute"]
        assert payload["episode"] == 1 and absolute["summary"] == {"episodes": 2, "executable": 1}
        assert absolute["episode"] == {"dual_fr3_executable": False, "reasons": ["arms 10 mm apart"]}
        with urllib.request.urlopen(f"{base}/api/replay/dual-fr3?episode=5", timeout=5) as response:
            assert json.loads(response.read())["modes"]["absolute"]["episode"] is None
    finally:
        server.shutdown()
        server.server_close()


def test_a_dataset_without_verdicts_is_a_404_that_says_what_to_run(tmp_path: Path) -> None:
    state = _state(tmp_path, None)
    server = gateway.DataCollectionGuiServer(("127.0.0.1", 0), state)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(f"{base}/api/replay/dual-fr3?episode=0", timeout=5)
        assert missing.value.code == 404
        assert "dual_fr3_executability.py" in json.loads(missing.value.read())["error"]
    finally:
        server.shutdown()
        server.server_close()
