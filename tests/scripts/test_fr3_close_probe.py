import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from tools.fr3.close_probe import chunk_behaviour, summarize  # noqa: E402


def _chunk(dz_first, close_at=None, dxy=(0.0, 0.0), horizon=50):
    chunk = np.zeros((horizon, 5))
    chunk[:10, 2] = dz_first / 10
    chunk[:, 0], chunk[:, 1] = dxy[0] / horizon, dxy[1] / horizon
    chunk[:, 4] = 1.0
    if close_at is not None:
        chunk[close_at:, 4] = 0.0
    return chunk


def test_a_chunk_that_rises_first_lifts_and_one_that_shuts_closes():
    assert chunk_behaviour(_chunk(0.002), 10) == {"lifts": True, "closes": False, "dx": 0.0, "dy": 0.0}
    down = chunk_behaviour(_chunk(-0.004, close_at=30, dxy=(0.003, -0.001)), 10)
    assert (down["lifts"], down["closes"]) == (False, True)
    assert down["dx"] == pytest.approx(0.003) and down["dy"] == pytest.approx(-0.001)
    # Under a millimetre of rise is the descent's own noise, not a lift.
    assert chunk_behaviour(_chunk(0.0008), 10)["lifts"] is False


def test_the_summary_rates_each_class_against_the_label():
    p = pd.DataFrame(
        {
            "ckpt": ["a"] * 4,
            "cls": ["lift", "lift", "descend", "descend"],
            "lifts": [True, False, False, False],
            "closes": [False, True, True, False],
            "lab_closes": [False, False, True, True],
            "mx": [0.01, 0.0, 0.002, 0.0],
            "my": [0.0, 0.0, 0.0, 0.0],
            "lx": [0.01, 0.01, 0.004, 0.0],
            "ly": [0.0, 0.0, 0.0, 0.0],
            "dz": [10.0, 20.0, 40.0, 50.0],
        }
    )
    s = summarize(p)["a"]
    assert s["lift"]["lifts"] == 0.5 and s["lift"]["closes"] == 0.5
    assert s["descend"]["closes"] == 0.5 and s["descend"]["label_closes"] == 1.0
    # A label with no xy travel has no direction to be along.
    assert s["lift"]["along"] == 0.5 and s["descend"]["along"] == 0.5
