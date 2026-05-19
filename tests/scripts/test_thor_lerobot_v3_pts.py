from pathlib import Path

from tools.thor.gmsl2 import thor_lerobot_v3 as lr3


def test_parse_ffprobe_pts_ignores_non_float_lines():
    assert lr3._parse_ffprobe_pts("0.100000\nN/A\n0.116667\n") == [0.1, 0.116667]


def test_extract_pts_uses_ffprobe_result(monkeypatch, tmp_path):
    calls = []

    def fake_ffprobe(path: Path, *, timeout_s: float):
        calls.append((path, timeout_s))
        return [1.0, 1.5]

    def fake_gstreamer(path: Path, *, timeout_s: float):
        raise AssertionError(
            "GStreamer fallback should not run when ffprobe returns a result"
        )

    monkeypatch.setattr(lr3, "_extract_pts_ffprobe", fake_ffprobe)
    monkeypatch.setattr(lr3, "_extract_pts_gstreamer", fake_gstreamer)

    mkv = tmp_path / "cam_00.mkv"
    assert lr3.extract_pts(mkv, timeout_s=2.0) == [1.0, 1.5]
    assert calls == [(mkv, 2.0)]


def test_extract_pts_falls_back_to_gstreamer_when_ffprobe_missing(monkeypatch, tmp_path):
    calls = []

    def fake_ffprobe(path: Path, *, timeout_s: float):
        calls.append(("ffprobe", path, timeout_s))
        return None

    def fake_gstreamer(path: Path, *, timeout_s: float):
        calls.append(("gstreamer", path, timeout_s))
        return [1.601, 1.655]

    monkeypatch.setattr(lr3, "_extract_pts_ffprobe", fake_ffprobe)
    monkeypatch.setattr(lr3, "_extract_pts_gstreamer", fake_gstreamer)

    mkv = tmp_path / "cam_00.mkv"
    assert lr3.extract_pts(mkv, timeout_s=3.0) == [1.601, 1.655]
    assert calls == [("ffprobe", mkv, 3.0), ("gstreamer", mkv, 3.0)]
