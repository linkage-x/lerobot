# ruff: noqa: N806  -- matrices keep their math names
"""box_six_d_store: the station's own 6D payload records, the self-service gate, retiring a record."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.scripts.test_box_six_d import K_TRUE, START_S, _episode, _meta
from tools.data_collection_gui import fr3_six_d_zero as z
from tools.thor.gmsl2 import (
    box_six_d as bsd,
    box_six_d_capture as cap,
    box_six_d_store as store,
    thor_lerobot_v3 as lr3,
)

K_OLD = (K_TRUE * 1.1).round(5).tolist()
K_NEW = K_TRUE.round(5).tolist()


def _repo(**entry_over):
    entry = {"role": "handheld", "hand": "right", "six_d_payload": [
        {"valid_from_unix": 100, "valid_from": "x", "model": store.MODEL, "K": K_OLD, "status": "provisional"},
    ]}
    entry.update(entry_over)
    return {"schema": store.SCHEMA, "devices": {"7": entry, "8": {"role": "fr3_gripper", "six_d_payload": []}}}


def _good_fit(**over):
    fit = {
        "K": K_NEW, "n_static": 400, "attitudes": dict.fromkeys(store.FACES, 40), "missing_attitudes": [],
        "heldout_worst_force_N": 0.12, "heldout_worst_moment_Nm": 0.01, "meets_target": True,
        "rigid_reading": {"mass_kg": 0.9}, "box_uptime_s": {"start_s": 3600.0, "end_s": 3900.0},
        "drift_same_attitude": {"pairs": 1, "worst_force_N": 0.08},
    }
    fit.update(over)
    return fit


def _register(fit, valid_from, now):
    record = store.payload_record(fit, end_effector="gripper + rig v9b + camera", valid_from_unix=valid_from,
                                  status="self-service", source="capture")
    return store.register_event("7", record, now_unix=now)


def test_the_gate_numbers_are_box_six_d_s():
    assert store.TARGET_HELDOUT_N == bsd.TARGET_HELDOUT_RMS_N
    assert store.DRIFT_LIMIT_N == bsd.DRIFT_WARN_N
    assert store.WARMUP_S == bsd.WARMUP_S
    assert store.REGISTRY_PATH == bsd.REGISTRY_PATH


def test_a_good_warm_hand_held_fit_passes_the_gate():
    assert store.self_service_refusals(
        _good_fit(), device_id=7, registry=_repo(), valid_from_unix=1.79e9, capture_started_unix=1.791e9, now_unix=1.8e9
    ) == []


@pytest.mark.parametrize(
    ("over", "words"),
    [
        ({"heldout_worst_force_N": 0.45, "meets_target": False}, "留一面"),
        ({"missing_attitudes": ["z-"]}, "缺"),
        ({"box_uptime_s": None}, "上电时长"),
        ({"box_uptime_s": {"start_s": 600.0}}, "预热"),
        # a capture that held every face once measures no drift: that is not a pass
        ({"drift_same_attitude": {"pairs": 0, "worst_force_N": None}}, "漂移估计不了"),
        ({"drift_same_attitude": None}, "漂移估计不了"),
        ({"drift_same_attitude": {"pairs": 2, "worst_force_N": 0.6}}, "漂了"),
        ({"error": "attitudes {...}: need at least 4", "K": None}, "拟合被拒"),
    ],
)
def test_the_gate_names_each_thing_it_refuses(over, words):
    found = store.self_service_refusals(
        _good_fit(**over), device_id=7, registry=_repo(), valid_from_unix=1.79e9, capture_started_unix=1.791e9,
        now_unix=1.8e9,
    )
    assert any(words in r for r in found), found


def test_the_gate_keeps_arm_boxes_unknown_boxes_and_history_for_a_maintainer():
    kw = {"capture_started_unix": 1.791e9, "now_unix": 1.8e9}
    (arm,) = store.self_service_refusals(_good_fit(), device_id=8, registry=_repo(), valid_from_unix=1.79e9, **kw)
    assert "维护者" in arm
    (unknown,) = store.self_service_refusals(_good_fit(), device_id=9, registry=_repo(), valid_from_unix=1.79e9, **kw)
    assert "box_devices.json" in unknown
    newer = _repo()
    newer["devices"]["7"]["six_d_payload"][0]["valid_from_unix"] = 1.7905e9
    assert any("最新的登记" in r for r in store.self_service_refusals(_good_fit(), device_id=7, registry=newer, valid_from_unix=1.79e9, **kw))
    # the same mount time is a re-fit of that configuration: allowed
    assert store.self_service_refusals(_good_fit(), device_id=7, registry=newer, valid_from_unix=1.7905e9, **kw) == []
    late = store.self_service_refusals(_good_fit(), device_id=7, registry=_repo(), valid_from_unix=1.792e9, **kw)
    assert any("晚于这次采集" in r for r in late)


def test_station_records_are_merged_over_the_repo_and_the_newest_re_fit_of_a_mount_is_in_force():
    first = _register(_good_fit(), 1.79e9, now=1.7901e9)
    refit = _register(_good_fit(K=K_OLD), 1.79e9, now=1.7902e9)
    reg = store.merge(_repo(), [first, refit])
    assert reg["local_store"]["events"] == 2
    records = reg["devices"]["7"]["six_d_payload"]
    assert [r["store"] for r in records] == ["repo", "local", "local"]
    assert store.in_force(reg, 7, 1.0e9)[0]["status"] == "provisional"  # before the mount: the repo record
    assert store.in_force(reg, 7, 1.80e9)[0]["registered_unix"] == 1790200000
    model = bsd.payload_model(reg, 7, 1.80e9)
    assert model.store == "local" and model.registered_unix == 1790200000
    # the arm zero picks the same K
    assert z.payload_k(reg, 7, 1.80e9)[0] == K_OLD
    with pytest.raises(ValueError, match="not in tools/thor/box_devices.json"):
        store.merge(_repo(), [{**first, "device_id": "9"}])


def test_retiring_falls_back_to_an_earlier_re_fit_of_the_mount_never_to_an_older_mount():
    first = _register(_good_fit(), 1.79e9, now=1.7901e9)
    refit = _register(_good_fit(K=K_OLD), 1.79e9, now=1.7902e9)

    def retire(registered):
        return {"event": "retire", "device_id": "7", "valid_from_unix": 1790000000, "registered_unix": registered,
                "reason": "rig was loose", "by": "maintainer", "at_unix": 1.7903e9}

    reg = store.merge(_repo(), [first, refit, retire(1790200000)])
    assert store.in_force(reg, 7, 1.8e9)[0]["registered_unix"] == 1790100000
    reg = store.merge(_repo(), [first, refit, retire(1790200000), retire(1790100000)])
    record, why = store.in_force(reg, 7, 1.8e9)
    assert record is None and "retired" in why and "rig was loose" in why
    model, why = bsd.payload_model_or_reason(reg, 7, 1.8e9)
    assert model is None and "rig was loose" in why
    assert z.payload_k(reg, 7, 1.8e9)[0] is None
    # a repo record is retired by naming it without registered_unix
    repo_retired = store.merge(_repo(), [{**retire(None), "valid_from_unix": 100}])
    assert store.in_force(repo_retired, 7, 1.0e9)[0] is None
    with pytest.raises(ValueError, match="names no record"):
        store.merge(_repo(), [retire(123)])


def test_an_export_of_a_retired_mount_refuses_and_says_why():
    names = list(lr3.BOX_STATE_NAMES)
    rows, _acc, _push = _episode(names)
    reg = store.merge(_repo(), [{"event": "retire", "device_id": "7", "valid_from_unix": 100, "registered_unix": None,
                                 "reason": "fitted before the camera went on", "by": "m", "at_unix": 1.0}])
    with pytest.raises(bsd.SixDExportError, match="fitted before the camera went on"):
        bsd.plan_episode(names, rows, _meta(), reg)


def test_a_station_record_labels_and_records_the_k_it_used():
    names = list(lr3.BOX_STATE_NAMES)
    rows, _acc, _push = _episode(names)
    a = store.merge(_repo(), [_register(_good_fit(), 200, now=START_S - 10)])
    b = store.merge(_repo(), [_register(_good_fit(), 200, now=START_S - 10), _register(_good_fit(K=K_OLD), 200, now=START_S - 5)])
    (pa,) = bsd.plan_episode(names, rows, _meta(), a)
    (pb,) = bsd.plan_episode(names, rows, _meta(), b)
    # two re-fits of one mount are two labels: the label chain will not mix them in one dataset
    assert pa.label() == f"box7@200r{int(START_S - 10)}" and pb.label() == f"box7@200r{int(START_S - 5)}"
    assert pa.record["payload_store"] == "local" and pa.record["payload_K"] == K_NEW
    assert pa.record["payload_registered_unix"] == int(START_S - 10)
    # a repo record keeps the label it had (exports made before this change compare equal)
    (repo_plan,) = bsd.plan_episode(names, rows, _meta(), _repo())
    assert repo_plan.label() == "box7@100" and "payload_registered_unix" not in repo_plan.record


def test_the_log_is_appended_and_a_broken_line_is_an_error_not_a_skip(tmp_path):
    path = tmp_path / "registry.jsonl"
    assert store.read_events(path) == []
    store.append_event(path, _register(_good_fit(), 1.79e9, now=1.7901e9))
    store.append_event(path, _register(_good_fit(), 1.79e9, now=1.7902e9))
    assert len(store.read_events(path)) == 2
    with path.open("a") as fh:
        fh.write('{"event": "retire", "device_id"\n')
    with pytest.raises(ValueError, match=":3: not JSON"):
        store.read_events(path)


def test_load_registry_merges_the_station_log_only_when_asked(tmp_path):
    repo = tmp_path / "box_devices.json"
    repo.write_text(json.dumps(_repo()))
    log = tmp_path / "registry.jsonl"
    store.append_event(log, _register(_good_fit(), 1.79e9, now=1.7901e9))
    assert len(bsd.load_registry(repo)["devices"]["7"]["six_d_payload"]) == 1  # an explicit file: as it is
    assert len(bsd.load_registry(repo, local=log)["devices"]["7"]["six_d_payload"]) == 2


def test_maintainer_cli_lists_and_retires(tmp_path, capsys):
    (tmp_path / "tools" / "thor").mkdir(parents=True)
    (tmp_path / "tools" / "thor" / "box_devices.json").write_text(json.dumps(_repo()))
    log = tmp_path / store.LOCAL_SUBPATH
    store.append_event(log, _register(_good_fit(), 1.79e9, now=1.7901e9))
    assert store.main(["--repo-root", str(tmp_path), "list"]) == 0
    assert "[local]" in capsys.readouterr().out
    assert store.main(["--repo-root", str(tmp_path), "retire", "--device-id", "7", "--valid-from-unix", "1790000000",
                       "--registered-unix", "1", "--reason", "x", "--by", "m"]) == 2  # names nothing: nothing written
    assert len(store.read_events(log)) == 1
    assert store.main(["--repo-root", str(tmp_path), "retire", "--device-id", "7", "--valid-from-unix", "1790000000",
                       "--registered-unix", "1790100000", "--reason", "loose rig", "--by", "m"]) == 0
    record, why = store.in_force(store.load_merged(tmp_path), 7, 1.8e9)
    assert record is None and "loose rig" in why


def test_the_capture_says_when_a_face_was_held_again_for_the_drift_check():
    c = cap.Counter(need=1)
    t = 0.0
    c.add([0, 0, 1.0], [40, 0, 0], t)  # pick-up
    for face in ([1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]):
        t += 3.0
        c.add(face, [0, 0, 0], t)
    assert c.faces_done and not c.done and c.revisited is None
    c.add([1.0, 0, 0], [0, 0, 0], t + 1.0)  # x+ again, but too soon to tell drift
    assert c.revisited is None
    c.add([1.0, 0, 0], [0, 0, 0], 3.0 + bsd.DRIFT_MIN_APART_S + 1.0)
    assert c.revisited == "x+" and c.done


def test_the_k_from_a_station_record_matches_the_numpy_model():
    reg = store.merge(_repo(), [_register(_good_fit(), 1.79e9, now=1.7901e9)])
    np.testing.assert_allclose(bsd.payload_model(reg, 7, 1.8e9).K, np.asarray(K_NEW))
