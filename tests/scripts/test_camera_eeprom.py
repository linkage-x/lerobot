"""EEPROM identity: parse the real dumps read off Thor on 2026-09-28."""

from __future__ import annotations

import json

import pytest

from tools.thor.gmsl2 import camera_eeprom as ce

# First 336 bytes of four modules' EEPROMs (sudo i2ctransfer, 2026-09-28).
DUMPS = {
    4: (
        "1a00800738041e000202ffffffffffffffffffffffffffffffffffff32b763dc7800004040cdcc0c404b002effffffffffffffffffffffffffffffffb8a2bbfcddb5043cffffffffffffffffffffffff"
        "ffffffffffffffffffffffff2aa9023f8007380401f910bd95f6648f400b2c43c622658f40d18a9e8554eb8c4028c412c55e96804077a897b0de633a404095b6db9e432f40854126cc85a0fbbeea59ea"
        "0259adfebeac19c89c2feae93f0d531b1859c83a408e38e83bcd463a40b35a53aea6d31140ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff0951618a483132304b2d493035313330303537ffffffffffffffffffffffffffffffffffff76697232ffffffffffffffffffffff"
    ),
    6: (
        "1a00800738041e000202ffffffffffffffffffffffffffffffffffff32b763dc7800004040cdcc0c404b002effffffffffffffffffffffffffffffffb8a2bbfcb9fc073cffffffffffffffffffffffff"
        "ffffffffffffffffffffffff9832ef9e8007380401d063ecb1cd368f40696d5cf79f368f40bd25e9f4d3258e4039961bf7476b8140decacaa2b49fae3f69c29990742cd8bfef9fbde69401293fa9deb4"
        "3e008017bf9c6be2e5eccb9cbf4780158f2befdd3f4b47a5c6375cddbf033e39ae314dc1bfffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff22707c5b483132304b2d493035313330303234ffffffffffffffffffffffffffffffffffff0bae3ae4ffffffffffffffffffffff"
    ),
    8: (
        "1a00800738041e000202ffffffffffffffffffffffffffffffffffff32b763dc7800004040cdcc0c404b002effffffffffffffffffffffffffffffffb8a2bbfc6c09f93bffffffffffffffffffffffff"
        "ffffffffffffffffffffffff2c1da9f3800738040120dfe28097488f401a2fccaf4c488f40b04fe60225b58e40fa86477c2d6b8240ae72a53d25d80440577edc28f9ebf23ff5fb387915422c3ff29bd6"
        "2eecb7d63e70d54192dc50ad3f3d4d7e34291e0840f998ee42aa1d0140a7b28280f091d43fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffbf0af510483132304b2d493035313330303636ffffffffffffffffffffffffffffffffffff230a586effffffffffffffffffffff"
    ),
    12: (
        "1a00800738041e000202ffffffffffffffffffffffffffffffffffff32b763dc7800004040cdcc0c404b002effffffffffffffffffffffffffffffffb8a2bbfc2497ff3bffffffffffffffffffffffff"
        "ffffffffffffffffffffffff461f3e128007380401a05426f55e848f40dcffbb963e848f4030d8c0275abe8e40fd3cb63b153e81401e698f1ffad61a40a95f5faf14fa0f40433685979d93003fde3d8d"
        "aa502c013fa01ea2635cf0cd3f3d1312585d721c40c742347fb3801a4026d086d16a3ff33fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffb723ec6d483132304b2d493035313330303637ffffffffffffffffffffffffffffffffffffb53a5f19ffffffffffffffffffffff"
    ),
}


def _reader(bus, addr, offset, length):
    sid = next((s for s in DUMPS if ce.eeprom_location(s) == (bus, addr)), None)
    return None if sid is None else bytes.fromhex(DUMPS[sid])[offset:offset + length]


def test_location_follows_the_device_tree():
    assert ce.eeprom_location(0) == (17, 0x60)
    assert ce.eeprom_location(6) == (18, 0x62)
    assert ce.eeprom_location(15) == (20, 0x63)


def test_serial_and_factory_intrinsics_parse():
    mods = {m.sid: m for m in ce.read_modules([4, 6, 7], reader=_reader)}
    assert mods[4].serial == "H120K-I05130057"
    assert mods[6].serial == "H120K-I05130024"
    assert (mods[6].width, mods[6].height, mods[6].model) == (1920, 1080, 1)
    assert mods[6].fx == pytest.approx(998.850437, abs=1e-5)
    assert mods[6].cx == pytest.approx(964.728494, abs=1e-5)
    assert set(mods[6].dist) == {"k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"}
    assert mods[6].problem is None
    assert not mods[7].answered and "did not answer" in mods[7].problem


def test_fingerprint_finds_the_camera_that_moved_ports():
    """0804 intrinsics per port: the module on port 4 today was on port 13 then."""
    calibrated = {
        "cam_06": (992.94, 966.27, 561.38), "cam_07": (999.72, 973.25, 529.03),
        "cam_08": (998.07, 983.13, 591.80), "cam_09": (1006.29, 929.89, 519.65),
        "cam_12": (1007.62, 983.78, 551.79), "cam_13": (1004.70, 927.49, 531.56),
        "cam_14": (1003.87, 962.80, 535.91),
    }
    mods = {m.sid: m for m in ce.read_modules([4, 6, 8, 12], reader=_reader)}
    for sid in (6, 8, 12):
        hit = ce.match_calibrated(mods[sid], calibrated)
        assert hit["camera"] == f"cam_{sid:02d}" and hit["same_port"] and hit["distance_px"] < 6.0
    moved = ce.match_calibrated(mods[4], calibrated)
    assert moved["camera"] == "cam_13" and not moved["same_port"]
    assert moved["runner_up_px"] > 2 * moved["distance_px"]


def test_cli_writes_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ce, "i2ctransfer_reader", _reader)
    monkeypatch.setattr(ce.read_modules, "__defaults__", (_reader,))
    out = tmp_path / "ids.json"
    assert ce.main(["--sids", "4,6,7", "--json", str(out)]) == 0
    rows = {r["sid"]: r for r in json.loads(out.read_text())["modules"]}
    assert rows[4]["serial"] == "H120K-I05130057" and rows[7]["answered"] is False
    assert "no answer: cam_07" in capsys.readouterr().out


def test_write_expected_ties_each_calibrated_camera_to_its_serial(tmp_path, monkeypatch):
    """The table is per calibrated camera, found wherever it is plugged in now."""
    monkeypatch.setattr(ce.read_modules, "__defaults__", (_reader,))
    conv = tmp_path / "calib" / "converted"
    for cam, (fx, cx, cy) in {"cam_06": (992.94, 966.27, 561.38), "cam_12": (1007.62, 983.78, 551.79),
                              "cam_13": (1004.70, 927.49, 531.56), "cam_09": (1001.0, 1002.0, 556.5)}.items():
        (conv / f"{cam}_X").mkdir(parents=True)
        (conv / f"{cam}_X" / "intrinsics_producer.json").write_text(
            json.dumps({"camera_matrix": [[fx, 0, cx], [0, fx, cy], [0, 0, 1]]}))
    out = tmp_path / "expected.json"
    assert ce.main(["--sids", "4,6,12", "--intrinsics", str(tmp_path / "calib"),
                    "--write-expected", str(out)]) == 0
    data = json.loads(out.read_text())
    # cam_13's camera is on port 4 now: still identified, by its serial.
    assert data["ports"] == {"cam_06": "H120K-I05130024", "cam_12": "H120K-I05130067",
                             "cam_13": "H120K-I05130057"}
    assert data["not_identified"] == {"unread": ["cam_09"], "ambiguous": []}
    with pytest.raises(SystemExit):
        ce.main(["--write-expected", str(out)])
