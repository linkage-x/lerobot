import { describe, expect, it } from "vitest";

import type { TeleopGains, TeleopGainValues } from "../types";
import { GAIN_ROWS, effectiveAxisGain } from "./TeleoperationPage";

/**
 * `SpaceMouseTeleopConfig` resolves a per-axis gain in two different ways depending on whether it
 * is set: an unset axis is the global gain *times* the device's per-axis calibration, and a set one
 * replaces the calibrated value outright. The panel quotes a m/s or rad/s figure next to a control
 * the operator tunes the arm with, so getting that fallback wrong is not a cosmetic bug -- on z the
 * two readings differ by 41%.
 */
const CALIBRATION: TeleopGains["axisCalibration"] = {
  scale_x: 1,
  scale_y: 0.9414634146341463,
  scale_z: 0.5902439024390244,
  scale_wx: 1,
  scale_wy: 0.9490740740740741,
  scale_wz: 0.9259259259259259
};

const row = (field: string) => GAIN_ROWS.find((entry) => entry.field === field)!;

describe("effectiveAxisGain", () => {
  const values: TeleopGainValues = {
    translation_scale: 0.000615,
    rotation_scale: 0.000648,
    scale_x: null,
    scale_y: null,
    scale_z: null,
    scale_wx: null,
    scale_wy: null,
    scale_wz: null
  };

  it("scales a blank axis by its calibration rather than quoting the global", () => {
    expect(effectiveAxisGain(values, row("scale_x"), CALIBRATION)).toBeCloseTo(0.000615, 12);
    expect(effectiveAxisGain(values, row("scale_z"), CALIBRATION)).toBeCloseTo(0.000615 * 0.5902439024390244, 12);
    expect(effectiveAxisGain(values, row("scale_wy"), CALIBRATION)).toBeCloseTo(0.000648 * 0.9490740740740741, 12);
  });

  it("takes a filled axis at face value, calibration and all", () => {
    // The reason the panel says so: typing the global's own number into z speeds that axis up 1.7x.
    const filled = { ...values, scale_z: 0.000615 };
    expect(effectiveAxisGain(filled, row("scale_z"), CALIBRATION)).toBe(0.000615);
  });

  it("keeps a disabling zero at zero instead of falling back to the global", () => {
    const off = { ...values, scale_wx: 0 };
    expect(effectiveAxisGain(off, row("scale_wx"), CALIBRATION)).toBe(0);
  });

  it("reports nothing when the global it would fall back to is unset", () => {
    const noGlobal = { ...values, rotation_scale: null };
    expect(effectiveAxisGain(noGlobal, row("scale_wz"), CALIBRATION)).toBeNull();
  });
});
