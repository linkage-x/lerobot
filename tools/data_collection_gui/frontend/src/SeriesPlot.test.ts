import { describe, expect, it } from "vitest";

import { groupSeriesNames } from "./SeriesPlot";

describe("groupSeriesNames", () => {
  it("groups BOX vector components into one plotted row", () => {
    const groups = groupSeriesNames([
      "box1672693301.box_imu.acc_x_g",
      "box1672693301.box_imu.acc_y_g",
      "box1672693301.box_imu.acc_z_g",
      "box1672693301.box_six_d_force.fx",
      "box1672693301.box_six_d_force.fy",
      "box1672693301.box_six_d_force.fz",
      "box1672693301.box_touch_left.mean_fx_0p1N",
      "box1672693301.box_touch_left.mean_fy_0p1N",
      "box1672693301.box_touch_left.mean_fz_0p1N"
    ]);

    expect(groups).toHaveLength(3);
    expect(groups[0]).toMatchObject({ key: "box1672693301.box_imu.acc_g", name: "box1672693301.box_imu.acc_g" });
    expect(groups[0].items.map((item) => item.component)).toEqual(["x", "y", "z"]);
    expect(groups[0].items.map((item) => item.dim)).toEqual([0, 1, 2]);
    expect(groups[1]).toMatchObject({ key: "box1672693301.box_six_d_force.f" });
    expect(groups[1].items.map((item) => item.component)).toEqual(["x", "y", "z"]);
    expect(groups[2]).toMatchObject({ key: "box1672693301.box_touch_left.mean_f_0p1N" });
    expect(groups[2].items.map((item) => item.component)).toEqual(["x", "y", "z"]);
  });

  it("keeps scalar dimensions standalone and groups orientation components", () => {
    const groups = groupSeriesNames([
      "box_gripper.distance_m",
      "box_imu.quat_x",
      "box_imu.quat_w",
      "box_imu.quat_z",
      "box_imu.quat_y",
      "box_imu.roll_deg",
      "box_imu.pitch_deg",
      "box_imu.yaw_deg",
      "box_trigger.travel_pct"
    ]);

    expect(groups.map((group) => group.key)).toEqual([
      "box_gripper.distance_m",
      "box_imu.quat",
      "box_imu.rpy_deg",
      "box_trigger.travel_pct"
    ]);
    expect(groups[1].items.map((item) => item.component)).toEqual(["w", "x", "y", "z"]);
    expect(groups[1].items.map((item) => item.dim)).toEqual([2, 1, 4, 3]);
    expect(groups[2].items.map((item) => item.component)).toEqual(["roll", "pitch", "yaw"]);
    expect(groups[2].items.map((item) => item.dim)).toEqual([5, 6, 7]);
  });
});
