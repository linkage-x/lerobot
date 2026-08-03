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

  it("keeps scalar dimensions standalone and groups the current xyzw attitude layout", () => {
    // Current recorder schema (ts_sync.md §9.1.1): attitude is the quaternion
    // only, stored xyzw. Dims are read from the name list position, so nothing
    // here may depend on a fixed state offset.
    const groups = groupSeriesNames([
      "box_gripper.distance_m",
      "box_trigger.travel_pct",
      "box_imu.quat_x",
      "box_imu.quat_y",
      "box_imu.quat_z",
      "box_imu.quat_w",
      "box_six_d_force.fx"
    ]);

    expect(groups.map((group) => group.key)).toEqual([
      "box_gripper.distance_m",
      "box_trigger.travel_pct",
      "box_imu.quat",
      "box_six_d_force.fx"
    ]);
    // Plotted w-first regardless of the xyzw storage order, with the dims
    // pointing back at the stored positions.
    expect(groups[2].items.map((item) => item.component)).toEqual(["w", "x", "y", "z"]);
    expect(groups[2].items.map((item) => item.dim)).toEqual([5, 2, 3, 4]);
  });

  it("still groups the legacy wxyz + rpy attitude layout for older datasets", () => {
    // Datasets recorded before the 31 → 28 dim change still carry rpy and a
    // scalar-first quaternion; replay must keep rendering them.
    const groups = groupSeriesNames([
      "box_gripper.distance_m",
      "box_imu.quat_w",
      "box_imu.quat_x",
      "box_imu.quat_y",
      "box_imu.quat_z",
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
    expect(groups[1].items.map((item) => item.dim)).toEqual([1, 2, 3, 4]);
    expect(groups[2].items.map((item) => item.component)).toEqual(["roll", "pitch", "yaw"]);
    expect(groups[2].items.map((item) => item.dim)).toEqual([5, 6, 7]);
  });

  it("groups the FR3 workstation absolute_ee state into position and quaternion rows", () => {
    // Feature names come from the FR3 processor (`ee.x` .. `prev_cmd.gripper.pos`): the
    // component is glued to a dot-separated stem, so nothing here has a `_x` suffix to split on.
    const groups = groupSeriesNames([
      "ee.x",
      "ee.y",
      "ee.z",
      "prev_cmd.ee.x",
      "prev_cmd.ee.y",
      "prev_cmd.ee.z",
      "ee.qx",
      "ee.qy",
      "ee.qz",
      "ee.qw",
      "prev_cmd.ee.qx",
      "prev_cmd.ee.qy",
      "prev_cmd.ee.qz",
      "prev_cmd.ee.qw",
      "gripper.pos",
      "prev_cmd.gripper.pos"
    ]);

    expect(groups.map((group) => group.key)).toEqual([
      "ee",
      "prev_cmd.ee",
      "ee.q",
      "prev_cmd.ee.q",
      "gripper.pos",
      "prev_cmd.gripper.pos"
    ]);
    expect(groups[0].items.map((item) => item.component)).toEqual(["x", "y", "z"]);
    expect(groups[0].items.map((item) => item.dim)).toEqual([0, 1, 2]);
    expect(groups[1].items.map((item) => item.dim)).toEqual([3, 4, 5]);
    // Plotted w-first, same as every other quaternion, with dims pointing back at xyzw storage.
    expect(groups[2].items.map((item) => item.component)).toEqual(["w", "x", "y", "z"]);
    expect(groups[2].items.map((item) => item.dim)).toEqual([9, 6, 7, 8]);
    expect(groups[3].items.map((item) => item.dim)).toEqual([13, 10, 11, 12]);
  });

  it("groups the FR3 delta action translation and rotvec triplets", () => {
    const groups = groupSeriesNames([
      "delta_ee_from_prev_cmd.dx",
      "delta_ee_from_prev_cmd.dy",
      "delta_ee_from_prev_cmd.dz",
      "delta_ee_from_prev_cmd.drx",
      "delta_ee_from_prev_cmd.dry",
      "delta_ee_from_prev_cmd.drz",
      "gripper.pos"
    ]);

    expect(groups.map((group) => group.key)).toEqual([
      "delta_ee_from_prev_cmd.d",
      "delta_ee_from_prev_cmd.dr",
      "gripper.pos"
    ]);
    expect(groups[0].items.map((item) => item.component)).toEqual(["x", "y", "z"]);
    expect(groups[1].items.map((item) => item.dim)).toEqual([3, 4, 5]);
  });

  it("keeps a lone word-final x/y/z/w dimension standalone", () => {
    // `max` parses as stem `ma` + component `x`, but a group needs two distinct components,
    // so an unrelated neighbour must not get swallowed into it.
    const groups = groupSeriesNames(["stats.max", "stats.mean", "ee.x", "ee.y"]);

    expect(groups.map((group) => group.key)).toEqual(["stats.max", "stats.mean", "ee"]);
    expect(groups[2].items.map((item) => item.dim)).toEqual([2, 3]);
  });

  it("groups cube pose position and quaternion display names", () => {
    const groups = groupSeriesNames([
      "cube_a.position_x",
      "cube_a.position_y",
      "cube_a.position_z",
      "cube_a.quat_x",
      "cube_a.quat_y",
      "cube_a.quat_z",
      "cube_a.quat_w"
    ]);

    expect(groups.map((group) => group.key)).toEqual(["cube_a.position", "cube_a.quat"]);
    expect(groups[0].items.map((item) => item.component)).toEqual(["x", "y", "z"]);
    expect(groups[0].items.map((item) => item.dim)).toEqual([0, 1, 2]);
    expect(groups[1].items.map((item) => item.component)).toEqual(["w", "x", "y", "z"]);
    expect(groups[1].items.map((item) => item.dim)).toEqual([6, 3, 4, 5]);
  });
});
