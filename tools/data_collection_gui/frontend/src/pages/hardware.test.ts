import { describe, expect, it } from "vitest";
import type { HardwareDocument, HardwareField, HardwareModel } from "../hardwareTypes";
import { changeRobot, changeTool, instanceFor, isHardwareDocument, nextCameraName, parseField, withParameter } from "./hardware";

const field: HardwareField = { name: "port", label: "port", kind: "string", default: "/dev/ttyUSB0", required: true, nullable: false, advanced: false };
const hand: HardwareModel = {
  kind: "end_effector", type: "franka_hand", label: "Franka Hand", fields: [],
  compatibleRobots: ["franka_research3"], mounting: { urdf_path: "fr3_hand.urdf", target_frame_name: "fr3_hand_tcp" },
  capabilities: ["binary_gripper"], description: "", available: true, error: ""
};
const document: HardwareDocument = {
  version: 1, id: "setup", label: "My setup", robot: { type: "franka_research3", params: { robot_ip: "10.0.0.1" } },
  endEffector: { type: "pika", params: { port: "/dev/pika" } },
  teleop: { type: "spacemouse", params: {} }, cameras: [], specializedCameras: false,
  mounting: { target_frame_name: "pika_gripper_ee", start_joint_positions: [1, 2, 3] }
};

describe("portable hardware editing", () => {
  it("keeps installation geometry when editing an IP", () => {
    const moved = { ...document, robot: withParameter(document.robot!, "robot_ip", "10.0.0.2") };
    expect(moved.mounting).toEqual(document.mounting);
    expect(moved.robot.params.robot_ip).toBe("10.0.0.2");
    expect(document.robot!.params.robot_ip).toBe("10.0.0.1");
  });
  it("replaces old TCP and home with the selected tool's mounting preset", () => {
    const changed = changeTool(document, hand);
    expect(changed.mounting).toEqual(hand.mounting);
    expect(changed.mounting).not.toHaveProperty("start_joint_positions");
    expect(changed.endEffector!.type).toBe("franka_hand");
    expect(changed.teleop!.params.tool_mode).toBe("binary");
    expect(document.mounting.target_frame_name).toBe("pika_gripper_ee");
  });
  it("does not carry an incompatible tool or teleoperator to another robot", () => {
    const robot: HardwareModel = { ...hand, kind: "robot", type: "so101_follower", fields: [field] };
    const changed = changeRobot(document, robot);
    expect(changed.robot).toEqual({ type: "so101_follower", params: { port: "/dev/ttyUSB0" } });
    expect(changed.endEffector).toBeNull();
    expect(changed.teleop).toBeNull();
    expect(changed.mounting).toEqual({});
  });
  it("keeps instance defaults independent", () => {
    const model = { ...hand, fields: [{ ...field, name: "limits", default: [1, 2] }] };
    const first = instanceFor(model), second = instanceFor(model);
    (first.params.limits as number[]).push(3);
    expect(second.params.limits).toEqual([1, 2]);
  });
  it("avoids occupied camera role names", () => {
    expect(nextCameraName(["camera_1", "camera_3"])).toBe("camera_2");
  });
  it("checks imported document shapes before rendering dynamic fields", () => {
    expect(isHardwareDocument(document)).toBe(true);
    expect(isHardwareDocument({ ...document, robot: { type: "franka_research3" } })).toBe(false);
    expect(isHardwareDocument({ ...document, cameras: [null] })).toBe(false);
    expect(isHardwareDocument(null)).toBe(false);
  });
});

describe("hardware parameter input", () => {
  it("preserves serial numbers and converts device indices deliberately", () => {
    expect(parseField(field, "0012345")).toBe("0012345");
    expect(parseField({ ...field, kind: "device" }, "0")).toBe(0);
  });
  it("refuses fractional integers and non-finite values", () => {
    expect(() => parseField({ ...field, kind: "integer" }, "30.5")).toThrow();
    expect(() => parseField({ ...field, kind: "number" }, "Infinity")).toThrow();
    expect(() => parseField({ ...field, kind: "number" }, "abc")).toThrow();
  });
  it("validates advanced JSON before a draft can be saved", () => {
    expect(parseField({ ...field, kind: "json" }, "[0.1, 0.2, 0.3]")).toEqual([0.1, 0.2, 0.3]);
    expect(() => parseField({ ...field, kind: "json" }, "[0.1,")).toThrow();
  });
});
