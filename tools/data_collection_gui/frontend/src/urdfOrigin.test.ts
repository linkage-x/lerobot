import { describe, expect, it } from "vitest";
import * as THREE from "three";
// The model the viewer actually fetches at runtime, so these assertions move with it rather
// than with a copy of its numbers.
import kinematics from "../public/fr3_mujoco_replay/kinematics.json";
import { matrixFromOrigin } from "./urdfOrigin";

type Vec3 = [number, number, number];

function vec3(values: number[]): Vec3 {
  const [x, y, z] = values;
  return [x, y, z];
}

function originOf(jointName: string): { xyz: number[]; rpy: number[] } {
  const joint = kinematics.joints.find((entry) => entry.name === jointName);
  if (!joint) throw new Error(`joint missing from kinematics.json: ${jointName}`);
  return joint.origin;
}

function applied(origin: { xyz: number[]; rpy: number[] }, local: Vec3): number[] {
  return new THREE.Vector3(...local)
    .applyMatrix4(matrixFromOrigin(vec3(origin.xyz), vec3(origin.rpy)))
    .toArray()
    .map((value) => Number(value.toFixed(4)));
}

describe("matrixFromOrigin", () => {
  it("mounts the gripper along the flange axis, not back into the wrist", () => {
    // The Pika gripper's own +x runs from its mount out through the jaws, and fr3_link8's +z
    // is the flange axis -- straight out of the wrist. Under URDF's fixed-axis rpy the first
    // is the second; read in THREE.Euler's default "XYZ" ordering it comes out 165 degrees
    // away, which is what a wrongly-oriented gripper looks like on the live viewer.
    expect(applied(originOf("fr3_gripper_joint"), [0.0815, 0, 0])).toEqual([0.002, 0, 0.0895]);
  });

  it("spreads the jaws across the flange axis", () => {
    // The finger joints travel along the gripper's local +/-y, which must come out
    // perpendicular to the flange axis rather than along it.
    const [, , z] = applied(originOf("fr3_gripper_joint"), [0, 1, 0]);
    expect(Math.abs(z - 0.008)).toBeLessThan(1e-6);
  });

  it("names the gripper mount as the model's only multi-axis origin", () => {
    // Why the arm has always drawn correctly and only the gripper has not: rpy ordering is
    // unobservable until two of the three components are non-zero, and this is the one origin
    // where they are. If the model ever gains another, the convention stops being a detail
    // confined to the wrist and this test says which joint made it one.
    const multiAxis = kinematics.joints
      .filter((joint) => joint.origin.rpy.filter((value) => Math.abs(value) > 1e-9).length > 1)
      .map((joint) => joint.name);
    expect(multiAxis).toEqual(["fr3_gripper_joint"]);
  });

  it("keeps the translation in the parent frame", () => {
    expect(applied({ xyz: [0.1, 0.2, 0.3], rpy: [0, 0, 0] }, [0, 0, 0])).toEqual([0.1, 0.2, 0.3]);
  });
});
