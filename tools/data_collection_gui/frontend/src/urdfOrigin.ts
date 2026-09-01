import * as THREE from "three";

type Vec3 = [number, number, number];

/**
 * A URDF joint origin (`xyz` + `rpy`) as a matrix.
 *
 * `rpy` is URDF's fixed-axis convention: roll about the parent's x, then pitch about the
 * parent's y, then yaw about the parent's z, each about axes that do not move --
 * `R = Rz(yaw) * Ry(pitch) * Rx(roll)`. THREE.Euler spells that composition "ZYX". Its default
 * "XYZ" multiplies the same three numbers in the opposite order, which is a different rotation
 * whenever more than one of them is non-zero.
 *
 * Every FR3 arm joint has exactly one non-zero rpy component, so the two conventions agree
 * there and the arm draws correctly under either. `fr3_gripper_joint` is the single origin in
 * this model with all three set (-0.7398, -pi/2, pi), so it is the only place the mistake is
 * visible -- and it is worth 165 degrees: read as "XYZ" the gripper points back into the wrist
 * instead of out along the flange axis, landing its jaws 152 mm from where they are.
 */
export function matrixFromOrigin(xyz: Vec3, rpy: Vec3): THREE.Matrix4 {
  const matrix = new THREE.Matrix4().makeRotationFromEuler(
    new THREE.Euler(rpy[0], rpy[1], rpy[2], "ZYX")
  );
  matrix.setPosition(xyz[0], xyz[1], xyz[2]);
  return matrix;
}
