import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import type { EePose, ForceVector } from "./types";

type Vec3 = [number, number, number];
type NamedTrajectory = { name: string; points: Vec3[]; color: number };
type TrajectorySegment = Vec3[];
type NamedPose = { name: string; pose: EePose | null; color: number };

const PIKA_ASSET_BASE = "/api/assets/pika";

const PIKA_TCP_OFFSET = new THREE.Vector3(0.185, 0, 0);
const PIKA_TCP_ROTATION = new THREE.Euler(3.1415926, -1.5707963, 0, "ZYX");

const PIKA_LEFT_ORIGIN = new THREE.Vector3(0.0815, 0.08851, 0.0064182);
const PIKA_RIGHT_ORIGIN = new THREE.Vector3(0.0815, -0.088529, 0.0064182);
const PIKA_JAW_STROKE = 0.05; // per-jaw travel in metres (URDF limit)
const FORCE_VECTOR_METERS_PER_NEWTON = 0.03;
const FORCE_VECTOR_MAX_LENGTH_M = 0.35;
const TRAJECTORY_MAX_STEP_M = 0.25;
// Approximation: use the Pika+ATI chain from fr3_pika_gripper_ati.urdf
// (ati_pika_joint rpy=-0.739815) to orient the wrist force vector.
// The current hardware is Monte gripper + Yuli force sensor; replace this
// with the exact URDF/extrinsic once that model is available.
const PIKA_ATI_TO_GRIPPER_BASE_RPY_X = -0.739815;
const PIKA_ATI_TO_GRIPPER_BASE_ROTATION = new THREE.Quaternion().setFromEuler(
  new THREE.Euler(PIKA_ATI_TO_GRIPPER_BASE_RPY_X, 0, 0, "XYZ")
);
const PIKA_FORCE_SENSOR_TO_GRIPPER_BASE_ROTATION = PIKA_ATI_TO_GRIPPER_BASE_ROTATION.clone().invert();

let cachedMeshPromise: Promise<{
  base: THREE.BufferGeometry;
  left: THREE.BufferGeometry;
  right: THREE.BufferGeometry;
}> | null = null;

async function fetchSTLBuffer(url: string, timeoutMs = 20000): Promise<ArrayBuffer> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) {
      throw new Error(`${url} → HTTP ${response.status} ${response.statusText}`);
    }
    return await response.arrayBuffer();
  } catch (error) {
    if ((error as Error).name === "AbortError") {
      throw new Error(`${url} → timeout after ${timeoutMs} ms`);
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

function loadPikaMeshes(): Promise<{
  base: THREE.BufferGeometry;
  left: THREE.BufferGeometry;
  right: THREE.BufferGeometry;
}> {
  if (cachedMeshPromise) {
    return cachedMeshPromise;
  }
  const loader = new STLLoader();
  const urls = {
    base: `${PIKA_ASSET_BASE}/pika_gripper_base_link.STL`,
    left: `${PIKA_ASSET_BASE}/pika_gripper_left_link.STL`,
    right: `${PIKA_ASSET_BASE}/pika_gripper_right_link.STL`
  };
  cachedMeshPromise = Promise.all([
    fetchSTLBuffer(urls.base).then((buffer) => loader.parse(buffer)),
    fetchSTLBuffer(urls.left).then((buffer) => loader.parse(buffer)),
    fetchSTLBuffer(urls.right).then((buffer) => loader.parse(buffer))
  ])
    .then(([base, left, right]) => {
      [base, left, right].forEach((geometry) => {
        geometry.computeVertexNormals();
      });
      return { base, left, right };
    })
    .catch((error) => {
      cachedMeshPromise = null;
      throw error;
    });
  return cachedMeshPromise;
}

function buildTcpFromBaseMatrix(): THREE.Matrix4 {
  const tcpFromBase = new THREE.Matrix4();
  tcpFromBase.makeRotationFromEuler(PIKA_TCP_ROTATION);
  tcpFromBase.setPosition(PIKA_TCP_OFFSET);
  return tcpFromBase;
}

function buildBaseFromTcpMatrix(): THREE.Matrix4 {
  return buildTcpFromBaseMatrix().invert();
}

function forceSensorToGripperBase(force: THREE.Vector3): THREE.Vector3 {
  return force.clone().applyQuaternion(PIKA_FORCE_SENSOR_TO_GRIPPER_BASE_ROTATION);
}

function gripperBaseToWorldVector(force: THREE.Vector3, baseGroup: THREE.Group): THREE.Vector3 {
  const baseWorldRotation = new THREE.Quaternion();
  baseGroup.getWorldQuaternion(baseWorldRotation);
  return force.clone().applyQuaternion(baseWorldRotation);
}

function makeAxisHelper(length: number): THREE.LineSegments {
  const points: number[] = [
    0, 0, 0, length, 0, 0,
    0, 0, 0, 0, length, 0,
    0, 0, 0, 0, 0, length
  ];
  const colors: number[] = [
    0.76, 0.25, 0.05, 0.76, 0.25, 0.05,
    0.06, 0.46, 0.43, 0.06, 0.46, 0.43,
    0.15, 0.39, 0.92, 0.15, 0.39, 0.92
  ];
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  const material = new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 });
  return new THREE.LineSegments(geometry, material);
}

function makeGroundGrid(size: number, divisions: number): THREE.GridHelper {
  const grid = new THREE.GridHelper(size, divisions, 0xc8d2da, 0xe1e6ea);
  grid.rotation.x = Math.PI / 2;
  return grid;
}

function isFinitePoint(point: Vec3): boolean {
  return point.every(Number.isFinite);
}

function distanceBetween(a: Vec3, b: Vec3): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function splitTrajectory(points: Vec3[], maxStepM = TRAJECTORY_MAX_STEP_M): TrajectorySegment[] {
  const segments: TrajectorySegment[] = [];
  let current: TrajectorySegment = [];
  let previous: Vec3 | null = null;
  for (const point of points) {
    if (!isFinitePoint(point)) {
      if (current.length > 1) {
        segments.push(current);
      }
      current = [];
      previous = null;
      continue;
    }
    if (previous && distanceBetween(previous, point) > maxStepM) {
      if (current.length > 1) {
        segments.push(current);
      }
      current = [];
    }
    current.push(point);
    previous = point;
  }
  if (current.length > 1) {
    segments.push(current);
  }
  return segments;
}

function makeTrajectoryLine(points: Vec3[], color: number): THREE.Line {
  const positions = new Float32Array(points.length * 3);
  for (let i = 0; i < points.length; i++) {
    positions[i * 3 + 0] = points[i][0];
    positions[i * 3 + 1] = points[i][1];
    positions[i * 3 + 2] = points[i][2];
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({ color, linewidth: 2 });
  return new THREE.Line(geometry, material);
}

function disposeLine(line: THREE.Line): void {
  line.geometry.dispose();
  (line.material as THREE.Material).dispose();
}

export function Pose3DViewer({
  trajectory,
  currentPose,
  forceVector = null,
  extraTrajectories = [],
  currentExtraPoses = []
}: {
  trajectory: Vec3[];
  currentPose: EePose | null;
  forceVector?: ForceVector | null;
  extraTrajectories?: NamedTrajectory[];
  currentExtraPoses?: NamedPose[];
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [meshStatus, setMeshStatus] = useState<"loading" | "ready" | "error">("loading");
  const [meshError, setMeshError] = useState<string | null>(null);
  const sceneStateRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    controls: OrbitControls;
    eeGroup: THREE.Group;
    baseGroup: THREE.Group;
    leftJaw: THREE.Mesh | null;
    rightJaw: THREE.Mesh | null;
    trajectoryLines: THREE.Line[];
    extraTrajectoryLines: THREE.Line[];
    extraPoseMarkers: THREE.Mesh[];
    forceArrow: THREE.ArrowHelper;
    eeFrame: THREE.LineSegments;
    requestRender: () => void;
    dispose: () => void;
  } | null>(null);

  const targetCenter = useMemo<Vec3>(() => {
    const allPoints = [trajectory, ...extraTrajectories.map((entry) => entry.points)].flat().filter(isFinitePoint);
    if (!allPoints.length) {
      return [0, 0, 0];
    }
    let sx = 0;
    let sy = 0;
    let sz = 0;
    for (const [x, y, z] of allPoints) {
      sx += x;
      sy += y;
      sz += z;
    }
    return [sx / allPoints.length, sy / allPoints.length, sz / allPoints.length];
  }, [trajectory, extraTrajectories]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    const width = container.clientWidth || 480;
    const height = container.clientHeight || 320;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf8fafb);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.05, 50);
    camera.up.set(0, 0, 1);
    camera.position.set(1.2, -1.2, 0.9);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.18;
    controls.minDistance = 0.2;
    controls.maxDistance = 8;
    controls.target.set(0.5, 0, 0.4);

    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const keyLight = new THREE.DirectionalLight(0xffffff, 0.85);
    keyLight.position.set(1.5, -1.0, 2.0);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0xffffff, 0.35);
    fillLight.position.set(-1.0, 1.5, 1.0);
    scene.add(fillLight);

    scene.add(makeGroundGrid(2, 20));
    scene.add(makeAxisHelper(0.2));

    const eeGroup = new THREE.Group();
    scene.add(eeGroup);

    const eeFrame = makeAxisHelper(0.08);
    eeGroup.add(eeFrame);

    const baseGroup = new THREE.Group();
    baseGroup.applyMatrix4(buildBaseFromTcpMatrix());
    eeGroup.add(baseGroup);

    const forceArrow = new THREE.ArrowHelper(
      new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(0, 0, 0),
      0.1,
      0xdc2626,
      0.035,
      0.018
    );
    forceArrow.visible = false;
    scene.add(forceArrow);

    let animationFrame = 0;
    let renderPending = false;
    function requestRender() {
      if (renderPending) {
        return;
      }
      renderPending = true;
      animationFrame = requestAnimationFrame(() => {
        renderPending = false;
        controls.update();
        renderer.render(scene, camera);
      });
    }

    controls.addEventListener("change", requestRender);

    const resizeObserver = new ResizeObserver(() => {
      const w = container.clientWidth || 480;
      const h = container.clientHeight || 320;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      requestRender();
    });
    resizeObserver.observe(container);

    sceneStateRef.current = {
      scene,
      camera,
      renderer,
      controls,
      eeGroup,
      baseGroup,
      leftJaw: null,
      rightJaw: null,
      trajectoryLines: [],
      extraTrajectoryLines: [],
      extraPoseMarkers: [],
      forceArrow,
      eeFrame,
      requestRender,
      dispose: () => {
        cancelAnimationFrame(animationFrame);
        resizeObserver.disconnect();
        controls.removeEventListener("change", requestRender);
        controls.dispose();
        renderer.dispose();
        renderer.domElement.remove();
      }
    };

    setMeshStatus("loading");
    setMeshError(null);
    loadPikaMeshes()
      .then((meshes) => {
        if (!sceneStateRef.current) {
          return;
        }
        const material = new THREE.MeshStandardMaterial({
          color: 0xd9dce0,
          metalness: 0.15,
          roughness: 0.55,
          flatShading: false
        });
        const baseMesh = new THREE.Mesh(meshes.base, material.clone());
        baseGroup.add(baseMesh);

        const leftJaw = new THREE.Mesh(meshes.left, material.clone());
        leftJaw.position.copy(PIKA_LEFT_ORIGIN);
        baseGroup.add(leftJaw);

        const rightJaw = new THREE.Mesh(meshes.right, material.clone());
        rightJaw.position.copy(PIKA_RIGHT_ORIGIN);
        baseGroup.add(rightJaw);

        sceneStateRef.current.leftJaw = leftJaw;
        sceneStateRef.current.rightJaw = rightJaw;
        setMeshStatus("ready");
        requestRender();
      })
      .catch((error) => {
        console.error("Failed to load pika meshes", error);
        setMeshStatus("error");
        setMeshError(error instanceof Error ? error.message : String(error));
      });

    requestRender();

    return () => {
      sceneStateRef.current?.dispose();
      sceneStateRef.current = null;
    };
  }, []);

  useEffect(() => {
    const state = sceneStateRef.current;
    if (!state) {
      return;
    }
    if (currentPose) {
      state.eeGroup.position.set(currentPose.x, currentPose.y, currentPose.z);
      const quaternion = new THREE.Quaternion(currentPose.qx, currentPose.qy, currentPose.qz, currentPose.qw).normalize();
      state.eeGroup.quaternion.copy(quaternion);
      state.eeGroup.visible = true;

      const gripperValue = currentPose.gripper ?? 0;
      const clamped = Math.max(0, Math.min(1, gripperValue));
      // URDF: q=0 is the open state at the link origin; q at the prismatic limit closes
      // the jaw toward the centerline. Dataset convention: gripper=1 -> fully open.
      const closedAmount = 1 - clamped;
      const leftOffset = -PIKA_JAW_STROKE * closedAmount;
      const rightOffset = PIKA_JAW_STROKE * closedAmount;
      if (state.leftJaw) {
        state.leftJaw.position.set(PIKA_LEFT_ORIGIN.x, PIKA_LEFT_ORIGIN.y + leftOffset, PIKA_LEFT_ORIGIN.z);
      }
      if (state.rightJaw) {
        state.rightJaw.position.set(PIKA_RIGHT_ORIGIN.x, PIKA_RIGHT_ORIGIN.y + rightOffset, PIKA_RIGHT_ORIGIN.z);
      }
    } else {
      state.eeGroup.visible = false;
    }

    const rawForce = forceVector ? new THREE.Vector3(forceVector.x, forceVector.y, forceVector.z) : null;
    const forceMagnitude = forceVector?.magnitude ?? rawForce?.length() ?? 0;
    if (currentPose && rawForce && forceMagnitude > 1e-6) {
      state.eeGroup.updateMatrixWorld(true);
      const forceInGripperBase = forceSensorToGripperBase(rawForce);
      const forceInWorld = gripperBaseToWorldVector(forceInGripperBase, state.baseGroup);
      const direction = forceInWorld.normalize();
      const length = Math.min(FORCE_VECTOR_MAX_LENGTH_M, forceMagnitude * FORCE_VECTOR_METERS_PER_NEWTON);
      const forceOrigin = new THREE.Vector3();
      state.baseGroup.getWorldPosition(forceOrigin);
      state.forceArrow.position.copy(forceOrigin);
      state.forceArrow.setDirection(direction);
      state.forceArrow.setLength(length, 0.035, 0.018);
      state.forceArrow.visible = true;
    } else {
      state.forceArrow.visible = false;
    }
    state.requestRender();
  }, [currentPose, forceVector]);

  useEffect(() => {
    const state = sceneStateRef.current;
    if (!state) {
      return;
    }
    for (const line of state.trajectoryLines) {
      state.scene.remove(line);
      disposeLine(line);
    }
    state.trajectoryLines = [];
    for (const segment of splitTrajectory(trajectory)) {
      const line = makeTrajectoryLine(segment, 0x2563eb);
      state.scene.add(line);
      state.trajectoryLines.push(line);
    }
    for (const line of state.extraTrajectoryLines) {
      state.scene.remove(line);
      disposeLine(line);
    }
    state.extraTrajectoryLines = [];
    for (const entry of extraTrajectories) {
      for (const segment of splitTrajectory(entry.points)) {
        const line = makeTrajectoryLine(segment, entry.color);
        state.scene.add(line);
        state.extraTrajectoryLines.push(line);
      }
    }
    state.controls.target.set(targetCenter[0], targetCenter[1], targetCenter[2]);
    state.requestRender();
  }, [trajectory, extraTrajectories, targetCenter]);

  useEffect(() => {
    const state = sceneStateRef.current;
    if (!state) {
      return;
    }
    for (const marker of state.extraPoseMarkers) {
      state.scene.remove(marker);
      marker.geometry.dispose();
      (marker.material as THREE.Material).dispose();
    }
    state.extraPoseMarkers = [];
    for (const entry of currentExtraPoses) {
      if (!entry.pose) {
        continue;
      }
      const geometry = new THREE.SphereGeometry(0.012, 16, 12);
      const material = new THREE.MeshStandardMaterial({ color: entry.color, roughness: 0.45 });
      const marker = new THREE.Mesh(geometry, material);
      marker.position.set(entry.pose.x, entry.pose.y, entry.pose.z);
      state.scene.add(marker);
      state.extraPoseMarkers.push(marker);
    }
    state.requestRender();
  }, [currentExtraPoses]);

  return (
    <div className="pose-3d" ref={containerRef}>
      {meshStatus !== "ready" ? (
        <div className={`pose-3d-overlay pose-3d-${meshStatus}`}>
          {meshStatus === "loading"
            ? "Loading pika URDF meshes (~8.7 MB)…"
            : `Failed to load gripper meshes: ${meshError}. Check that the gateway is restarted and that /api/assets/pika/* is reachable.`}
        </div>
      ) : null}
    </div>
  );
}
