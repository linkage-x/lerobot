import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import type { EePose } from "./types";

type Vec3 = [number, number, number];

const PIKA_ASSET_BASE = "/api/assets/pika";

const PIKA_TCP_OFFSET = new THREE.Vector3(0.185, 0, 0);
const PIKA_TCP_ROTATION = new THREE.Euler(3.1415926, -1.5707963, 0, "ZYX");

const PIKA_LEFT_ORIGIN = new THREE.Vector3(0.0815, 0.08851, 0.0064182);
const PIKA_RIGHT_ORIGIN = new THREE.Vector3(0.0815, -0.088529, 0.0064182);
const PIKA_JAW_STROKE = 0.05; // per-jaw travel in metres (URDF limit)

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

export function Pose3DViewer({
  trajectory,
  currentPose
}: {
  trajectory: Vec3[];
  currentPose: EePose | null;
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
    trajectoryLine: THREE.Line | null;
    eeFrame: THREE.LineSegments;
    requestRender: () => void;
    dispose: () => void;
  } | null>(null);

  const targetCenter = useMemo<Vec3>(() => {
    if (!trajectory.length) {
      return [0, 0, 0];
    }
    let sx = 0;
    let sy = 0;
    let sz = 0;
    for (const [x, y, z] of trajectory) {
      sx += x;
      sy += y;
      sz += z;
    }
    return [sx / trajectory.length, sy / trajectory.length, sz / trajectory.length];
  }, [trajectory]);

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
      trajectoryLine: null,
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
    state.requestRender();
  }, [currentPose]);

  useEffect(() => {
    const state = sceneStateRef.current;
    if (!state) {
      return;
    }
    if (state.trajectoryLine) {
      state.scene.remove(state.trajectoryLine);
      state.trajectoryLine.geometry.dispose();
      (state.trajectoryLine.material as THREE.Material).dispose();
      state.trajectoryLine = null;
    }
    if (trajectory.length > 1) {
      const positions = new Float32Array(trajectory.length * 3);
      for (let i = 0; i < trajectory.length; i++) {
        positions[i * 3 + 0] = trajectory[i][0];
        positions[i * 3 + 1] = trajectory[i][1];
        positions[i * 3 + 2] = trajectory[i][2];
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      const material = new THREE.LineBasicMaterial({ color: 0x2563eb, linewidth: 2 });
      const line = new THREE.Line(geometry, material);
      state.scene.add(line);
      state.trajectoryLine = line;
    }
    state.controls.target.set(targetCenter[0], targetCenter[1], targetCenter[2]);
    state.requestRender();
  }, [trajectory, targetCenter]);

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
