import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { MujocoPreview, MujocoPreviewFrame } from "./types";

type RobotVisual = {
  line: THREE.Line;
  joints: THREE.Mesh[];
  target: THREE.Mesh;
  actual: THREE.Mesh;
};

const FR3_FIXED: Array<{ p: [number, number, number]; q: [number, number, number, number] }> = [
  { p: [0, 0, 0.333], q: [0, 0, 0, 1] },
  { p: [0, 0, 0], q: [-Math.SQRT1_2, 0, 0, Math.SQRT1_2] },
  { p: [0, -0.316, 0], q: [Math.SQRT1_2, 0, 0, Math.SQRT1_2] },
  { p: [0.0825, 0, 0], q: [Math.SQRT1_2, 0, 0, Math.SQRT1_2] },
  { p: [-0.0825, 0.384, 0], q: [-Math.SQRT1_2, 0, 0, Math.SQRT1_2] },
  { p: [0, 0, 0], q: [Math.SQRT1_2, 0, 0, Math.SQRT1_2] },
  { p: [0.088, 0, 0], q: [Math.SQRT1_2, 0, 0, Math.SQRT1_2] }
];

function robotPoints(joints: number[], offset: [number, number, number]): THREE.Vector3[] {
  const transform = new THREE.Matrix4().makeTranslation(...offset);
  const points = [new THREE.Vector3().setFromMatrixPosition(transform)];
  FR3_FIXED.forEach((fixed, index) => {
    const fixedTransform = new THREE.Matrix4().compose(
      new THREE.Vector3(...fixed.p),
      new THREE.Quaternion(...fixed.q),
      new THREE.Vector3(1, 1, 1)
    );
    transform.multiply(fixedTransform);
    points.push(new THREE.Vector3().setFromMatrixPosition(transform));
    transform.multiply(new THREE.Matrix4().makeRotationZ(joints[index] ?? 0));
  });
  transform.multiply(new THREE.Matrix4().makeTranslation(0, 0, 0.107));
  points.push(new THREE.Vector3().setFromMatrixPosition(transform));
  return points;
}

function closestFrame(frames: MujocoPreviewFrame[], frameIndex: number): MujocoPreviewFrame | null {
  if (!frames.length) return null;
  let best = frames[0];
  for (const frame of frames) {
    if (Math.abs(frame.frame_index - frameIndex) < Math.abs(best.frame_index - frameIndex)) best = frame;
  }
  return best;
}

export function MujocoReplayViewer({ preview, currentFrame }: { preview: MujocoPreview; currentFrame: number }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const visualsRef = useRef<Partial<Record<"left" | "right", RobotVisual>>>({});
  const renderRef = useRef<(() => void) | null>(null);
  const robotNames = useMemo(() => (["left", "right"] as const).filter((name) => preview.robots[name]), [preview]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const width = container.clientWidth || 760;
    const height = container.clientHeight || 430;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf4f7f8);
    const camera = new THREE.PerspectiveCamera(42, width / height, 0.05, 20);
    camera.up.set(0, 0, 1);
    camera.position.set(1.8, -2.0, 1.25);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0.25, 0, 0.45);
    scene.add(new THREE.GridHelper(3, 30, 0x94a3b8, 0xdbe3e8).rotateX(Math.PI / 2));
    scene.add(new THREE.AmbientLight(0xffffff, 1.6));
    const light = new THREE.DirectionalLight(0xffffff, 2.0);
    light.position.set(2, -2, 3);
    scene.add(light);

    const visuals: Partial<Record<"left" | "right", RobotVisual>> = {};
    robotNames.forEach((name) => {
      const robot = preview.robots[name]!;
      const color = name === "left" ? 0xc2410c : 0x0f766e;
      const line = new THREE.Line(
        new THREE.BufferGeometry(),
        new THREE.LineBasicMaterial({ color: 0x475569, linewidth: 3 })
      );
      scene.add(line);
      const joints = Array.from({ length: 9 }, () => {
        const mesh = new THREE.Mesh(
          new THREE.SphereGeometry(0.035, 18, 12),
          new THREE.MeshStandardMaterial({ color: 0xe2e8f0, metalness: 0.25, roughness: 0.45 })
        );
        scene.add(mesh);
        return mesh;
      });
      const target = new THREE.Mesh(
        new THREE.SphereGeometry(0.025, 16, 12),
        new THREE.MeshStandardMaterial({ color, transparent: true, opacity: 0.55 })
      );
      const actual = new THREE.Mesh(
        new THREE.SphereGeometry(0.018, 16, 12),
        new THREE.MeshStandardMaterial({ color: 0x2563eb })
      );
      scene.add(target, actual);
      const trajectoryPoints = robot.frames.map((frame) => new THREE.Vector3(
        frame.target_position_m[0] + robot.base_offset_m[0],
        frame.target_position_m[1] + robot.base_offset_m[1],
        frame.target_position_m[2] + robot.base_offset_m[2]
      ));
      if (trajectoryPoints.length > 1) {
        scene.add(new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(trajectoryPoints),
          new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.45 })
        ));
      }
      visuals[name] = { line, joints, target, actual };
    });
    visualsRef.current = visuals;

    let animation = 0;
    const render = () => {
      controls.update();
      renderer.render(scene, camera);
      animation = requestAnimationFrame(render);
    };
    renderRef.current = () => renderer.render(scene, camera);
    render();
    const resize = new ResizeObserver(() => {
      const nextWidth = container.clientWidth || width;
      const nextHeight = container.clientHeight || height;
      camera.aspect = nextWidth / nextHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(nextWidth, nextHeight);
    });
    resize.observe(container);
    return () => {
      cancelAnimationFrame(animation);
      resize.disconnect();
      controls.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      visualsRef.current = {};
      renderRef.current = null;
    };
  }, [preview, robotNames]);

  useEffect(() => {
    robotNames.forEach((name) => {
      const robot = preview.robots[name]!;
      const frame = closestFrame(robot.frames, currentFrame);
      const visual = visualsRef.current[name];
      if (!frame || !visual) return;
      const points = robotPoints(frame.joints_rad, robot.base_offset_m);
      visual.line.geometry.dispose();
      visual.line.geometry = new THREE.BufferGeometry().setFromPoints(points);
      visual.joints.forEach((joint, index) => joint.position.copy(points[Math.min(index, points.length - 1)]));
      visual.target.position.set(
        frame.target_position_m[0] + robot.base_offset_m[0],
        frame.target_position_m[1] + robot.base_offset_m[1],
        frame.target_position_m[2] + robot.base_offset_m[2]
      );
      visual.actual.position.set(
        frame.mujoco_position_m[0] + robot.base_offset_m[0],
        frame.mujoco_position_m[1] + robot.base_offset_m[1],
        frame.mujoco_position_m[2] + robot.base_offset_m[2]
      );
    });
    renderRef.current?.();
  }, [currentFrame, preview, robotNames]);

  return <div className="mujoco-3d" ref={containerRef} />;
}
