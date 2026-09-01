import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import type { MujocoPreview, MujocoPreviewBodyPose, MujocoPreviewFrame } from "./types";
import { matrixFromOrigin } from "./urdfOrigin";

type Vec3 = [number, number, number];
type Quat = [number, number, number, number];
type JointType = "fixed" | "revolute" | "prismatic";

type KinematicLink = {
  name: string;
  body_name?: string;
  mesh?: {
    url: string;
    scale?: Vec3;
  };
};

type KinematicJoint = {
  name: string;
  type: JointType;
  parent: string;
  child: string;
  origin: {
    xyz: Vec3;
    rpy: Vec3;
  };
  axis?: Vec3;
};

type KinematicsModel = {
  schema_version: number;
  name: string;
  root: string;
  active_joints: string[];
  links: KinematicLink[];
  joints: KinematicJoint[];
  gripper?: {
    jaw_stroke_m?: number;
    left_joint?: string;
    right_joint?: string;
  };
};

type RobotTrack = {
  key: string;
  label: string;
  frames: MujocoPreviewFrame[];
  baseOffset: Vec3;
};

type BuiltRobot = {
  root: THREE.Group;
  fkRoot: THREE.Group;
  flatRoot: THREE.Group;
  flatLinks: Map<string, THREE.Group>;
  linkBodyNames: Map<string, string>;
  jointGroups: Map<string, THREE.Group>;
  jointAxes: Map<string, THREE.Vector3>;
  jointTypes: Map<string, JointType>;
  targetMarker: THREE.Object3D;
  actualMarker: THREE.Object3D;
};

type SceneState = {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  robots: Map<string, BuiltRobot>;
  model: KinematicsModel | null;
  geometries: Map<string, THREE.BufferGeometry>;
  dispose: () => void;
};

const DEFAULT_KINEMATICS_PATH = "/fr3_mujoco_replay/kinematics.json";
const LINK_COLORS = [0xe5e7eb, 0xd1d5db, 0xf3f4f6, 0xd6d3d1, 0xe2e8f0, 0xdad7cd, 0xe7e5e4, 0xdbeafe];

let modelPromise: Promise<KinematicsModel> | null = null;
let geometryPromise: Promise<Map<string, THREE.BufferGeometry>> | null = null;

async function fetchKinematics(path: string): Promise<KinematicsModel> {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`${path} -> HTTP ${response.status}`);
  return (await response.json()) as KinematicsModel;
}

function loadModel(path: string): Promise<KinematicsModel> {
  if (!modelPromise) {
    modelPromise = fetchKinematics(path).catch((error) => {
      modelPromise = null;
      throw error;
    });
  }
  return modelPromise;
}

async function fetchSTL(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} -> HTTP ${response.status}`);
  return await response.arrayBuffer();
}

function loadGeometries(model: KinematicsModel): Promise<Map<string, THREE.BufferGeometry>> {
  if (geometryPromise) return geometryPromise;
  const loader = new STLLoader();
  geometryPromise = Promise.all(
    model.links
      .filter((link) => link.mesh)
      .map(async (link) => {
        const geometry = loader.parse(await fetchSTL(link.mesh!.url));
        geometry.computeVertexNormals();
        geometry.userData.shared = true;
        return [link.name, geometry] as const;
      })
  )
    .then((entries) => new Map<string, THREE.BufferGeometry>(entries))
    .catch((error) => {
      geometryPromise = null;
      throw error;
    });
  return geometryPromise;
}

function tracksFromPreview(preview: MujocoPreview | null): RobotTrack[] {
  if (!preview) return [];
  const topLevelFrames = preview.frames ?? [];
  if (topLevelFrames.length > 0) {
    return [{ key: "fr3", label: "FR3", frames: topLevelFrames, baseOffset: [0, 0, 0] }];
  }
  return (Object.entries(preview.robots) as Array<[string, { frames?: MujocoPreviewFrame[]; base_offset_m?: Vec3 } | undefined]>)
    .filter((entry): entry is [string, { frames: MujocoPreviewFrame[]; base_offset_m?: Vec3 }] => Array.isArray(entry[1]?.frames) && entry[1]!.frames.length > 0)
    .map(([key, robot]) => ({
      key,
      label: `${key[0].toUpperCase()}${key.slice(1)} FR3`,
      frames: robot.frames,
      baseOffset: robot.base_offset_m ?? [0, 0, 0]
    }));
}

function makeAxis(length: number): THREE.LineSegments {
  const points = [
    0, 0, 0, length, 0, 0,
    0, 0, 0, 0, length, 0,
    0, 0, 0, 0, 0, length
  ];
  const colors = [
    0.9, 0.18, 0.18, 0.9, 0.18, 0.18,
    0.12, 0.65, 0.32, 0.12, 0.65, 0.32,
    0.2, 0.42, 0.9, 0.2, 0.42, 0.9
  ];
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  return new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ vertexColors: true }));
}

function makeMarker(color: number): THREE.Object3D {
  const group = new THREE.Group();
  group.add(new THREE.Mesh(
    new THREE.SphereGeometry(0.014, 18, 12),
    new THREE.MeshStandardMaterial({ color, roughness: 0.5 })
  ));
  group.add(makeAxis(0.075));
  return group;
}

function makeLinkGroup(link: KinematicLink, geometries: Map<string, THREE.BufferGeometry>): THREE.Group {
  const group = new THREE.Group();
  group.name = link.name;
  const geometry = link.mesh ? geometries.get(link.name) : null;
  if (geometry) {
    const colorIndex = Math.max(0, Number.parseInt(link.name.replace(/\D/g, ""), 10) || 0);
    const material = new THREE.MeshStandardMaterial({
      color: LINK_COLORS[colorIndex % LINK_COLORS.length],
      metalness: 0.08,
      roughness: 0.58
    });
    const mesh = new THREE.Mesh(geometry, material);
    const scale = link.mesh?.scale ?? [1, 1, 1];
    mesh.scale.set(scale[0], scale[1], scale[2]);
    group.add(mesh);
  }
  return group;
}

function disposeObject(object: THREE.Object3D): void {
  object.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (mesh.geometry && !(mesh.geometry as THREE.BufferGeometry).userData.shared) {
      mesh.geometry.dispose();
    }
    const material = mesh.material;
    if (Array.isArray(material)) material.forEach((item) => item.dispose());
    else if (material) material.dispose();
  });
}

function buildFkTree(
  model: KinematicsModel,
  geometries: Map<string, THREE.BufferGeometry>,
  jointGroups: Map<string, THREE.Group>,
  jointAxes: Map<string, THREE.Vector3>,
  jointTypes: Map<string, JointType>
): THREE.Group {
  const root = new THREE.Group();
  const linkGroups = new Map<string, THREE.Group>();
  for (const link of model.links) linkGroups.set(link.name, makeLinkGroup(link, geometries));
  const rootLink = linkGroups.get(model.root);
  if (!rootLink) throw new Error(`Kinematics root link missing: ${model.root}`);
  root.add(rootLink);

  for (const joint of model.joints) {
    const parent = linkGroups.get(joint.parent);
    const child = linkGroups.get(joint.child);
    if (!parent || !child) continue;
    const originGroup = new THREE.Group();
    originGroup.name = `${joint.name}:origin`;
    originGroup.applyMatrix4(matrixFromOrigin(joint.origin.xyz, joint.origin.rpy));
    const motionGroup = new THREE.Group();
    motionGroup.name = joint.name;
    originGroup.add(motionGroup);
    motionGroup.add(child);
    parent.add(originGroup);
    jointGroups.set(joint.name, motionGroup);
    jointTypes.set(joint.name, joint.type);
    jointAxes.set(joint.name, new THREE.Vector3(...(joint.axis ?? [0, 0, 1])).normalize());
  }
  return root;
}

function buildFlatLinks(
  model: KinematicsModel,
  geometries: Map<string, THREE.BufferGeometry>,
  flatLinks: Map<string, THREE.Group>,
  linkBodyNames: Map<string, string>
): THREE.Group {
  const root = new THREE.Group();
  for (const link of model.links) {
    const group = makeLinkGroup(link, geometries);
    const bodyName = link.body_name ?? link.name;
    flatLinks.set(link.name, group);
    linkBodyNames.set(link.name, bodyName);
    if (link.mesh || bodyName) root.add(group);
  }
  return root;
}

function buildRobot(model: KinematicsModel, geometries: Map<string, THREE.BufferGeometry>, track: RobotTrack): BuiltRobot {
  const root = new THREE.Group();
  root.position.set(track.baseOffset[0], track.baseOffset[1], track.baseOffset[2]);
  const jointGroups = new Map<string, THREE.Group>();
  const jointAxes = new Map<string, THREE.Vector3>();
  const jointTypes = new Map<string, JointType>();
  const flatLinks = new Map<string, THREE.Group>();
  const linkBodyNames = new Map<string, string>();
  const fkRoot = buildFkTree(model, geometries, jointGroups, jointAxes, jointTypes);
  const flatRoot = buildFlatLinks(model, geometries, flatLinks, linkBodyNames);
  flatRoot.visible = false;
  root.add(fkRoot);
  root.add(flatRoot);

  const targetMarker = makeMarker(0xf97316);
  const actualMarker = makeMarker(0x0891b2);
  targetMarker.visible = false;
  actualMarker.visible = false;
  root.add(targetMarker);
  root.add(actualMarker);
  root.add(makeAxis(0.2));
  return { root, fkRoot, flatRoot, flatLinks, linkBodyNames, jointGroups, jointAxes, jointTypes, targetMarker, actualMarker };
}

function frameForCurrent(track: RobotTrack, currentFrame: number): MujocoPreviewFrame | null {
  if (track.frames.length === 0) return null;
  return track.frames[Math.max(0, Math.min(track.frames.length - 1, currentFrame))];
}

function setJointValue(robot: BuiltRobot, jointName: string, value: number): void {
  const group = robot.jointGroups.get(jointName);
  const axis = robot.jointAxes.get(jointName);
  const type = robot.jointTypes.get(jointName);
  if (!group || !axis || !type || !Number.isFinite(value)) return;
  if (type === "revolute") {
    group.quaternion.setFromAxisAngle(axis, value);
    group.position.set(0, 0, 0);
  } else if (type === "prismatic") {
    group.quaternion.identity();
    group.position.copy(axis.clone().multiplyScalar(value));
  }
}

function setObjectPose(object: THREE.Object3D, position?: Vec3, quaternion?: Quat): void {
  if (!position) {
    object.visible = false;
    return;
  }
  object.position.set(position[0], position[1], position[2]);
  if (quaternion) object.quaternion.set(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
  else object.quaternion.identity();
  object.visible = true;
}

function applyBodyPose(group: THREE.Group, pose: MujocoPreviewBodyPose): void {
  group.position.set(pose.position_m[0], pose.position_m[1], pose.position_m[2]);
  group.quaternion.set(pose.quaternion_xyzw[0], pose.quaternion_xyzw[1], pose.quaternion_xyzw[2], pose.quaternion_xyzw[3]);
  group.visible = true;
}

function applyFlatBodyPoses(robot: BuiltRobot, frame: MujocoPreviewFrame): boolean {
  const bodyPoses = frame.body_poses ?? {};
  if (Object.keys(bodyPoses).length === 0) return false;
  robot.fkRoot.visible = false;
  robot.flatRoot.visible = true;
  robot.flatLinks.forEach((group, linkName) => {
    const bodyName = robot.linkBodyNames.get(linkName) ?? linkName;
    const pose = bodyPoses[bodyName] ?? bodyPoses[linkName];
    if (pose) applyBodyPose(group, pose);
    else group.visible = false;
  });
  return true;
}

function applyFkFrame(model: KinematicsModel, robot: BuiltRobot, frame: MujocoPreviewFrame): void {
  robot.fkRoot.visible = true;
  robot.flatRoot.visible = false;
  const qpos = frame.qpos ?? frame.joints_rad ?? [];
  model.active_joints.forEach((jointName, index) => setJointValue(robot, jointName, qpos[index] ?? 0));
  const gripper = Math.max(0, Math.min(1, frame.gripper ?? 1));
  const stroke = model.gripper?.jaw_stroke_m ?? 0.05;
  const closed = 1 - gripper;
  if (model.gripper?.left_joint) setJointValue(robot, model.gripper.left_joint, -stroke * closed);
  if (model.gripper?.right_joint) setJointValue(robot, model.gripper.right_joint, stroke * closed);
}

function applyFrame(model: KinematicsModel, robot: BuiltRobot, frame: MujocoPreviewFrame | null): void {
  if (!frame) {
    robot.root.visible = false;
    return;
  }
  robot.root.visible = true;
  if (!applyFlatBodyPoses(robot, frame)) applyFkFrame(model, robot, frame);
  setObjectPose(robot.targetMarker, frame.target_position_m, frame.target_quaternion_xyzw);
  setObjectPose(robot.actualMarker, frame.actual_position_m ?? frame.mujoco_position_m, frame.actual_quaternion_xyzw);
}

export function MujocoReplayViewer({ preview, currentFrame }: { preview: MujocoPreview | null; currentFrame: number }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<SceneState | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [error, setError] = useState<string>("");
  const tracks = useMemo(() => tracksFromPreview(preview), [preview]);
  const kinematicsPath = preview?.model?.kinematics_path ?? DEFAULT_KINEMATICS_PATH;
  // Which robots are on the canvas, not what they are doing. A live stream hands this component
  // a new frame array several times a second; rebuilding on the array itself would reload every
  // STL mesh at that rate, which is both a stall and a flicker. The meshes depend on the model
  // and on where each robot's base sits -- nothing else in a frame can change them.
  const robotSignature = useMemo(
    () => tracks.map((track) => `${track.key}@${track.baseOffset.join(",")}`).join("|"),
    [tracks]
  );
  // Read inside the build effect, which must not re-run when the frames change.
  const tracksRef = useRef<RobotTrack[]>(tracks);
  tracksRef.current = tracks;

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const width = container.clientWidth || 720;
    const height = container.clientHeight || 420;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf8fafc);
    const camera = new THREE.PerspectiveCamera(42, width / height, 0.03, 20);
    camera.up.set(0, 0, 1);
    camera.position.set(1.3, -1.35, 0.95);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.14;
    controls.target.set(0.35, 0, 0.38);
    controls.minDistance = 0.35;
    controls.maxDistance = 5;

    scene.add(new THREE.HemisphereLight(0xffffff, 0xd9e2ec, 1.1));
    const key = new THREE.DirectionalLight(0xffffff, 1.1);
    key.position.set(1.4, -1.2, 2.2);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.45);
    fill.position.set(-1.3, 1.2, 1.0);
    scene.add(fill);
    const grid = new THREE.GridHelper(2.4, 24, 0xb8c2cc, 0xe2e8f0);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);

    let animationFrame = 0;
    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      animationFrame = window.requestAnimationFrame(animate);
    };
    animationFrame = window.requestAnimationFrame(animate);

    const resizeObserver = new ResizeObserver(() => {
      const w = container.clientWidth || 720;
      const h = container.clientHeight || 420;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    });
    resizeObserver.observe(container);

    sceneRef.current = {
      scene,
      camera,
      renderer,
      controls,
      robots: new Map(),
      model: null,
      geometries: new Map(),
      dispose: () => {
        window.cancelAnimationFrame(animationFrame);
        resizeObserver.disconnect();
        controls.dispose();
        renderer.dispose();
        renderer.domElement.remove();
      }
    };

    return () => {
      sceneRef.current?.robots.forEach((robot) => {
        scene.remove(robot.root);
        disposeObject(robot.root);
      });
      sceneRef.current?.dispose();
      sceneRef.current = null;
    };
  }, []);

  useEffect(() => {
    const state = sceneRef.current;
    if (!state) return;
    let cancelled = false;
    const builtTracks = tracksRef.current;
    setStatus(builtTracks.length ? "loading" : "idle");
    setError("");
    state.robots.forEach((robot) => {
      state.scene.remove(robot.root);
      disposeObject(robot.root);
    });
    state.robots.clear();
    if (!builtTracks.length) return;

    loadModel(kinematicsPath)
      .then(async (model) => {
        const geometries = await loadGeometries(model);
        if (cancelled || !sceneRef.current) return;
        sceneRef.current.model = model;
        sceneRef.current.geometries = geometries;
        for (const track of builtTracks) {
          const robot = buildRobot(model, geometries, track);
          sceneRef.current.scene.add(robot.root);
          sceneRef.current.robots.set(track.key, robot);
        }
        setStatus("ready");
      })
      .catch((err) => {
        if (cancelled) return;
        setStatus("error");
        setError(err instanceof Error ? err.message : String(err));
      });

    return () => {
      cancelled = true;
    };
  }, [kinematicsPath, robotSignature]);

  useEffect(() => {
    const state = sceneRef.current;
    if (!state?.model) return;
    for (const track of tracks) {
      const robot = state.robots.get(track.key);
      if (robot) applyFrame(state.model, robot, frameForCurrent(track, currentFrame));
    }
  }, [currentFrame, tracks, status]);

  const frame = tracks[0] ? frameForCurrent(tracks[0], currentFrame) : null;
  const hasBodyPoses = Boolean(frame?.body_poses && Object.keys(frame.body_poses).length > 0);
  const statusText = status === "ready"
    ? `${tracks.length} robot view${tracks.length === 1 ? "" : "s"} driven by ${hasBodyPoses ? "MuJoCo body poses" : "qpos FK"}`
    : status === "loading"
      ? "Loading kinematics and STL meshes"
      : status === "error"
        ? `Failed to load model: ${error}`
        : "Run MuJoCo to create a qpos replay report";

  return (
    <div className="mujoco-three-viewer" ref={containerRef}>
      <div className="mujoco-three-hud">
        <span>{preview?.streaming ? `${statusText} · streaming ${preview.stream_frame_count ?? tracks[0]?.frames.length ?? 0} frames` : statusText}</span>
        {frame ? (
          <span>
            qpos {frame.qpos?.length ?? frame.joints_rad?.length ?? 0}D
            {frame.target_frame_name || preview?.target_frame_name ? ` | ${frame.target_frame_name ?? preview?.target_frame_name}` : ""}
            {frame.position_error_mm == null ? "" : ` | pos ${frame.position_error_mm.toFixed(2)} mm`}
            {frame.rotation_error_deg == null ? "" : ` | rot ${frame.rotation_error_deg.toFixed(2)} deg`}
          </span>
        ) : null}
      </div>
      {status !== "ready" ? <div className="mujoco-three-overlay">{statusText}</div> : null}
    </div>
  );
}
