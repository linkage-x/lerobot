"""Shared scene of the real-input image runner.

Inputs, target models, mounts, truth trajectories and the fisheye ray renderer
all live here so the runner and the tests cannot disagree about geometry.
Units are metres and radians; ``T_A_B`` maps B coordinates into A.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
KALIBR = ROOT / 'third_party/opencv_kalibr'
HIKON = KALIBR / 'hikon_cube_tracking_offline'
for _path in (KALIBR, HIKON):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from metrology.calibration_io import CameraCalibration  # noqa: E402
from metrology.cli.compare_fiducial_targets import marker_layout_as_carrier  # noqa: E402
from metrology.hybrid_carrier import HybridCarrierModel, project_rig, transform  # noqa: E402
from metrology.hybrid_carrier_render import PaintPalette  # noqa: E402

INPUTS = HERE / 'inputs'
CAD_DIR = KALIBR / 'metrology/fixtures/cad'
R0_CAD = CAD_DIR / 'marker_rig_20260818_cad.json'
HYBRID_DESCRIPTOR = CAD_DIR / 'hybrid_carrier_v1_20260907.json'
HYBRID_PRODUCTION_LAYOUT = HIKON / 'config_thor/hybrid_carrier_v1_20260907_marker_layout.json'

ARMS = ('R0', 'H0', 'H1')
TARGET_OF_ARM = {'R0': 'R0', 'H0': 'H', 'H1': 'H'}
R0_PLATE_OF_MARKER = {7: 4, 12: 2, 14: 1, 16: 0, 17: 3}
# Two-capture means from box_umi_marker_rig_tcp_20260818.html (same values as simulator-core.js).
R0_MEASURED_EDGE_MM = {7: 60.88, 12: 60.63, 14: 60.55, 16: 60.56, 17: 61.12}
R0_PAD_MM = 69.6
R0_BRACKET_FACE_BASE = 900
BGR_OCCLUDER = (58, 60, 64)
BGR_SKIN = (120, 150, 196)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pose_matrix(position, quat_xyzw) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = position
    return T


def invert(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def transform_triangles(T: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    triangles = np.asarray(triangles, dtype=np.float64)
    if not len(triangles):
        return triangles.reshape(0, 3, 3)
    return triangles @ T[:3, :3].T + T[:3, 3]


# --- targets -----------------------------------------------------------------

R0_MEASURED_INPUT = 'r0_layout_measured.json'


def r0_measured_layout(doc: dict, cad: dict) -> dict:
    """A measured rig0818 layout, checked before it can replace the proxy.

    It must be ``marker_layout/measured_v1`` in metres, expressed in the CAD
    part-studio frame (origin = socket sphere centre; a data-defined frame
    would silently move the TCP), hold exactly the five rig ids, and wind each
    marker clockwise about the CAD plate's outward normal (ArUco order).
    """
    if doc.get('schema') != 'marker_layout/measured_v1':
        raise ValueError('R0 layout must be marker_layout/measured_v1')
    if str(doc.get('units', 'm')).lower() != 'm':
        raise ValueError('R0 layout must be in metres')
    if 'CAD part-studio frame' not in str(doc.get('frame_convention', '')):
        raise ValueError('R0 layout must be expressed in the CAD frame (origin = socket sphere centre)')
    by_id = {int(m['id']): m for m in doc.get('markers', [])}
    if sorted(by_id) != sorted(R0_PLATE_OF_MARKER) or len(doc['markers']) != len(R0_PLATE_OF_MARKER):
        raise ValueError(f'R0 layout must hold exactly ids {sorted(R0_PLATE_OF_MARKER)}, got {sorted(by_id)}')
    markers = []
    for marker_id, index in R0_PLATE_OF_MARKER.items():
        corners = np.asarray(by_id[marker_id]['corners_rig'], dtype=np.float64)
        if corners.shape != (4, 3) or not np.isfinite(corners).all():
            raise ValueError(f'marker {marker_id}: corners_rig must be 4 finite xyz points')
        outward = -np.cross(corners[1] - corners[0], corners[2] - corners[0])
        outward /= np.linalg.norm(outward)
        if float(outward @ np.asarray(cad['plates'][index]['normal'], dtype=np.float64)) < 0.9:
            raise ValueError(f'marker {marker_id}: corner winding or plate assignment disagrees with the CAD normal')
        markers.append({'id': marker_id, 'corners_rig': corners.tolist()})
    return {'schema': 'marker_layout/measured_v1', 'layout_id': str(doc.get('layout_id', 'r0_measured')),
            'units': 'm', 'geometry_status': 'measured', 'markers': markers}


def r0_proxy_layout(cad: dict) -> dict:
    """CAD plate centres/normals with the measured sticker edge; ArUco (CW) winding.

    Mirrors ``buildRigTarget`` in simulator-core.js without a measured layout.
    """
    markers = []
    for marker_id, index in R0_PLATE_OF_MARKER.items():
        plate = cad['plates'][index]
        centre = np.asarray(plate['centre_mm'], dtype=np.float64)
        corners = np.asarray(plate['corners_mm'], dtype=np.float64)
        scaled = (centre + (corners - centre) * R0_MEASURED_EDGE_MM[marker_id] / float(plate['edge_mm'])) * 1e-3
        markers.append({'id': marker_id, 'corners_rig': scaled[::-1].tolist()})
    return {'schema': 'marker_layout/measured_v1', 'layout_id': 'rig0818_cad_measured_size_proxy',
            'units': 'm', 'geometry_status': 'proxy', 'markers': markers}


def _square_prism(a: np.ndarray, b: np.ndarray, half: float) -> list:
    axis = b - a
    axis = axis / np.linalg.norm(axis)
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    ring = [half * (su * u + sv * v) for su, sv in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    tris = []
    for k in range(4):
        p0, p1 = a + ring[k], a + ring[(k + 1) % 4]
        q0, q1 = b + ring[k], b + ring[(k + 1) % 4]
        tris += [[p0.tolist(), p1.tolist(), q1.tolist()], [p0.tolist(), q1.tolist(), q0.tolist()]]
    return tris


def r0_carrier_doc(layout: dict) -> dict:
    """The five pads as an anchors-only carrier, plus the black bracket as rods.

    The 0818 bracket mesh is not in the repository. Each arm is drawn as a 10 mm
    square rod from the socket to the back of its plate so the pads can shadow
    each other the way a printed bracket makes them; the rods are declared.
    """
    doc = marker_layout_as_carrier(layout, carrier_id=str(layout.get('layout_id', 'rig0818')), pad_size_mm=R0_PAD_MM)
    triangles = list(doc['occluders']['triangles'])
    face_ids = list(doc['occluders']['face_entity_ids'])
    for k, anchor in enumerate(doc['anchors']):
        centre = np.asarray(anchor['centre'], dtype=np.float64)
        normal = np.asarray(anchor['normal'], dtype=np.float64)
        rod = _square_prism(np.zeros(3), centre - 4.0 * normal, 5.0)
        triangles += rod
        face_ids += [R0_BRACKET_FACE_BASE + k] * len(rod)
    doc['occluders'] = {'triangles': triangles, 'face_entity_ids': face_ids}
    doc['bracket'] = {'status': 'assumed', 'rod_half_width_mm': 5.0,
                      'note': 'socket-to-plate rods standing in for the unavailable bracket mesh'}
    return doc


def hybrid_layout(model: HybridCarrierModel) -> dict:
    """Anchor corners in detection order -- what export_carrier_marker_layout writes."""
    return {'layout_id': 'hybrid_carrier_v1_20260907', 'units': 'm', 'markers': [
        {'id': int(a.marker_id), 'corners_rig': a.object_points(a.paste_quadrant or 0).tolist()}
        for a in model.anchors]}


def cameras_from_doc(doc: dict) -> list[CameraCalibration]:
    return [CameraCalibration(c['name'], c.get('serial', ''), int(c['width']), int(c['height']),
                              np.asarray(c['K'], dtype=np.float64), np.asarray(c['D'], dtype=np.float64),
                              np.asarray(c['T_world_camera'], dtype=np.float64), model='fisheye',
                              sigma_pixel=float(c.get('sigma_pixel', 0.2)))
            for c in doc['cameras']]


@dataclass
class Scene:
    cameras: list[CameraCalibration]
    camera_doc: dict
    task: dict
    gripper: dict
    layouts: dict
    carrier_docs: dict
    models: dict
    T_box_tcp: np.ndarray
    mechanism: np.ndarray
    housing: np.ndarray
    hand: list
    sources: dict
    r0_status: str = 'proxy'

    def body_colour(self, target: str) -> tuple:
        return PaintPalette().black if target == 'R0' else PaintPalette().body


def load_scene(inputs: Path = INPUTS) -> Scene:
    camera_path, task_path, gripper_path = (inputs / 'cameras_0804_fisheye.json', inputs / 'task_132514.json',
                                            inputs / 'gripper_v2.json')
    camera_doc, task, gripper = load_json(camera_path), load_json(task_path), load_json(gripper_path)
    cad, hybrid_doc = load_json(R0_CAD), load_json(HYBRID_DESCRIPTOR)
    measured_path = inputs / R0_MEASURED_INPUT
    measured = measured_path.exists()
    r0_layout = r0_measured_layout(load_json(measured_path), cad) if measured else r0_proxy_layout(cad)
    r0_doc = r0_carrier_doc(r0_layout)
    models = {'R0': HybridCarrierModel.from_dict(r0_doc), 'H': HybridCarrierModel.from_dict(hybrid_doc)}
    layouts = {'R0': r0_layout, 'H': hybrid_layout(models['H'])}
    files = (camera_path, task_path, gripper_path, R0_CAD, HYBRID_DESCRIPTOR) + ((measured_path,) if measured else ())
    sources = {rel(p): sha256(p) for p in files}
    return Scene(
        cameras=cameras_from_doc(camera_doc), camera_doc=camera_doc, task=task, gripper=gripper,
        layouts=layouts, carrier_docs={'R0': r0_doc, 'H': hybrid_doc}, models=models,
        T_box_tcp=np.asarray(gripper['T_box_tcp'], dtype=np.float64),
        mechanism=np.asarray(gripper['mechanism']['triangles'], dtype=np.float64),
        housing=np.asarray(gripper['housing']['triangles'], dtype=np.float64),
        hand=[{'c': np.asarray(s['c'], dtype=np.float64), 'r': float(s['r'])} for s in gripper['hand']['spheres']],
        sources=sources, r0_status='measured' if measured else 'proxy')


# --- mounts and truth ----------------------------------------------------------

def target_min_z(model: HybridCarrierModel) -> float:
    return float(np.min(model.occluder_triangles[:, :, 2]))


def mount_T_box_rig(scene: Scene, target: str, group: str, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    """Where a target's CAD frame sits on the BOX.

    Both groups turn the target by ``yaw_deg`` about the box z axis and then tilt
    it by ``pitch_deg`` about its own y axis (negative lifts the +x boom), so CAD
    +z starts along box +z: the socket opens toward the housing, the dome is up.
    ``gripper``: the socket sits on the declared stud over the housing, with the
    lowest point of the rotated target 5 mm clear of it. ``socket_tcp``: only the
    socket moves, onto the TCP point, with no own gripper. The carrier bundle's
    identity rotation relates the carrier's own rig and TCP frames; it is not the
    URDF TCP's axes (TCP z is -y of the box), and taking those laid both targets
    on their side until 2026-09-11.
    """
    if group not in ('gripper', 'socket_tcp'):
        raise ValueError(f'unknown mount group {group!r}')
    R = Rotation.from_euler('ZY', [float(yaw_deg), float(pitch_deg)], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    if group == 'socket_tcp':
        T[:3, 3] = scene.T_box_tcp[:3, 3]
        return T
    mount = scene.gripper['mount']['gripper']
    lowest = float(np.min((scene.models[target].occluder_triangles.reshape(-1, 3) @ R.T)[:, 2]))
    T[:3, 3] = [mount['socket_xy_box_m'][0], mount['socket_xy_box_m'][1],
                mount['housing_top_z_m'] + mount['clearance_m'] - lowest]
    return T


def T_rig_tcp(scene: Scene, T_box_rig: np.ndarray) -> np.ndarray:
    return invert(T_box_rig) @ scene.T_box_tcp


@dataclass
class Truth:
    hand: str
    index: int
    episode: int
    frame: int
    t: float
    interpolated: bool
    T_world_box: np.ndarray
    v: np.ndarray  # world velocity of the box origin, m/s
    w: np.ndarray  # world angular velocity, rad/s

    def box_at(self, dt: float) -> np.ndarray:
        """Constant-twist extrapolation of the box pose by ``dt`` seconds."""
        T = self.T_world_box.copy()
        if dt:
            T[:3, :3] = Rotation.from_rotvec(self.w * dt).as_matrix() @ T[:3, :3]
            T[:3, 3] = T[:3, 3] + self.v * dt
        return T


def hand_truth(scene: Scene, hand: str) -> dict[int, Truth]:
    doc = scene.task['hands'][hand]
    T_cube_box = np.asarray(doc['T_cube_box'], dtype=np.float64)
    requested = np.asarray(doc['requested'], dtype=bool)
    episode = np.asarray(doc['episode'])
    t = np.asarray(doc['t_s'], dtype=np.float64)
    boxes = {}
    for i in np.flatnonzero(requested):
        boxes[i] = pose_matrix(doc['T_world_cube_position_m'][i], doc['T_world_cube_quat_xyzw'][i]) @ T_cube_box
    out = {}
    for i, T in boxes.items():
        lo = i - 1 if (i - 1) in boxes and episode[i - 1] == episode[i] else i
        hi = i + 1 if (i + 1) in boxes and episode[i + 1] == episode[i] else i
        dt = t[hi] - t[lo]
        if hi == lo or dt <= 0:
            v, w = np.zeros(3), np.zeros(3)
        else:
            v = (boxes[hi][:3, 3] - boxes[lo][:3, 3]) / dt
            w = Rotation.from_matrix(boxes[hi][:3, :3] @ boxes[lo][:3, :3].T).as_rotvec() / dt
        out[int(i)] = Truth(hand, int(i), int(episode[i]), int(doc['frame_in_episode'][i]), float(t[i]),
                            bool(doc['interpolated'][i]), T, v, w)
    return out


def tcp_speed(truth: Truth, T_box_tcp: np.ndarray) -> float:
    r = truth.T_world_box[:3, :3] @ T_box_tcp[:3, 3]
    return float(np.linalg.norm(truth.v + np.cross(truth.w, r)))


def gripper_occluders(scene: Scene, T_world_box: np.ndarray) -> tuple[np.ndarray, list]:
    triangles = np.concatenate([transform_triangles(T_world_box, scene.mechanism),
                                transform_triangles(T_world_box, scene.housing)])
    spheres = [{'c': T_world_box[:3, :3] @ s['c'] + T_world_box[:3, 3], 'r': s['r']} for s in scene.hand]
    return triangles, spheres


def frame_occluders(scene: Scene, group: str, own_box: np.ndarray, other_box: np.ndarray | None):
    triangles, spheres = [], []
    if group == 'gripper':
        tris, sph = gripper_occluders(scene, own_box)
        triangles.append(tris)
        spheres += sph
    if other_box is not None:
        tris, sph = gripper_occluders(scene, other_box)
        triangles.append(tris)
        spheres += sph
    return (np.concatenate(triangles) if triangles else np.zeros((0, 3, 3))), spheres


# --- renderer ------------------------------------------------------------------

def nearest_hits(origin: np.ndarray, rays: np.ndarray, triangles: np.ndarray,
                 ray_chunk: int = 60000, tri_chunk: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Batched Moller-Trumbore: nearest positive distance along non-unit rays."""
    n = len(rays)
    depth = np.full(n, np.inf)
    which = np.full(n, -1, dtype=np.int64)
    if not len(triangles) or not n:
        return depth, which
    v0 = triangles[:, 0]
    e1, e2 = triangles[:, 1] - v0, triangles[:, 2] - v0
    s = origin[None, :] - v0
    q = np.cross(s, e1)
    t_num = np.einsum('bk,bk->b', e2, q)
    for rs in range(0, n, ray_chunk):
        R = rays[rs:rs + ray_chunk]
        rows = np.arange(len(R))
        for ts in range(0, len(triangles), tri_chunk):
            sl = slice(ts, ts + tri_chunk)
            h = np.cross(R[:, None, :], e2[None, sl, :])
            a = np.einsum('rbk,bk->rb', h, e1[sl])
            valid = np.abs(a) > 1e-12
            inv = np.divide(1.0, a, out=np.zeros_like(a), where=valid)
            u = np.einsum('rbk,bk->rb', h, s[sl]) * inv
            v = (R @ q[sl].T) * inv
            t = t_num[sl][None, :] * inv
            ok = valid & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 0)
            t = np.where(ok, t, np.inf)
            j = np.argmin(t, axis=1)
            tj = t[rows, j]
            better = tj < depth[rs:rs + len(R)]
            idx = np.flatnonzero(better)
            depth[rs + idx] = tj[idx]
            which[rs + idx] = j[idx] + ts
    return depth, which


def sphere_hits(origin: np.ndarray, rays: np.ndarray, spheres: list) -> np.ndarray:
    depth = np.full(len(rays), np.inf)
    a = np.einsum('ij,ij->i', rays, rays)
    for sphere in spheres:
        f = origin - sphere['c']
        b = 2.0 * (rays @ f)
        disc = b * b - 4.0 * a * (f @ f - sphere['r'] ** 2)
        near = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a)
        hit = (disc >= 0) & (near > 0)
        depth = np.where(hit & (near < depth), near, depth)
    return depth


class SceneRenderer:
    """Subpixel fisheye rays against a target plus world occluders.

    Only a window around the target is ray-cast; occluders outside that window
    cannot hide a feature, so they are culled before the intersection test.
    """

    def __init__(self, supersample: int = 2):
        self.supersample = int(supersample)
        self.palette = PaintPalette()
        self.textures: dict = {}
        self._samples: dict = {}

    def _target_samples(self, model: HybridCarrierModel) -> np.ndarray:
        key = id(model)
        if key not in self._samples:
            tri = model.occluder_triangles
            alpha = np.linspace(0, 1, 13)[None, :, None]
            edges = [tri[:, i, None, :] * (1 - alpha) + tri[:, (i + 1) % 3, None, :] * alpha for i in range(3)]
            self._samples[key] = np.concatenate(edges).reshape(-1, 3)
        return self._samples[key]

    def window(self, model: HybridCarrierModel, camera: CameraCalibration, poses: list) -> tuple | None:
        samples = self._target_samples(model)
        lo, hi = np.array([np.inf, np.inf]), np.array([-np.inf, -np.inf])
        for T in poses:
            Tcr = camera.T_cam_base @ T
            if np.any(transform(Tcr, samples)[:, 2] <= 0.05):
                return None
            uv = project_rig(camera, Tcr, samples)
            lo, hi = np.minimum(lo, uv.min(axis=0)), np.maximum(hi, uv.max(axis=0))
        x0, y0 = np.maximum(np.floor(lo - 4), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(hi + 5), [camera.width, camera.height]).astype(int)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        return int(x0), int(y0), int(x1), int(y1)

    def _cull(self, camera, triangles, window, max_depth):
        if not len(triangles):
            return triangles
        cam = transform_triangles(camera.T_cam_base, triangles)
        z = cam[:, :, 2]
        front = np.any(z > 0.02, axis=1) & (np.min(np.linalg.norm(cam, axis=2), axis=1) < max_depth)
        keep = np.zeros(len(triangles), bool)
        straddle = front & np.any(z <= 0.02, axis=1)
        keep |= straddle
        ahead = np.flatnonzero(front & ~straddle)
        if len(ahead):
            uv = project_rig(camera, np.eye(4), cam[ahead].reshape(-1, 3)).reshape(-1, 3, 2)
            x0, y0, x1, y1 = window
            overlap = ((uv[:, :, 0].max(1) >= x0 - 2) & (uv[:, :, 0].min(1) <= x1 + 2)
                       & (uv[:, :, 1].max(1) >= y0 - 2) & (uv[:, :, 1].min(1) <= y1 + 2))
            keep[ahead[overlap]] = True
        return triangles[keep]

    def window_rays(self, camera: CameraCalibration, window: tuple) -> np.ndarray:
        """Unit-depth camera-frame rays of the supersampled window, row-major."""
        s = self.supersample
        x0, y0, x1, y1 = window
        yy, xx = np.mgrid[0:(y1 - y0) * s, 0:(x1 - x0) * s]
        # OpenCV integer coordinates identify pixel centres.
        pixels = np.column_stack([(xx.ravel() + .5) / s + x0 - .5, (yy.ravel() + .5) / s + y0 - .5])
        undistorted = cv2.fisheye.undistortPoints(pixels[:, None, :], camera.K, camera.D).reshape(-1, 2)
        return np.column_stack([undistorted, np.ones(len(pixels))])

    def _grid_boxes(self, camera, Tcr, window, points_rig):
        """Supersampled-grid bounding boxes of projected point groups (M, k, 3).

        Groups with a point near or behind the camera get the whole window.
        """
        s = self.supersample
        x0, y0, x1, y1 = window
        W, H = (x1 - x0) * s, (y1 - y0) * s
        M, k = points_rig.shape[:2]
        boxes = np.tile(np.array([0, H, 0, W]), (M, 1))
        cam = transform(Tcr, points_rig.reshape(-1, 3)).reshape(M, k, 3)
        ahead = np.all(cam[:, :, 2] > 0.02, axis=1)
        if ahead.any():
            uv = project_rig(camera, Tcr, points_rig[ahead].reshape(-1, 3)).reshape(-1, k, 2)
            lo, hi = uv.min(axis=1), uv.max(axis=1)
            c0 = np.floor((lo[:, 0] - x0 + .5) * s - .5).astype(int) - 2
            c1 = np.ceil((hi[:, 0] - x0 + .5) * s - .5).astype(int) + 3
            r0 = np.floor((lo[:, 1] - y0 + .5) * s - .5).astype(int) - 2
            r1 = np.ceil((hi[:, 1] - y0 + .5) * s - .5).astype(int) + 3
            boxes[ahead] = np.column_stack([np.clip(r0, 0, H), np.clip(r1, 0, H), np.clip(c0, 0, W), np.clip(c1, 0, W)])
        return boxes

    def _raster_hits(self, camera, Tcr, window, origin, rays, triangles):
        """Nearest hit per ray, testing each triangle only inside its projected box.

        Edges are sampled 7 times so a fisheye-curved edge cannot bulge out of
        the box; 2 supersampled pixels of margin cover the chord between samples.
        """
        s = self.supersample
        x0, y0, x1, y1 = window
        H, W = (y1 - y0) * s, (x1 - x0) * s
        depth = np.full((H, W), np.inf)
        which = np.full((H, W), -1, dtype=np.int64)
        if not len(triangles):
            return depth.ravel(), which.ravel()
        alpha = np.linspace(0, 1, 7)[None, :, None]
        dense = np.concatenate([triangles[:, i, None, :] * (1 - alpha) + triangles[:, (i + 1) % 3, None, :] * alpha
                                for i in range(3)], axis=1)
        boxes = self._grid_boxes(camera, Tcr, window, dense)
        grid = rays.reshape(H, W, 3)
        for m, (r0, r1, c0, c1) in enumerate(boxes):
            if r1 <= r0 or c1 <= c0:
                continue
            sub = grid[r0:r1, c0:c1].reshape(-1, 3)
            v0, e1, e2 = triangles[m, 0], triangles[m, 1] - triangles[m, 0], triangles[m, 2] - triangles[m, 0]
            h = np.cross(sub, e2)
            a = h @ e1
            valid = np.abs(a) > 1e-12
            inv = np.divide(1.0, a, out=np.zeros_like(a), where=valid)
            sv = origin - v0
            u = (h @ sv) * inv
            q = np.cross(sv, e1)
            v = (sub @ q) * inv
            t = float(e2 @ q) * inv
            ok = valid & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 0)
            t = np.where(ok, t, np.inf).reshape(r1 - r0, c1 - c0)
            view = depth[r0:r1, c0:c1]
            better = t < view
            view[better] = t[better]
            which[r0:r1, c0:c1][better] = m
        return depth.ravel(), which.ravel()

    def _sphere_hits(self, camera, Tcr, window, origin, rays, spheres):
        s = self.supersample
        x0, y0, x1, y1 = window
        H, W = (y1 - y0) * s, (x1 - x0) * s
        depth = np.full((H, W), np.inf)
        grid = rays.reshape(H, W, 3)
        for sphere in spheres:
            c, r = sphere['c'], sphere['r']
            view_dir = c - origin
            d = float(np.linalg.norm(view_dir))
            if d <= r * 1.05:
                box = (0, H, 0, W)
            else:
                n = view_dir / d
                u = np.cross(n, [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0])
                u /= np.linalg.norm(u)
                v = np.cross(n, u)
                radius = r * d / np.sqrt(d * d - r * r)
                ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
                ring = c + radius * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v)
                box = tuple(self._grid_boxes(camera, Tcr, window, ring[None])[0])
            r0, r1, c0, c1 = box
            if r1 <= r0 or c1 <= c0:
                continue
            sub = grid[r0:r1, c0:c1].reshape(-1, 3)
            f = origin - c
            a = np.einsum('ij,ij->i', sub, sub)
            b = 2.0 * (sub @ f)
            disc = b * b - 4.0 * a * (f @ f - r * r)
            near = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a)
            t = np.where((disc >= 0) & (near > 0), near, np.inf).reshape(r1 - r0, c1 - c0)
            view = depth[r0:r1, c0:c1]
            np.minimum(view, t, out=view)
        return depth.ravel()

    def render_window(self, model: HybridCarrierModel, camera: CameraCalibration, T: np.ndarray, window: tuple,
                      occluder_triangles: np.ndarray, spheres: list, body_colour: tuple,
                      rays_cam: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        s = self.supersample
        x0, y0, x1, y1 = window
        Tcr = camera.T_cam_base @ T
        if rays_cam is None:
            rays_cam = self.window_rays(camera, window)
        rays = rays_cam @ Tcr[:3, :3]
        origin = -Tcr[:3, :3].T @ Tcr[:3, 3]
        depth, which = self._raster_hits(camera, Tcr, window, origin, rays, model.occluder_triangles)
        target_hit = np.isfinite(depth)
        max_depth = float(np.linalg.norm(camera.T_cam_base[:3, :3] @ T[:3, 3] + camera.T_cam_base[:3, 3])) + 0.4
        world_to_rig = invert(T)
        occ_depth = np.full(len(rays), np.inf)
        occ_is_sphere = np.zeros(len(rays), bool)
        culled = self._cull(camera, occluder_triangles, window, max_depth)
        if len(culled):
            occ_depth, _ = self._raster_hits(camera, Tcr, window, origin, rays, transform_triangles(world_to_rig, culled))
        if spheres:
            local = [{'c': world_to_rig[:3, :3] @ sp['c'] + world_to_rig[:3, 3], 'r': sp['r']} for sp in spheres]
            sphere_depth = self._sphere_hits(camera, Tcr, window, origin, rays, local)
            occ_is_sphere = sphere_depth < occ_depth
            occ_depth = np.minimum(occ_depth, sphere_depth)
        occluded = target_hit & (occ_depth < depth)
        visible = target_hit & ~occluded
        rgb = np.empty((len(rays), 3), np.float32)
        rgb[:] = self.palette.background
        show_occluder = np.isfinite(occ_depth) & ~visible
        rgb[show_occluder & ~occ_is_sphere] = BGR_OCCLUDER
        rgb[show_occluder & occ_is_sphere] = BGR_SKIN
        points = origin + rays * np.where(visible, depth, 0.0)[:, None]
        face = np.full(len(rays), -1, dtype=np.int64)
        face[visible] = model.occluder_face_ids[which[visible]]
        rgb[visible] = body_colour
        for facet in model.facets:
            mask = face == facet.face_entity_id
            if not mask.any():
                continue
            rgb[mask] = self.palette.of(facet.colour)
            if model.border_width_m <= 0:
                continue
            p, polygon = points[mask], facet.polygon_m
            distances = []
            for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                d = b - a
                t = np.clip((p - a) @ d / np.dot(d, d), 0, 1)
                distances.append(np.linalg.norm(p - a - t[:, None] * d, axis=1))
            width = model.border_width_m * (.5 if model.border_alignment == 'centred' else 1.)
            ink = np.min(distances, axis=0) <= width
            rgb[np.flatnonzero(mask)[ink]] = self.palette.border
        for anchor in model.anchors:
            mask = face == anchor.face_entity_id
            if not mask.any() or np.dot(anchor.normal, origin - anchor.centre_m) <= 0:
                continue
            rgb[mask] = self.palette.white
            corners = anchor.object_points(anchor.paste_quadrant or 0)
            basis = np.column_stack([corners[1] - corners[0], corners[3] - corners[0]])
            st = (points[mask] - corners[0]) @ np.linalg.pinv(basis).T
            inside = np.all((st >= 0) & (st < 1), axis=1)
            key = (anchor.dictionary, anchor.marker_id)
            if key not in self.textures:
                dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, anchor.dictionary))
                self.textures[key] = cv2.aruco.generateImageMarker(dictionary, anchor.marker_id, 256)
            xy = np.clip((st[inside] * 256).astype(int), 0, 255)
            rgb[np.flatnonzero(mask)[inside]] = self.textures[key][xy[:, 1], xy[:, 0]][:, None]
        crop = rgb.reshape((y1 - y0) * s, (x1 - x0) * s, 3)
        crop = cv2.resize(crop, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        stats = {'target_px': float(target_hit.sum()) / s / s, 'occluded_px': float(occluded.sum()) / s / s}
        return crop, stats

    def render_camera(self, model, camera, poses: list, occluders: list, body_colour: tuple, *,
                      blur_sigma: float, noise_sigma: float, rng: np.random.Generator):
        """Exposure-integrated view: one ray-cast per sub-pose, averaged.

        ``poses`` and ``occluders`` are parallel lists over the exposure; the
        middle sample provides the occlusion statistics.
        """
        window = self.window(model, camera, poses)
        if window is None:
            return None, None, {'target_px': 0.0, 'occluded_px': 0.0}
        crops, stats = [], None
        middle = len(poses) // 2
        rays_cam = self.window_rays(camera, window)
        for k, (T, (tris, spheres)) in enumerate(zip(poses, occluders)):
            crop, st = self.render_window(model, camera, T, window, tris, spheres, body_colour, rays_cam)
            crops.append(crop)
            if k == middle:
                stats = st
        image = np.full((camera.height, camera.width, 3), self.palette.background, np.float32)
        x0, y0, x1, y1 = window
        image[y0:y1, x0:x1] = np.mean(crops, axis=0)
        # Blur and noise over a margin around the window only; the rest is flat background.
        m = 8
        bx0, by0, bx1, by1 = max(0, x0 - m), max(0, y0 - m), min(camera.width, x1 + m), min(camera.height, y1 + m)
        region = image[by0:by1, bx0:bx1]
        if blur_sigma:
            region = cv2.GaussianBlur(region, (0, 0), blur_sigma)
        region = region + rng.normal(0.0, noise_sigma, region.shape).astype(np.float32)
        out = np.full((camera.height, camera.width, 3), self.palette.background, np.uint8)
        out[by0:by1, bx0:bx1] = np.clip(np.rint(region), 0, 255).astype(np.uint8)
        return out, (bx0, by0, bx1, by1), stats
