#!/usr/bin/env python3
"""Offline ray-render -> real ArUco/native Hybrid -> shared multi-camera BA pilot.

No web server, hardware access, truth seed, or target-dependent success lottery.
Outputs a self-contained JSON report importable by the static browser page.
The native per-camera branch is kept separate from the H0/H1 controlled ablation.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'third_party/opencv_kalibr'))
from metrology.calibration_io import CameraCalibration
from metrology.cli.compare_fiducial_targets import marker_layout_as_carrier
from metrology.hybrid_carrier import (HybridCarrierModel, CarrierView, _pnp_candidates,
    build_aruco_detector, detect_carrier, solve_carrier_pose, project_rig, transform)
from metrology.hybrid_carrier_render import PaintPalette


def camera_from_json(c):
    T = np.eye(4)
    T[:3, :3] = np.column_stack([c['right'], c['down'], c['forward']])
    T[:3, 3] = c['position']
    return CameraCalibration(c['name'], 'synthetic', c['width'], c['height'],
        np.array([[c['fx'], 0, c['cx']], [0, c['fy'], c['cy']], [0, 0, 1.]]),
        np.array(c['D']), T, model='fisheye')


def pose_matrix(p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = p['R'], p['t']
    return T


def intersect_triangle(origin, rays, tri):
    """Independent Moller-Trumbore ray oracle; distances along non-unit rays."""
    e1, e2 = tri[1] - tri[0], tri[2] - tri[0]
    h = np.cross(rays, e2)
    a = h @ e1
    inv = np.divide(1., a, out=np.zeros_like(a), where=np.abs(a) > 1e-12)
    s = origin - tri[0]
    u = (h @ s) * inv
    q = np.cross(s, e1)
    v = (rays @ q) * inv
    t = np.dot(e2, q) * inv
    return np.where((np.abs(a) > 1e-12) & (u >= -1e-9) & (v >= -1e-9)
                    & (u + v <= 1 + 1e-9) & (t > 0), t, np.inf)


class RayRenderer:
    """Subpixel fisheye rays, exact triangle depth and metric-space textures.

    The crop only accelerates rendering; edges are densely sampled to bound
    curved fisheye projections. Rays still use full-image pixel coordinates.
    """
    def __init__(self, supersample=2):
        self.supersample = supersample
        self.palette = PaintPalette()
        self.textures = {}

    def render(self, model, camera, T, spheres=()):
        s = self.supersample
        Tcr = camera.T_cam_base @ T
        triangles = model.occluder_triangles
        vertices = triangles.reshape(-1, 3)
        samples = np.concatenate([triangles[:, i, None, :] * (1 - np.linspace(0, 1, 17)[None, :, None])
            + triangles[:, (i+1) % 3, None, :] * np.linspace(0, 1, 17)[None, :, None] for i in range(3)]).reshape(-1, 3)
        cam_points = transform(Tcr, samples)
        image = np.full((camera.height, camera.width, 3), self.palette.background, np.uint8)
        if not np.all(cam_points[:, 2] > .01):
            raise ValueError('Target intersects camera near plane; unsupported pilot scene')
        uv = project_rig(camera, Tcr, samples)
        x0, y0 = np.maximum(np.floor(uv.min(axis=0) - 4), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(uv.max(axis=0) + 5), [camera.width, camera.height]).astype(int)
        if x1 <= x0 or y1 <= y0:
            return image
        # OpenCV integer coordinates identify pixel centres.
        yy, xx = np.mgrid[0:(y1-y0)*s, 0:(x1-x0)*s]
        pixels = np.column_stack([(xx.ravel()+.5)/s+x0-.5, (yy.ravel()+.5)/s+y0-.5])
        undistorted = cv2.fisheye.undistortPoints(pixels[:, None, :], camera.K, camera.D).reshape(-1, 2)
        rays_cam = np.column_stack([undistorted, np.ones(len(pixels))])
        rays = rays_cam @ Tcr[:3, :3]
        origin = -Tcr[:3, :3].T @ Tcr[:3, 3]
        depth, face = np.full(len(rays), np.inf), np.full(len(rays), -1, int)
        for tri, face_id in zip(triangles, model.occluder_face_ids):
            d = intersect_triangle(origin, rays, tri)
            hit = d < depth
            depth[hit], face[hit] = d[hit], face_id
        visible = np.isfinite(depth)
        points = origin + rays * np.where(visible, depth, 0)[:, None]
        rgb = np.full((len(rays), 3), self.palette.background, np.uint8)
        rgb[visible] = self.palette.body
        for facet in model.facets:
            mask = face == facet.face_entity_id
            rgb[mask] = self.palette.of(facet.colour)
            if not mask.any() or model.border_width_m <= 0:
                continue
            p, polygon = points[mask], facet.polygon_m
            distances = []
            for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                d = b-a
                t = np.clip((p-a) @ d / np.dot(d,d), 0, 1)
                distances.append(np.linalg.norm(p-a-t[:,None]*d, axis=1))
            width = model.border_width_m * (.5 if model.border_alignment == 'centred' else 1.)
            ink = np.min(distances, axis=0) <= width
            rgb[np.flatnonzero(mask)[ink]] = self.palette.border
        for anchor in model.anchors:
            mask = face == anchor.face_entity_id
            if np.dot(anchor.normal,origin-anchor.centre_m) <= 0:
                continue  # An unprinted pad back must not display a mirrored ID.
            rgb[mask] = self.palette.white
            if not mask.any():
                continue
            # Corner 0 of the actual sticker follows the descriptor quadrant.
            corners = anchor.object_points(anchor.paste_quadrant or 0)
            basis = np.column_stack([corners[1]-corners[0], corners[3]-corners[0]])
            st = (points[mask]-corners[0]) @ np.linalg.pinv(basis).T
            inside = np.all((st >= 0) & (st < 1), axis=1)
            key = (anchor.dictionary, anchor.marker_id)
            if key not in self.textures:
                dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, anchor.dictionary))
                self.textures[key] = cv2.aruco.generateImageMarker(dictionary, anchor.marker_id, 256)
            xy = np.clip((st[inside]*256).astype(int), 0, 255)
            value = self.textures[key][xy[:,1], xy[:,0]]
            rgb[np.flatnonzero(mask)[inside]] = value[:,None]
        for sphere in spheres:
            centre = transform(np.linalg.inv(T), np.asarray(sphere['c'])[None])[0]
            f = origin-centre
            a = np.einsum('ij,ij->i', rays, rays)
            b = 2*(rays @ f)
            disc = b*b-4*a*(np.dot(f,f)-sphere['r']**2)
            near = (-b-np.sqrt(np.maximum(0,disc)))/(2*a)
            blocked = (disc >= 0) & (near > 0) & (near < depth)
            rgb[blocked] = self.palette.obstruction
        crop = rgb.reshape((y1-y0)*s, (x1-x0)*s, 3)
        image[y0:y1,x0:x1] = cv2.resize(crop, (x1-x0,y1-y0), interpolation=cv2.INTER_AREA)
        return image


def anchor_views(images, cameras, model, detector):
    views, candidates, ledger = [], [], []
    for image, camera in zip(images, cameras):
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        decoded = [] if ids is None else list(zip(ids.ravel().tolist(), corners))
        known = [(model.anchors_by_id[i], np.asarray(p).reshape(4,2)) for i,p in decoded if i in model.anchors_by_id]
        ledger.append({'camera':camera.name, 'ids':[i for i,_ in decoded], 'known_count':len(known)})
        if not known:
            continue
        obj = np.vstack([a.object_points(a.paste_quadrant or 0) for a,_ in known])
        uv = np.vstack([p for _,p in known])
        views.append(CarrierView(camera, obj, uv))
        try:
            candidates.extend(camera.T_base_cam @ p for p in _pnp_candidates(camera, obj, uv))
        except cv2.error:
            pass
    return views, candidates, ledger


def joint_solve(views, candidates):
    if len(views) < 2 or not candidates:
        return None, 'insufficient_multicamera_anchors'
    def score(T):
        residuals = []
        for v in views:
            Tcr = v.camera.T_cam_base @ T
            if np.any(transform(Tcr,v.anchor_points_rig)[:,2] <= .01):
                return np.inf
            residuals.extend(np.linalg.norm(project_rig(v.camera,Tcr,v.anchor_points_rig)-v.anchor_uv,axis=1))
        return float(np.mean(np.minimum(residuals,20))) if residuals else np.inf
    initial = min(candidates, key=score)
    if not np.isfinite(score(initial)):
        return None, 'no_positive_depth_initialization'
    fit = solve_carrier_pose(views, initial)
    if fit.message or not np.isfinite(fit.T_base_rig).all():
        return None, 'solver_failed'
    if fit.covariance is None or not np.isfinite(fit.covariance).all():
        return None, 'singular_information'
    if fit.anchor_rmse_px > 3:
        return None, 'anchor_reprojection_gate'
    return fit, None


def evaluate(fit, reason, truth, tcp, index, episode, views):
    row = {'index':index,'episode':episode,'accepted':fit is not None,'failureReason':reason,
           'translationErrorMm':None,'rotationErrorDeg':None,'cameraCount':len(views),
           'cornerRows':sum(len(v.anchor_uv)*2 for v in views),
           'edgeRows':sum(len(v.edges) for v in views)}
    if fit is not None:
        estimate = fit.T_base_rig
        row.update(translationErrorMm=float(np.linalg.norm(transform(estimate,tcp[None])-transform(truth,tcp[None]))*1000),
            rotationErrorDeg=float(Rotation.from_matrix(estimate[:3,:3] @ truth[:3,:3].T).magnitude()*180/np.pi),
            estimate=estimate.tolist(),anchorRmsePx=float(fit.anchor_rmse_px),
            covariance=fit.covariance.tolist(), covarianceConvention='Rodrigues rotation then translation (native solver)')
    return row


def summarize(rows):
    accepted = [r for r in rows if r['accepted']]
    errors = [r['translationErrorMm'] for r in accepted]
    rot = [r['rotationErrorDeg'] for r in accepted]
    p = lambda data,q:float(np.quantile(data,q)) if data else None
    return dict(total=len(rows),accepted=len(accepted),p50=p(errors,.5),p95=p(errors,.95),p99=p(errors,.99),
        rotationP95=p(rot,.95),coverage=len(accepted)/len(rows),
        accurateYield=sum(r['translationErrorMm']<=3 and r['rotationErrorDeg']<=.5 for r in accepted)/len(rows),
        catastrophicRate=sum(r['translationErrorMm']>20 or r['rotationErrorDeg']>5 for r in accepted)/len(rows),
        failureReasons=dict(Counter(r['failureReason'] for r in rows if not r['accepted'])))


def run(args):
    sources = [HERE/'l2_runner.py',HERE/'simulator-core.js',HERE/'export-world.mjs',ROOT/'third_party/opencv_kalibr/metrology/hybrid_carrier.py',
        ROOT/'third_party/opencv_kalibr/metrology/fixtures/cad/marker_rig_20260818_cad.json',
        ROOT/'third_party/opencv_kalibr/metrology/fixtures/cad/hybrid_carrier_v1_20260907.json']
    source_hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    config = dict(poses=args.poses,episodes=min(4,args.poses),occlusion=args.occlusion,
                  seed=args.seed,linearSpeedMps=args.speed,angularSpeedDegS=args.angular_speed)
    command = ['node', str(HERE/'export-world.mjs'), json.dumps(config)]
    if args.layout:
        command.append(str(args.layout.resolve()))
    world = json.loads(subprocess.check_output(command, text=True))
    layout = {'schema':'marker_layout/measured_v1','units':'m','markers':[
        {'id':a['id'],'corners_rig':a['points']} for a in world['rig']['anchors']]}
    rig_doc = marker_layout_as_carrier(layout,carrier_id='R0',pad_size_mm=69.6)
    model_r = HybridCarrierModel.from_dict(rig_doc)
    model_h = HybridCarrierModel.from_dict(world['hybrid'])
    cameras = [camera_from_json(c) for c in world['cameras']]
    renderer = RayRenderer(args.supersample)
    detector = build_aruco_detector('DICT_6X6_50')
    rows = {a:[] for a in ['R0','H0','H1']}
    native, preview, frame_ledger = [], [], []
    tcp = np.array([args.tcp_mm*.001,0,0])
    started = time.perf_counter()
    for index,p in enumerate(world['poses']):
        truth = pose_matrix(p)
        images_by_arm, views_by_arm, seeds_by_arm = {}, {}, {}
        ledger = {'index':index,'episode':p['episode'],'truth':truth.tolist(),'cameras':{},'native':[]}
        for arm,model in [('R0',model_r),('H0',model_h)]:
            images = []
            for ci,camera in enumerate(cameras):
                samples = []
                exposure_times = [0.] if args.exposure_samples == 1 else np.linspace(-args.exposure_ms/2000,args.exposure_ms/2000,args.exposure_samples)
                for dt in exposure_times:
                    during = truth.copy()
                    during[:3,3] += np.array(p['velocity'])*dt
                    during[:3,:3] = Rotation.from_rotvec(np.array(p['angularVelocity'])*dt).as_matrix() @ truth[:3,:3]
                    samples.append(renderer.render(model,camera,during,p['occluders']))
                image = np.mean(samples,axis=0)
                if args.blur_sigma:
                    image = cv2.GaussianBlur(image,(0,0),args.blur_sigma)
                rng = np.random.default_rng(args.seed+index*7+ci)
                image = np.clip(np.rint(image+rng.normal(0,args.noise_sigma,image.shape)),0,255).astype(np.uint8)
                images.append(image)
            views,seeds,decoded = anchor_views(images,cameras,model,detector)
            images_by_arm[arm],views_by_arm[arm],seeds_by_arm[arm] = images,views,seeds
            ledger['cameras'][arm] = decoded
            fit,reason = joint_solve(views,seeds)
            rows[arm].append(evaluate(fit,reason,truth,tcp,index,p['episode'],views))
            if index == 0:
                ci = max(range(len(decoded)),key=lambda i:decoded[i]['known_count'])
                ok,png = cv2.imencode('.png',images[ci])
                preview.append({'arm':arm,'camera':cameras[ci].name,'image':'data:image/png;base64,'+base64.b64encode(png).decode()})
        # H1 uses the identical H0 image/anchor observations and H0 initialization.
        edge_views = copy.deepcopy(views_by_arm['H0'])
        by_name = {v.camera.name:v for v in edge_views}
        native_views, native_seeds = [], []
        for image,camera in zip(images_by_arm['H0'],cameras):
            detection = detect_carrier(image,camera,model_h,detector=detector,allow_anchor_free=False)
            ledger['native'].append({'camera':camera.name,'success':detection.success,
                'message':detection.message,'anchor_ids':[a.marker_id for a in detection.anchors],
                'edge_count':len(detection.measurements),'seeded':detection.seeded})
            if not detection.success:
                continue
            anchors = detection.anchors
            obj = np.vstack([model_h.anchors_by_id[a.marker_id].object_points(a.quadrant) for a in anchors])
            uv = np.vstack([a.corners_uv for a in anchors])
            # Correlated samples cannot manufacture independent information.
            groups = Counter(m.edge_index for m in detection.measurements)
            edges = [replace(m,sigma_px=m.sigma_px*np.sqrt(1+(groups[m.edge_index]-1)*args.edge_rho)) for m in detection.measurements]
            native_views.append(CarrierView(camera,obj,uv,edges=edges))
            native_seeds.append(detection.T_base_rig)
            if camera.name in by_name:
                by_name[camera.name].edges = edges
        h0 = rows['H0'][-1]
        seed = [np.array(h0['estimate'])] if h0['accepted'] else []
        fit,reason = joint_solve(edge_views,seed)
        if fit is None and h0['accepted']:
            # Declared controlled fallback: preserve the common anchor solution.
            row = copy.deepcopy(h0)
            row['fallback'] = reason
        else:
            row = evaluate(fit,reason,truth,tcp,index,p['episode'],edge_views)
        rows['H1'].append(row)
        fit,reason = joint_solve(native_views,native_seeds)
        native.append(evaluate(fit,reason,truth,tcp,index,p['episode'],native_views))
        frame_ledger.append(ledger)
        print(f'L2 {index+1}/{args.poses}: '+ ' '.join(f'{a}={rows[a][-1]["translationErrorMm"]}' for a in rows), flush=True)
    report = dict(schema='rig_target_ab_l2/v1',layer='L2 image pilot',config=vars(args)|{'out':str(args.out),'layout':str(args.layout) if args.layout else None},
        elapsedSeconds=time.perf_counter()-started,rowsByArm=rows,metrics={a:summarize(r) for a,r in rows.items()},
        native={'name':'H1 native cold-start + joint BA','rows':native,'metrics':summarize(native)},
        previews=preview,frames=frame_ledger,world=world,
        provenance={'opencv':cv2.__version__,'files':source_hashes},
        limitations=['Synthetic camera/task domain; R0 support mesh unavailable.',
            'Ray-rendered ideal diffuse colours; real lighting and calibrated noise remain unvalidated.',
            'Native detector is cold-start per camera; controlled H1 uses native edge measurements with H0 joint initialization.',
            'Joint pilot gate: >=2 cameras, finite covariance, anchor RMSE <=3px; not the full frozen production strategy.',
            'No evidence-backed L3 biases or smoothing in this image-layer result.'],decision='INCONCLUSIVE')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,allow_nan=False),encoding='utf8')
    print(f'Saved {args.out} ({report["elapsedSeconds"]:.1f}s)',flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'l2_report.json')
    parser.add_argument('--layout',type=Path)
    parser.add_argument('--poses',type=int,default=12)
    parser.add_argument('--seed',type=int,default=20260910)
    parser.add_argument('--supersample',type=int,choices=[1,2,3,4],default=2)
    parser.add_argument('--noise-sigma',type=float,default=1.)
    parser.add_argument('--blur-sigma',type=float,default=.4)
    parser.add_argument('--occlusion',type=float,default=.22)
    parser.add_argument('--speed',type=float,default=0.)
    parser.add_argument('--angular-speed',type=float,default=0.)
    parser.add_argument('--tcp-mm',type=float,default=120.)
    parser.add_argument('--exposure-ms',type=float,default=5.)
    parser.add_argument('--exposure-samples',type=int,default=1)
    parser.add_argument('--edge-rho',type=float,default=.55)
    args = parser.parse_args()
    if not 1 <= args.poses <= 1200 or not 1 <= args.exposure_samples <= 25:
        parser.error('poses must be 1..1200 and exposure-samples 1..25')
    if not 0 <= args.occlusion <= .75 or not 0 <= args.edge_rho <= .95:
        parser.error('invalid occlusion or edge correlation')
    if min(args.noise_sigma,args.blur_sigma,args.speed,args.angular_speed,args.exposure_ms) < 0:
        parser.error('noise, blur, speed and exposure must be nonnegative')
    if args.exposure_samples == 1 and (args.speed or args.angular_speed):
        parser.error('moving exposures need at least 2 exposure samples')
    cv2.setNumThreads(1)
    run(args)


if __name__ == '__main__':
    main()
