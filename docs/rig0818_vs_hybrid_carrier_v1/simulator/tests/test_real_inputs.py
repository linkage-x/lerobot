"""Checks on the real-input scene: parity with production files, renderer oracles, mounts."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import ab_scene as S  # noqa: E402
from metrology.hybrid_carrier import build_aruco_detector  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402


class RealInputSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        cls.scene = S.load_scene()
        cls.left = S.hand_truth(cls.scene, 'left')
        cls.right = S.hand_truth(cls.scene, 'right')
        cls.index = sorted(cls.left)[700]
        cls.detector = build_aruco_detector('DICT_6X6_50')

    def test_carrier_layout_is_the_production_layout(self):
        production = S.load_json(S.HYBRID_PRODUCTION_LAYOUT)
        ours = {m['id']: np.asarray(m['corners_rig']) for m in self.scene.layouts['H']['markers']}
        for marker in production['markers']:
            np.testing.assert_array_equal(ours[marker['id']], np.asarray(marker['corners_rig']))

    def test_r0_geometry_is_the_browser_geometry(self):
        command = ['node', str(HERE / 'export-world.mjs'), '{"poses":1,"episodes":1}']
        if self.scene.r0_status == 'measured':
            command.append(str(S.INPUTS / S.R0_MEASURED_INPUT))
        world = json.loads(subprocess.check_output(command, text=True))
        browser = {a['id']: np.asarray(a['points']) for a in world['rig']['anchors']}
        for marker in self.scene.layouts['R0']['markers']:
            np.testing.assert_allclose(browser[marker['id']], marker['corners_rig'], atol=1e-12)

    def test_measured_r0_layout_is_guarded(self):
        cad = S.load_json(S.R0_CAD)
        proxy = S.r0_proxy_layout(cad)
        doc = {'schema': 'marker_layout/measured_v1', 'units': 'm', 'layout_id': 'guard',
               'frame_convention': 'CAD part-studio frame; origin = pivot socket sphere centre', 'markers': proxy['markers']}
        self.assertEqual(S.r0_measured_layout(doc, cad)['geometry_status'], 'measured')
        with self.assertRaises(ValueError):  # a data-defined frame would silently move the TCP
            S.r0_measured_layout(dict(doc, frame_convention='origin = centroid of marker centres'), cad)
        with self.assertRaises(ValueError):  # reversed winding = mirrored marker
            S.r0_measured_layout(dict(doc, markers=[dict(m, corners_rig=m['corners_rig'][::-1]) for m in doc['markers']]), cad)
        with self.assertRaises(ValueError):
            S.r0_measured_layout(dict(doc, markers=doc['markers'][:4]), cad)
        if self.scene.r0_status == 'measured':
            # The real stickers are pasted at their own quarter turns (7 and 14 differ from the proxy's first
            # corner), so geometry agrees to millimetres only up to a cyclic roll of the corner order.
            for a, b in zip(self.scene.layouts['R0']['markers'], proxy['markers']):
                measured, cad_corners = np.asarray(a['corners_rig']), np.asarray(b['corners_rig'])
                best = min(np.max(np.linalg.norm(np.roll(measured, k, axis=0) - cad_corners, axis=1)) for k in range(4))
                self.assertLess(best, 0.004, f'marker {a["id"]}')

    def test_gripper_chain_agrees_with_the_task_bundle(self):
        from freeze_inputs import urdf_box_tcp
        np.testing.assert_allclose(urdf_box_tcp(), self.scene.T_box_tcp, atol=1e-9)
        for hand in ('left', 'right'):
            doc = self.scene.task['hands'][hand]
            T_cube_box, T_cube_tcp = np.asarray(doc['T_cube_box']), np.asarray(doc['T_cube_tcp'])
            np.testing.assert_allclose(T_cube_box @ self.scene.T_box_tcp, T_cube_tcp, atol=1e-8)

    def test_every_target_sits_the_same_clearance_above_the_housing(self):
        mount = self.scene.gripper['mount']['gripper']
        for target in ('R0', 'H'):
            for yaw in mount['yaw_candidates_deg']:
                for pitch in mount['pitch_candidates_deg']:
                    T = S.mount_T_box_rig(self.scene, target, 'gripper', yaw, pitch)
                    lowest = S.transform_triangles(T, self.scene.models[target].occluder_triangles)[:, :, 2].min()
                    self.assertAlmostEqual(lowest, mount['housing_top_z_m'] + mount['clearance_m'], places=9)
                    np.testing.assert_allclose(T[:2, 3], mount['socket_xy_box_m'], atol=1e-12)
                    boom = T[:3, :3] @ [1.0, 0.0, 0.0]
                    self.assertAlmostEqual(np.degrees(np.arcsin(boom[2])), -pitch, places=9)
        # Socket-TCP control: the socket sits on the TCP point, turned exactly like the gripper mount (dome up).
        # Taking the TCP's own axes instead would lay the carrier on its side (TCP z is -y of the box).
        for target in ('R0', 'H'):
            for yaw in mount['yaw_candidates_deg']:
                for pitch in mount['pitch_candidates_deg']:
                    T = S.mount_T_box_rig(self.scene, target, 'socket_tcp', yaw, pitch)
                    np.testing.assert_allclose(S.T_rig_tcp(self.scene, T)[:3, 3], 0.0, atol=1e-12)
                    np.testing.assert_allclose(T[:3, :3], S.mount_T_box_rig(self.scene, target, 'gripper', yaw, pitch)[:3, :3],
                                               atol=1e-12)

    def test_point_error_is_centroid_error_plus_rotation_times_arm(self):
        import l3_runner as R
        theta = np.deg2rad(0.5)
        R_est = Rotation.from_rotvec([0.0, 0.0, theta]).as_matrix()[None]
        R_true = np.eye(3)[None]
        centroid = np.array([0.1, 0.0, 0.0])
        t_true = np.zeros((1, 3))
        t_est = (centroid - R_est[0] @ centroid)[None]  # a pure rotation about the centroid
        self.assertAlmostEqual(R.point_errors_mm(R_est, t_est, R_true, t_true, centroid)[0], 0.0, places=9)
        far = centroid + np.array([0.0, 0.3, 0.0])
        self.assertAlmostEqual(R.point_errors_mm(R_est, t_est, R_true, t_true, far)[0], 600.0 * np.sin(theta / 2), places=6)
        t = np.linspace(0.0, 1.0, 31)
        cubic = np.stack([1 + t - 2 * t ** 2 + 0.5 * t ** 3, -3 * t ** 3], axis=1)
        np.testing.assert_allclose(R.cubic_residuals(t, cubic), 0.0, atol=1e-9)

    def test_constant_twist_predicts_the_next_frame(self):
        errors = []
        indices = sorted(self.left)
        for i, j in zip(indices, indices[1:]):
            a, b = self.left[i], self.left[j]
            if a.episode != b.episode or j != i + 1:
                continue
            errors.append(np.linalg.norm(a.box_at(b.t - a.t)[:3, 3] - b.T_world_box[:3, 3]))
        # One whole 16.7 ms frame of extrapolation through hand acceleration; the
        # exposure uses +/-2 ms, where the same curvature costs ~1/70 of this.
        self.assertLess(np.percentile(errors, 95), 2.5e-3)

    def test_window_raster_matches_brute_force_intersection(self):
        renderer = S.SceneRenderer(2)
        truth = self.left[self.index]
        T = truth.box_at(0) @ S.mount_T_box_rig(self.scene, 'H', 'gripper', 90)
        for camera in self.scene.cameras[:3]:
            window = renderer.window(self.scene.models['H'], camera, [T])
            self.assertIsNotNone(window)
            Tcr = camera.T_cam_base @ T
            rays = renderer.window_rays(camera, window) @ Tcr[:3, :3]
            origin = -Tcr[:3, :3].T @ Tcr[:3, 3]
            fast, _ = renderer._raster_hits(camera, Tcr, window, origin, rays, self.scene.models['H'].occluder_triangles)
            slow, _ = S.nearest_hits(origin, rays, self.scene.models['H'].occluder_triangles)
            np.testing.assert_array_equal(np.isfinite(fast), np.isfinite(slow))
            np.testing.assert_allclose(fast[np.isfinite(fast)], slow[np.isfinite(slow)], rtol=0, atol=1e-12)

    def test_world_occluders_hide_the_target_from_the_real_decoder(self):
        renderer = S.SceneRenderer(2)
        truth = self.left[self.index]
        T = truth.box_at(0) @ S.mount_T_box_rig(self.scene, 'H', 'gripper', 90)
        camera = self.scene.cameras[1]
        clear, _, stats = renderer.render_camera(self.scene.models['H'], camera, [T], [(np.zeros((0, 3, 3)), [])],
                                                 self.scene.body_colour('H'), blur_sigma=0, noise_sigma=0,
                                                 rng=np.random.default_rng(0))
        _, ids, _ = self.detector.detectMarkers(cv2.cvtColor(clear, cv2.COLOR_BGR2GRAY))
        self.assertIsNotNone(ids)
        self.assertEqual(stats['occluded_px'], 0)
        mid = 0.5 * (camera.T_base_cam[:3, 3] + T[:3, 3])
        # Two triangles forming a 0.6 m square facing the camera, halfway along the line of sight.
        normal = (camera.T_base_cam[:3, 3] - T[:3, 3]) / np.linalg.norm(camera.T_base_cam[:3, 3] - T[:3, 3])
        u = np.cross(normal, [0, 0, 1.0])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        c = [mid + .3 * (su * u + sv * v) for su, sv in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        wall = np.array([[c[0], c[1], c[2]], [c[0], c[2], c[3]]])
        blocked, _, stats = renderer.render_camera(self.scene.models['H'], camera, [T], [(wall, [])],
                                                   self.scene.body_colour('H'), blur_sigma=0, noise_sigma=0,
                                                   rng=np.random.default_rng(0))
        _, ids, _ = self.detector.detectMarkers(cv2.cvtColor(blocked, cv2.COLOR_BGR2GRAY))
        self.assertIsNone(ids)
        self.assertAlmostEqual(stats['occluded_px'], stats['target_px'])

    def test_unit_rays_reproject_to_their_pixels(self):
        renderer = S.SceneRenderer(1)
        camera = self.scene.cameras[0]
        window = (900, 500, 940, 530)
        rays = renderer.window_rays(camera, window)
        uv = cv2.fisheye.projectPoints(rays[:, None, :], np.zeros(3), np.zeros(3), camera.K, camera.D)[0].reshape(-1, 2)
        yy, xx = np.mgrid[0:30, 0:40]
        np.testing.assert_allclose(uv, np.column_stack([xx.ravel() + 900, yy.ravel() + 500]), atol=1e-3)

    def test_bootstrap_blocks_do_not_depend_on_the_process_hash_seed(self):
        import inspect
        import l3_runner as R
        truth = self.left[self.index]
        self.assertEqual(R.time_block('left', truth), 1_000_000 + truth.episode * 10_000 + int(truth.t // 2.0))
        self.assertNotEqual(R.time_block('left', truth), R.time_block('right', truth))
        self.assertNotIn('hash(', inspect.getsource(R.load_level))

    def test_sessions_perturb_only_the_estimator_and_are_reproducible(self):
        import l3_runner as R
        R._init_worker({'supersample': 2, 'seed': 20260910, 'run': '/nonexistent'})
        self.assertIsNone(R.session_perturbation('nominal', 0))
        a, b = R.session_perturbation('evidence', 3), R.session_perturbation('evidence', 3)
        self.assertEqual(a['timing_ms'], b['timing_ms'])
        np.testing.assert_array_equal(a['cameras'][2][0], b['cameras'][2][0])
        before = [c.T_base_cam.copy() for c in R.SCENE.cameras]
        estimator = R.estimator_cameras(a)
        for camera, original in zip(R.SCENE.cameras, before):
            np.testing.assert_array_equal(camera.T_base_cam, original)
        angle = Rotation.from_matrix(estimator[0].T_base_cam[:3, :3].T @ R.SCENE.cameras[0].T_base_cam[:3, :3]).magnitude()
        self.assertGreater(np.degrees(angle), 0)
        nominal, perturbed = R._layout('H'), R._layout('H', a)
        self.assertGreater(np.max(np.abs(nominal.corners_for(30) - perturbed.corners_for(30))), 0)


if __name__ == '__main__':
    unittest.main()
