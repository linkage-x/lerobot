"""Narrow classification of operator-reviewed V2 distal pin contacts.

This is a mesh-contact diagnostic, not a solid penetration or stopping-distance
certificate. No entire collision pair is removed from the collision model.
"""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

MODEL_SHA256 = '2e10116497fac7ea8fe4f4217d9622a62f424ec32c1eb5aed088060e43e78535'
RADIUS_M = .0067
HALF_LENGTH_M = .0101
CONTACT_CAP = 10000


def inside_pin_region(points, *, capped=False):
    points = np.asarray(points, dtype=float)
    return bool(not capped and points.ndim == 2 and points.shape[1] == 3
                and len(points) > 0 and np.isfinite(points).all()
                and np.max(abs(points[:, 0])) <= HALF_LENGTH_M
                and np.max(np.linalg.norm(points[:, 1:], axis=1)) <= RADIUS_M)


class ReviewedHinges:
    def __init__(self, checker, urdf):
        if hashlib.sha256(Path(urdf).read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError('Hinge review is only valid for the exact reviewed V2 URDF')
        # The portable URDF embeds the content digest in every mesh filename.
        for record in checker.mesh_hashes.values():
            if Path(record['path']).stem != record['sha256']:
                raise ValueError('Reviewed mesh content changed')
        self.checker = checker
        tree = ET.parse(urdf)
        def xyz(name):
            return np.fromstring(tree.find(f"joint[@name='{name}']/origin").get('xyz'), sep=' ')
        self.pairs = {}
        for side in ('left', 'right'):
            finger = 'link_gripper_finger_' + side
            inner = 'link_gripper_knuckle_' + side + '_inner'
            self.pairs[frozenset((finger + '_0', inner + '_0'))] = (
                checker.model.getFrameId(finger), checker.model.getFrameId(inner),
                xyz('joint_gripper_knuckle_' + side + '_inner') - xyz('joint_gripper_knuckle_' + side + '_outer'),
                xyz('joint_gripper_finger_' + side))

    def classify(self, a, b):
        import coal
        c = self.checker
        key = frozenset((c.names[a], c.names[b]))
        if key not in self.pairs:
            return False, {'reason': 'unreviewed pair'}
        fi, ii, fp, ip = self.pairs[key]
        f, inner = c.data.oMf[fi], c.data.oMf[ii]
        closure = float(np.linalg.norm(f.act(fp) - inner.act(ip)))
        axis_error = float(np.linalg.norm(np.cross(f.rotation[:, 0], inner.rotation[:, 0])))
        if not np.isfinite([closure, axis_error]).all() or closure > 1e-7 or axis_error > 1e-7:
            return False, {'reason': 'hinge axis/closure mismatch'}
        request = coal.CollisionRequest()
        request.enable_contact = True
        request.num_max_contacts = CONTACT_CAP
        result = coal.CollisionResult()
        coal.collide(c.geom.geometryObjects[a].geometry, c.gd.oMg[a],
                     c.geom.geometryObjects[b].geometry, c.gd.oMg[b], request, result)
        contacts = list(result.getContacts())
        points = np.asarray([x.pos for x in contacts])
        local = (points - f.act(fp)) @ f.rotation if len(points) else np.empty((0, 3))
        accepted = inside_pin_region(local, capped=len(contacts) >= CONTACT_CAP)
        stats = dict(contacts=len(contacts), closure_error_m=closure, axis_error=axis_error,
                     accepted=accepted)
        if len(local) and np.isfinite(local).all():
            stats.update(axial_abs_max_m=float(np.max(abs(local[:, 0]))),
                         radial_max_m=float(np.max(np.linalg.norm(local[:, 1:], axis=1))))
        return accepted, stats

    @staticmethod
    def description():
        return dict(assembly_review='User confirmed the two V2 distal pin connections, 2026-09-08.',
                    model_sha256=MODEL_SHA256, radius_m=RADIUS_M, half_length_m=HALF_LENGTH_M,
                    scope='Only reported contacts inside these local pin cylinders; any outside, empty, nonfinite or capped result blocks.',
                    provenance='Envelope from existing mesh hinge audit, with small numeric allowance; not measured pin dimensions.',
                    whole_collision_pairs_removed=[])
