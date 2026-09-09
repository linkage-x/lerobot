"""Project candidate base and reconstructed TCP onto existing source frames.

These are same-camera consistency overlays, NOT independent accuracy tests.
"""
import csv
import json
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
INPUT = HERE.parent / 'recalculation_20260907_193725/coordinate_chain_audit_20260907/inputs'

if __name__ == '__main__':
    snapshot = json.loads((INPUT / 'calibration_snapshot.json').read_text())
    intr = snapshot['per_camera_intrinsics']['cam_13']['data']
    K, D = np.asarray(intr['camera_matrix']), np.asarray(intr['dist_coeffs']).reshape(4,1)
    C_W = np.linalg.inv(np.asarray(snapshot['extrinsics_0902']['data']['joint_solution']['cameras']['cam_13']['base_to_camera']['matrix_4x4']))
    W_B = np.asarray(json.loads((HERE/'preparation_manifest.json').read_text())['base_matrix'])
    with (HERE/'contact_ik.right.csv').open() as f:
        rows = list(csv.DictReader(f))
    def pixel(world):
        pc = C_W[:3,:3] @ np.asarray(world) + C_W[:3,3]
        if pc[2] <= 0:
            raise ValueError('Point behind camera')
        uv,_=cv2.fisheye.projectPoints(pc.reshape(1,1,3),np.zeros(3),np.zeros(3),K,D)
        return tuple(np.round(uv[0,0]).astype(int))
    for ep in [0,1]:
        path = HERE/f'episode_{ep}_source_frame0_overlay.jpg'
        if path.exists():
            raise FileExistsError(path)
        capture = cv2.VideoCapture(str(INPUT/f'replay_ep{ep}_cam_13.mkv'))
        ok,frame = capture.read()
        capture.release()
        assert ok and frame.shape[:2] == (1080,1920)
        row = next(r for r in rows if int(r['episode_index'])==ep and int(r['frame_index'])==0)
        p = pixel([float(row[f'contact_target_{a}_m']) for a in 'xyz'])
        cv2.drawMarker(frame,p,(0,180,255),cv2.MARKER_CROSS,30,3)
        cv2.putText(frame,'reconstructed contact TCP',(p[0]+18,p[1]-18),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,180,255),2)
        origin = pixel(W_B[:3,3])
        for axis,color in enumerate([(0,0,255),(0,255,0),(255,120,0)]):
            endpoint = pixel(W_B[:3,3]+W_B[:3,axis]*.10)
            cv2.arrowedLine(frame,origin,endpoint,color,3,tipLength=.12)
            cv2.putText(frame,'XYZ'[axis],endpoint,cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
        cv2.putText(frame,'candidate BASE',(origin[0]+15,origin[1]+28),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,0),2)
        cv2.rectangle(frame,(0,0),(1920,70),(25,25,25),-1)
        cv2.putText(frame,f'Episode {ep}, original cam_13 frame 0 | corrected P0 + 0902 extrinsics | axes 100 mm',(20,27),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
        cv2.putText(frame,'Consistency overlay only: the same camera/calibration contributes to reconstruction, not independent proof.',(20,55),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,205,255),1)
        assert cv2.imwrite(str(path),frame)
        print(path)
