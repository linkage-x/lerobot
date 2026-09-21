import sys
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from l2_runner import (RayRenderer, CameraCalibration, HybridCarrierModel,
    marker_layout_as_carrier, intersect_triangle, joint_solve, project_rig,
    build_aruco_detector)


class ImageOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        cls.camera=CameraCalibration('test','synthetic',640,480,
            np.array([[380.,0,320],[0,380,240],[0,0,1]]),np.array([-.012,.0018,-.00022,.00001]),np.eye(4),model='fisheye')
        cls.points=np.array([[-.03,-.03,0],[.03,-.03,0],[.03,.03,0],[-.03,.03,0]])
        layout={'units':'m','markers':[{'id':30,'corners_rig':cls.points.tolist()}]}
        cls.model=HybridCarrierModel.from_dict(marker_layout_as_carrier(layout,carrier_id='oracle',pad_size_mm=74))
        cls.detector=build_aruco_detector('DICT_6X6_50')

    def test_ray_intersection_has_known_metric_depth(self):
        tri=np.array([[-1,-1,2],[1,-1,2],[0,1,2.]])
        got=intersect_triangle(np.zeros(3),np.array([[0,0,1],[2,0,1.]]),tri)
        self.assertEqual(got[0],2.)
        self.assertTrue(np.isinf(got[1]))

    def test_real_decoder_matches_independent_projection_for_rotated_stickers(self):
        # Native decoding provides the independent corner numbering oracle.
        for quadrant in [0,2,3]:
            model=replace(self.model,anchors=[replace(self.model.anchors[0],paste_quadrant=quadrant)])
            for offset in [0.,.24]:
                T=np.eye(4);T[:3,3]=[offset,.03,.55]
                image=RayRenderer(3).render(model,self.camera,T)
                corners,ids,_=self.detector.detectMarkers(image)
                self.assertIsNotNone(ids)
                index=list(ids.ravel()).index(30)
                expected=project_rig(self.camera,T,np.roll(self.points,-quadrant,axis=0))
                self.assertLess(float(np.max(np.linalg.norm(corners[index].reshape(4,2)-expected,axis=1))),.9)

    def test_empty_detection_never_uses_truth_or_manufactures_pose(self):
        fit,reason=joint_solve([],[])
        self.assertIsNone(fit)
        self.assertEqual(reason,'insufficient_multicamera_anchors')

    def test_foreground_sphere_hides_sticker_from_actual_decoder(self):
        T=np.eye(4);T[:3,3]=[0,0,.55]
        image=RayRenderer(2).render(self.model,self.camera,T,[{'c':[0,0,.4],'r':.07}])
        _,ids,_=self.detector.detectMarkers(image)
        self.assertIsNone(ids)


if __name__=='__main__':
    unittest.main()
