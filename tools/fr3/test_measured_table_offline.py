import unittest
import numpy as np
import coal
from check_measured_table_offline import distance_lower_bound, transit_values

class Tests(unittest.TestCase):
    def test_front_clearance(self):
        self.assertAlmostEqual(float(distance_lower_bound(np.array([0.,0.,0.]),np.array([.0716,.05,.1]),np.array([.1336,-3.,-3.]),np.array([3.,3.,.11]))),.062)
    def test_above_and_corner(self):
        b0=np.array([.1336,-3.,-3.]);b1=np.array([3.,3.,.11])
        self.assertAlmostEqual(float(distance_lower_bound(np.array([.2,0.,.21]),np.array([.3,.1,.3]),b0,b1)),.1)
        self.assertAlmostEqual(float(distance_lower_bound(np.array([0.,0.,.15]),np.array([.1036,.1,.2]),b0,b1)),.05)
    def test_collision_bound(self):
        self.assertEqual(float(distance_lower_bound(np.array([.1,0.,0.]),np.array([.2,.1,.2]),np.array([.1336,-3.,-3.]),np.array([3.,3.,.11]))),0.)
    def test_coal_distance_semantics(self):
        result=coal.DistanceResult()
        value=coal.distance(coal.Box(.1,.1,.1),coal.Transform3s(),coal.Box(.1,.1,.1),coal.Transform3s(np.eye(3),np.array([.2,0.,0.])),coal.DistanceRequest(),result)
        self.assertAlmostEqual(value,.1,places=7)
    def test_proposed_transit_endpoints_and_steps(self):
        q0=np.array([0.,-.4,0.,-2.4,0.,2.,.4]);q1=np.array([-.2,.3,.4,-2.4,.6,3.9,.2])
        t,q=transit_values(q0,q1)
        np.testing.assert_allclose(q[0],q0);np.testing.assert_allclose(q[-1],q1)
        self.assertLessEqual(np.diff(t).max(),.0200001)
        self.assertLessEqual(abs(np.diff(q,axis=0)).max(),.0040001)
    def test_virtual_wall_rejected(self):
        with self.assertRaises(ValueError): transit_values(np.zeros(7),np.zeros(7))

if __name__=='__main__': unittest.main()
