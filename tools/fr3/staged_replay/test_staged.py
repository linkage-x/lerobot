"""Software-only staging checks: no SDK, socket or robot construction."""
import json
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
import numpy as np

from box_transport import BoxTransport
from plans import samples,transit,validate
from runtime import execution_gate,require_motion_checks,sha,state_checks,save
from replay_timed_candidate import evaluate,FreshBoxMeasurement


class StagingTests(unittest.TestCase):
    def setUp(self):
        self.q=np.array([-.15,-.41,.067,-2.4,.045,2.06,.44])

    def test_transit_exact_endpoints_and_rest(self):
        target=self.q+np.array([.1,.4,-.04,.2,-.03,1.,.1])
        ep=transit(self.q,.088,target,.032)
        np.testing.assert_allclose(evaluate(ep,0.),np.r_[self.q,.088],atol=1e-12)
        np.testing.assert_allclose(evaluate(ep,ep['duration_s']),np.r_[target,.032],atol=1e-12)
        for d in [1,2]:
            for t in [0.,ep['duration_s']]:
                np.testing.assert_allclose(evaluate(ep,t,d),np.zeros(8),atol=1e-10)

    def test_dense_joint_and_width_steps(self):
        ep=transit(self.q,.088,self.q+.03,.01)
        points=list(samples(ep));ts=np.array([s[0] for s in points])
        values=np.array([np.r_[s[1],s[2]] for s in points])
        self.assertLessEqual(np.max(np.diff(ts)),.02000001)
        self.assertLessEqual(np.max(abs(np.diff(values[:,:7],axis=0))),.00400001)
        self.assertLessEqual(np.max(abs(np.diff(values[:,7]))),.000250001)

    def test_stationary_transit(self):
        ep=transit(self.q,.04,self.q,.04)
        validate(ep)
        np.testing.assert_allclose(evaluate(ep,.5),np.r_[self.q,.04])

    def test_bad_width_and_nan_rejected(self):
        for width in [-.01,.12,float('nan')]:
            with self.assertRaises(ValueError):transit(self.q,width,self.q,.04)

    def test_bad_limits_rejected(self):
        q=self.q.copy();q[5]=4.5
        with self.assertRaises(ValueError):transit(self.q,.04,q,.04)

    def test_readonly_transport_cannot_send_or_reset(self):
        t=BoxTransport('/nonexistent',123,'127.0.0.1')
        native=Mock();t.box=native;t.mode=1
        with self.assertRaises(RuntimeError):t.send(.04)
        native.set_clamp_pos.assert_not_called()
        t.close()
        native.stop.assert_called_once();native.close.assert_called_once()
        native.set_mode.assert_not_called();native.set_trigger_zero.assert_not_called()

    def test_explicit_identity_and_ack(self):
        t=BoxTransport('/nonexistent',123,'127.0.0.1',allow_commands=True)
        t.box=Mock();t.mode=0
        with self.assertRaises(RuntimeError):t.send(.04)
        t.mode=1;t.box.set_clamp_pos.return_value=0
        self.assertTrue(t.send(.04));t.box.set_clamp_pos.assert_called_once_with(.04,123)
        t.box.set_clamp_pos.return_value=4
        self.assertFalse(t.send(.04))

    def test_dhcp_lookup_keeps_exact_identity_without_mode_writes(self):
        @dataclass
        class Device:
            device_id:int
            ip:str
            data_port:int=15000
            capabilities:int=1
            sn:str='test-device'
            device_type:int=1
            fw_version:int=770
            proto_ver:int=1
            msg_type:int=2
            uptime_ms:int=1000
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'python').mkdir()
            (root/'python/box_collection_sdk-test.whl').touch()
            sdk=Mock();native=sdk.Box.return_value
            sdk.discover.side_effect=[[Device(999,'192.168.9.9')],[Device(123,'192.168.9.5')]]
            native.start.return_value=0;native.register_device.return_value=0
            native.get_mode.return_value=(0,0)
            with patch('box_transport.socket.socket'),patch('box_transport.importlib.import_module',return_value=sdk),patch.dict(os.environ):
                t=BoxTransport(root,123,'192.168.2.60').open()
                self.assertEqual(t.ip,'192.168.9.5')
                native.register_device.assert_called_once_with(123,'192.168.9.5',15000)
                native.set_mode.assert_not_called();native.set_clamp_pos.assert_not_called()
                t.close()

    def test_missing_identity_never_opens_control_sdk(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'python').mkdir()
            (root/'python/box_collection_sdk-test.whl').touch()
            sdk=Mock();sdk.discover.return_value=[]
            with patch('box_transport.socket.socket'),patch('box_transport.importlib.import_module',return_value=sdk),patch.dict(os.environ):
                with self.assertRaisesRegex(RuntimeError,'not uniquely discovered'):
                    BoxTransport(root,123,'192.168.2.60').open()
                sdk.Box.assert_not_called()

    def test_cached_measurement_does_not_count_as_new(self):
        d=FreshBoxMeasurement(123)
        s=dict(status=dict(active=True,device_id=123),sensors=dict(box_gripper=dict(timestamp=12,distance_m=.04)))
        self.assertIsNone(d.decode(s));self.assertIsNone(d.decode(s))
        s['sensors']['box_gripper']['timestamp']=13
        self.assertEqual(d.decode(s),(1,.04))

    def test_mode_reply_before_first_sensor_waits_but_reset_rejected(self):
        from types import SimpleNamespace as N
        t=BoxTransport('/nonexistent',123,'127.0.0.1')
        cache=N(valid=True,device_id=123,data=N(gripper_data=N(timestamp=0,distance=.04)))
        t.box=Mock();t.box.get_sensor_cache.return_value=(0,cache)
        self.assertIsNone(t.fresh())
        cache.data.gripper_data.timestamp=100
        self.assertIsNone(t.fresh())
        cache.data.gripper_data.timestamp=101
        self.assertEqual(t.fresh(),(1,.04))
        cache.data.gripper_data.timestamp=0
        with self.assertRaises(ValueError):t.fresh()

    def test_collision_cannot_be_overridden_by_watcher(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'manifest.json').write_text('{}')
            p=root/'p.json';p.write_text('{}')
            data=dict(bundle_manifest_sha256=sha(root/'manifest.json'),geometry=dict(sampled_geometry_pass=False))
            with patch.dict(os.environ,FRANKA_PHYSICAL_WATCHER_CONFIRMED='YES'):
                with self.assertRaisesRegex(RuntimeError,'Geometry gate closed'):
                    execution_gate(root,p,data,sha(p))

    def test_confirmation_is_exact_plan_not_generic_yes(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'p.json';p.write_text('{}')
            with patch.dict(os.environ,FRANKA_PHYSICAL_WATCHER_CONFIRMED='YES'):
                with self.assertRaisesRegex(RuntimeError,'plan-hash'):
                    execution_gate(Path(temp),p,{},'YES')

    def test_wrong_start_state_blocks(self):
        ep=transit(self.q,.04,self.q+.01,.05)
        state=dict(q=(self.q+.1).tolist(),current_errors='[]')
        checks=dict(idle=True,at_rest=True)
        with self.assertRaisesRegex(RuntimeError,'State changed'):
            require_motion_checks(state,checks,dict(execution_plan=ep),.04)
        checks['idle']=False
        with self.assertRaisesRegex(RuntimeError,'Idle'):
            require_motion_checks(state,checks,dict(execution_plan=ep),.04)

    def test_serialization_failure_does_not_leave_partial_report(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'report.json'
            with self.assertRaises(TypeError):save(p,dict(value=np.bool_(True)))
            self.assertFalse(p.exists())


if __name__=='__main__':unittest.main(verbosity=2)
