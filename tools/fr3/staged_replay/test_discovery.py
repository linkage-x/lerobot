"""Regression for the actual duplicate discovery reports; no hardware access."""
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from box_transport import BoxTransport, select_physical_device


@dataclass
class Device:
    device_id: int = 596523097
    ip: str = '192.168.1.119'
    sn: str = 'box_collection_controller_000001'
    device_type: int = 1
    fw_version: int = 770
    data_port: int = 15000
    capabilities: int = 31
    uptime_ms: int = 16237500
    msg_type: int = 3
    proto_ver: int = 1
    capability_names: list = field(default_factory=lambda: ['GRIPPER', 'TRIGGER', 'TOUCH', '6D_FORCE', 'IMU'])


class DiscoveryTests(unittest.TestCase):
    def test_actual_two_message_types_are_one_device(self):
        first = Device()
        second = replace(first, msg_type=2)
        for reports in ([first, second], [second, first], [first, second, first]):
            selected, info = select_physical_device(reports, first.device_id)
            self.assertEqual(selected.ip, '192.168.1.119')
            self.assertEqual(selected.msg_type, 2)
            self.assertEqual(info['unique_devices'], 1)
            self.assertEqual(info['duplicate_messages_merged'], len(reports)-1)
            self.assertEqual(info['message_types'], [2, 3])

    def test_differing_uptime_does_not_invent_second_device(self):
        d = Device()
        result, _ = select_physical_device([d, replace(d, uptime_ms=d.uptime_ms+100, msg_type=2)], d.device_id)
        self.assertEqual(result.msg_type, 2)

    def test_real_conflicts_remain_blocked(self):
        d = Device()
        for change in [dict(ip='192.168.2.134'), dict(data_port=15001), dict(sn='different'),
                       dict(device_type=2), dict(fw_version=771), dict(capabilities=1), dict(proto_ver=2)]:
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                select_physical_device([d, replace(d, **change)], d.device_id)

    def test_no_expected_id_and_bad_endpoint_blocked(self):
        d = Device()
        with self.assertRaises(RuntimeError): select_physical_device([replace(d, device_id=123)], d.device_id)
        for address in ['not-an-ip', '0.0.0.0', '127.0.0.1', '224.0.0.1']:
            with self.assertRaises(RuntimeError): select_physical_device([replace(d, ip=address)], d.device_id)
        with self.assertRaises(RuntimeError): select_physical_device([replace(d, capabilities=0)], d.device_id)

    def test_other_boxes_are_not_selected(self):
        d = Device()
        selected, info = select_physical_device([replace(d, device_id=1819152274, ip='192.168.216.196'), d], d.device_id)
        self.assertEqual(selected.device_id, d.device_id)
        self.assertEqual(info['raw_matching_messages'], 1)

    def test_dhcp_fallback_accepts_duplicate_messages_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'python').mkdir()
            (root/'python/box_collection_sdk-test.whl').touch()
            sdk = Mock(); native = sdk.Box.return_value
            d = Device()
            sdk.discover.side_effect = [[], [d, replace(d, msg_type=2)]]
            native.start.return_value = 0; native.register_device.return_value = 0
            native.get_mode.return_value = (0, 1)
            with patch('box_transport.socket.socket'), patch('box_transport.importlib.import_module', return_value=sdk), patch.dict(os.environ):
                box = BoxTransport(root, d.device_id, '192.168.2.134').open()
                self.assertEqual(box.ip, '192.168.1.119')
                native.register_device.assert_called_once_with(d.device_id, '192.168.1.119', 15000)
                self.assertEqual(box.discovery_resolution['duplicate_messages_merged'], 1)
                box.close()
                native.set_mode.assert_not_called(); native.set_clamp_pos.assert_not_called()

    def test_conflicting_addresses_never_create_device_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'python').mkdir()
            (root/'python/box_collection_sdk-test.whl').touch()
            sdk = Mock(); d = Device()
            sdk.discover.return_value = [d, replace(d, ip='192.168.2.134')]
            with patch('box_transport.socket.socket'), patch('box_transport.importlib.import_module', return_value=sdk), patch.dict(os.environ):
                with self.assertRaises(RuntimeError): BoxTransport(root, d.device_id, d.ip).open()
                sdk.Box.assert_not_called()


if __name__ == '__main__': unittest.main(verbosity=2)
