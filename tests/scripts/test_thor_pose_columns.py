from __future__ import annotations

from tools.thor.gmsl2 import pose_columns as pc


def test_new_exports_write_world_and_readers_take_old_base_columns_too():
    new = {"observation.ee_pose.right.world": {}, "action.ee_pose.right.world": {}}
    old = {"observation.ee_pose.right.base": {}, "observation.cube_pose.right.base": {}}

    assert pc.world_key(pc.EE_STATE, "right") == "observation.ee_pose.right.world"
    assert pc.pose_key(new, pc.EE_ACTION, "right") == "action.ee_pose.right.world"
    assert pc.pose_key(old, pc.EE_STATE, "right") == "observation.ee_pose.right.base"
    assert pc.pose_key(old, pc.CUBE, "right") == "observation.cube_pose.right.base"
    assert pc.pose_key(old, pc.EE_ACTION, "right") is None
    assert pc.pose_key(new, pc.EE_STATE, "left") is None


def test_tracked_hands_ignores_camera_frame_columns():
    columns = [
        "observation.ee_pose.right.world",
        "observation.ee_pose.left.base",
        "observation.cube_pose.right.camera.cam_03",
        "observation.ee_pose.right.camera.cam_03",
        "observation.state",
    ]
    assert pc.tracked_hands(columns) == ["left", "right"]
