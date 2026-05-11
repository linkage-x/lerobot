.venv/bin/python3 third_party/opencv_kalibr/scripts/hikon_cube_tracking_in_robot_base.py
    --config third_party/opencv_kalibr/hikon_cube_tracking_offline/config_hikon/hikon_cube_tracking_in_robot_base_umi.yaml     
    --dataset-root /home/corenetic/Code/lerobot/data/single_cube2_20260429_164746

.venv/bin/python3 third_party/opencv_kalibr/hikon_cube_tracking_offline/interpolate_missing_ee_pose.py   
    --config third_party/opencv_kalibr/hikon_cube_tracking_offline/config_hikon/interpolate_missing_ee_pose.yaml   
    --dry-run