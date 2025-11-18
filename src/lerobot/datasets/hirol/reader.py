import os
import json
import cv2
import numpy as np
import enum, copy
from scipy.spatial.transform import Rotation as R
os.environ["RUST_LOG"] = "error"
import glog as log

class ObservationType(enum.Enum):
    JointPosition = "q"
    DeltaJointPosition = "dq"
    EEPose = "ee"
    DeltaEEPose = "dee"
    JP_EEPose = "q_ee"
    Mask = "mask"
    FT = "ft"

class ActionType(enum.Enum):
    JointPosition = "q"
    DeltaJointPosition = "dq"
    EEPose = "ee"
    DeltaEEPose = "dee"
    CommandJointPosition = "cq"
    CommandDeltaJointPosition = "cdq"
    CommandEEPose = "cee"
    CommandDeltaEEPose = "cdee"

class RerunEpisodeReader:
    def __init__(
        self,
        task_dir=".",
        json_file="data.json",
        action_type: ActionType = ActionType.JointPosition,
        action_prediction_step=2,
        action_ori_type="euler",
        observation_type: ObservationType = ObservationType.JointPosition,
        rotation_transform=None,
        contain_ft: bool = False,
    ):
        self.task_dir = task_dir
        self.json_file = json_file
        self.action_type = action_type
        self._obs_type = observation_type
        self._action_prediction_step = action_prediction_step
        self._action_ori_type = action_ori_type
        # None or dict[str, np.ndarray]
        self._rotation_transform = rotation_transform
        # Whether force/torque readings are present in ee_states and should be used
        self._contain_ft = contain_ft

    def return_episode_data(self, episode_idx, skip_steps_nums=1):
        # Load episode data on-demand
        episode_dir = os.path.join(self.task_dir, f"episode_{episode_idx:04d}")
        json_path = os.path.join(episode_dir, self.json_file)

        if not os.path.exists(json_path):
            log.warn(f"Episode {episode_idx} data.json not found.")
            return None

        with open(json_path, 'r', encoding='utf-8') as jsonf:
            json_file = json.load(jsonf)

        episode_data = []

        # Loop over the data entries and process each one
        counter = 0
        skip_steps_nums = int(skip_steps_nums)
        if skip_steps_nums > self._action_prediction_step:
            self._action_prediction_step = skip_steps_nums
        len_json_file = len(json_file['data'])
        json_data = json_file['data']
        init_ee_poses = {}
        # @TODO: maybe pose-process for time synchronization
        for i, item_data in enumerate(json_file['data']):
            # Early skip by index before any heavy image decoding
            if counter % skip_steps_nums != 0:
                counter += 1
                continue
            # Process images and other data (only for kept indices)
            colors, colors_time_stamp = self._process_images(item_data, 'colors', episode_dir)
            if colors is None or len(colors) == 0:
                log.warn(f'Do not get the {i}th color image from {self.task_dir} {episode_dir}, color is None {colors}')
                counter += 1
                continue
            depths, depths_time_stamp = self._process_images(item_data, 'depths', episode_dir)
            if depths is None:
                counter += 1
                continue
            audios = self._process_audio(item_data, 'audios', episode_dir)
            
            # Append the observation state data in the item_data list
            joint_states = item_data.get("joint_states", {})
            joint_check = [ObservationType.JointPosition, ObservationType.JP_EEPose]
            if self._obs_type in joint_check:
                if joint_states is None or len(joint_states) == 0:
                    raise ValueError(f'Do not get the {i}th joint state from {self.task_dir} {episode_dir} for {self._obs_type}')
            ee_states = item_data.get('ee_states', {})
            if len(init_ee_poses) != len(ee_states):
                for key, cur_ee_state in ee_states.items():
                    if self._rotation_transform:
                        pose = cur_ee_state["pose"]
                        init_ee_poses[key] = self.apply_rotation_offset(pose, key)      
                        log.info(f'Successfully updated the init ee pose for relative pose calculation {list(init_ee_poses.keys())}')
                    else: init_ee_poses[key] = None
            # @TODO: used for latter head tracker
            head_pose = ee_states.pop('head', None)
            ee_check = [ObservationType.JP_EEPose, ObservationType.EEPose,
                        ObservationType.DeltaEEPose] 
            if self._obs_type in ee_check:
                if ee_states is None or len(ee_states) == 0:
                    raise ValueError(f'Do not get the {i}th ee state pose from {self.task_dir} {episode_dir} for {self._obs_type}')

            # Validate FT signals if requested
            if self._obs_type == ObservationType.FT or self._contain_ft:
                for key, state in ee_states.items():
                    if "ft" not in state:
                        raise ValueError(
                            f"ee state '{key}' does not contain 'ft' field "
                            f"while obs_type={self._obs_type} and contain_ft={self._contain_ft}"
                        )

            # Append the observation state data in the item_data list
            cur_obs = {}
            joint_states = item_data.get("joint_states", {})
            if self._obs_type in (ObservationType.JointPosition, ObservationType.JP_EEPose):
                if joint_states is None or len(joint_states) == 0:
                    raise ValueError(f'Do not get the {i}th joint state from {self.task_dir} {episode_dir} for {self._obs_type}')
            ee_states = item_data.get('ee_states', {})
            if len(init_ee_poses) != len(ee_states):
                for key, cur_ee_state in ee_states.items():
                    if self._rotation_transform:
                        pose = cur_ee_state["pose"]
                        init_ee_poses[key] = self.apply_rotation_offset(pose, key)
                        log.info(f'Successfully updated the init ee pose for relative pose calculation {list(init_ee_poses.keys())}')
                    else: init_ee_poses[key] = None
            # @TODO: used for latter head tracker
            head_pose = ee_states.pop('head', None)
            if self._obs_type in (ObservationType.JP_EEPose, ObservationType.EEPose):
                if ee_states is None or len(ee_states) == 0:
                    raise ValueError(f'Do not get the {i}th ee state pose from {self.task_dir} {episode_dir} for {self._obs_type}')
            
            if self._obs_type in (ObservationType.JointPosition, ObservationType.JP_EEPose):
                for key in joint_states.keys():
                    cur_obs[key] = np.array(joint_states[key]["position"])
                    if self._obs_type == ObservationType.JP_EEPose:
                        ee_pose = self.apply_rotation_offset(ee_states[key]["pose"], key,
                                                            init_data=init_ee_poses[key])
                        cur_obs[key] = np.hstack((cur_obs[key], ee_pose))
            elif self._obs_type in (ObservationType.EEPose, ObservationType.DeltaEEPose):
                next_id = i + 1
                if next_id >= len_json_file: continue
                next_ee_states = json_data[next_id].get("ee_states", {})
                for key in ee_states.keys():
                    ee_pose = self.apply_rotation_offset(ee_states[key]["pose"], key,
                                                        init_data=init_ee_poses[key])
                    if self._obs_type == ObservationType.EEPose:
                        cur_obs[key] = np.array(ee_pose)
                    else:
                        next_pose = next_ee_states[key]["pose"]
                        # @TODO: think about it
                        next_pose = self.apply_rotation_offset(next_pose, key,
                                                init_data=init_ee_poses[key])
                        cur_obs[key] = self.get_pose_diff(next_pose, ee_pose)
            elif self._obs_type == ObservationType.Mask:
                for key in ee_states.keys():
                    cur_obs[key] = np.zeros(7)
            elif self._obs_type == ObservationType.FT:
                # Pure FT observation: use force/torque vector per ee
                for key, state in ee_states.items():
                    ft_vals = state.get("ft", None)
                    if ft_vals is None:
                        raise ValueError(
                            f"ee state '{key}' missing 'ft' data for ObservationType.FT"
                        )
                    cur_obs[key] = np.asarray(ft_vals, dtype=np.float32)

            # Optionally append FT to existing observations when contain_ft is enabled
            if self._contain_ft and self._obs_type != ObservationType.FT:
                for key, state in ee_states.items():
                    ft_vals = state.get("ft", None)
                    if ft_vals is None:
                        raise ValueError(
                            f"ee state '{key}' missing 'ft' data while contain_ft=True"
                        )
                    ft_arr = np.asarray(ft_vals, dtype=np.float32)
                    if key in cur_obs:
                        cur_obs[key] = np.hstack((cur_obs[key], ft_arr))
                    else:
                        cur_obs[key] = ft_arr
            
            # Append the action data in the item_data list
            cur_actions = {}
            action_state_id = i+self._action_prediction_step
            if action_state_id >= len_json_file: continue
            if self.action_type == ActionType.JointPosition:
                joint_states = item_data.get("joint_states", {})
                cur_actions = self._get_absolute_action(joint_states, 
                        action_state=json_data[action_state_id]["joint_states"],
                                                    attribute_name="position")
            elif self.action_type == ActionType.EEPose:
                init_data = None if len(init_ee_poses) == 0 else init_ee_poses
                cur_actions = self._get_absolute_action(item_data.get("ee_states", {}),
                                    action_state=json_data[action_state_id]["ee_states"],
                                    attribute_name="pose", init_data=init_data)
                if self._action_ori_type == 'euler':
                    modified_action = {}
                    for key, action in cur_actions.items():
                        modified_action[key] = np.zeros(6)
                        modified_action[key][:3] = action[:3]
                        modified_action[key][3:] = R.from_quat(action[3:]).as_euler("xyz", False)
                    cur_actions = modified_action
                elif self._action_ori_type != "quaternion":
                    raise ValueError(f'The action orientation type {self._action_ori_type} is not supported for reading episode data')
            elif self.action_type == ActionType.DeltaJointPosition:
                joint_states = item_data.get("joint_states", {})
                next_state_data = json_data[action_state_id].get("joint_states", {})
                cur_actions = self._get_delta_action(joint_states, next_state_data, "position")
            elif self.action_type == ActionType.DeltaEEPose:
                ee_states = item_data.get("ee_states", {})
                next_state_data = json_data[action_state_id].get("ee_states", {})
                for key, pose in ee_states.items():
                    cur_actions[key] = np.zeros(7)
                    next_pose = np.array(next_state_data[key]["pose"])
                    next_pose = self.apply_rotation_offset(next_pose, key, init_ee_poses[key])
                    cur_pose = self.apply_rotation_offset(np.array(pose["pose"]), key, init_ee_poses[key])
                    cur_actions[key] = self.get_pose_diff(next_pose, cur_pose)
                if self._action_ori_type == "euler":
                    modified_action = {}
                    for key, action in cur_actions.items():
                        modified_action[key] = np.zeros(6)
                        modified_action[key][:3] = action[:3]
                        modified_action[key][3:] = R.from_quat(action[3:]).as_euler("xyz")
                    cur_actions = modified_action
                elif self._action_ori_type != "quaternion":
                    raise ValueError(f'The action orientation type {self._action_ori_type} is not supported for reading episode data')
            else:
                raise ValueError(f'The action type {self.action_type} is not supported for reading episode data')
            # tool state
            tool_states = item_data.get("tools", {})
            for key, tool_state in tool_states.items():
                cur_actions[key] = np.hstack((cur_actions[key], tool_state["position"]))
                if self._obs_type == ObservationType.Mask:
                    cur_obs[key] = np.hstack((cur_obs[key], [0]))
                else:
                    cur_obs[key] = np.hstack((cur_obs[key], tool_state["position"]))
            
            episode_data.append(
                {
                    'idx': item_data.get('idx', 0),
                    'colors': colors,
                    'colors_time_stamp': colors_time_stamp,
                    'depths': depths,
                    'depths_time_stamp': depths_time_stamp,
                    'joint_states': item_data.get('joint_states', {}),
                    'ee_states': item_data.get('ee_states', {}),
                    'tools': item_data.get('tools', {}),
                    'imus': item_data.get('imus', {}),
                    'tactiles': item_data.get('tactiles', {}),
                    'audios': audios,
                    'actions': cur_actions,
                    'observations': cur_obs
                }
            )
            counter += 1
        
        return episode_data
    
    def get_episode_text_info(self, episode_id):
        episode_dir = os.path.join(self.task_dir, f"episode_{episode_id:04d}")
        json_path = os.path.join(episode_dir, self.json_file)

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Episode {episode_id} data.json not found.")

        with open(json_path, 'r', encoding='utf-8') as jsonf:
            json_file = json.load(jsonf)

        text_info = json_file["text"]
        if not "steps" in text_info:
            return None
        
        steps = ""
        if isinstance(text_info["steps"], dict):
            for step_number, cur_step in text_info["steps"].items():
                steps += cur_step
                steps += " "
        else: steps = text_info["steps"]
        
        text_info = 'description: ' + text_info["desc"] + ' ' \
                + 'steps: ' + steps + ' ' + 'goal: ' + text_info["goal"]        
        return text_info
    
    def _get_absolute_action(self, states, action_state, attribute_name = None, init_data = None):
        cur_action = {}
        for key, state in states.items():
            if attribute_name is not None:
                if attribute_name == "pose":
                    cur_init_pose = init_data[key] if init_data else None
                    action_state[key][attribute_name] = self.apply_rotation_offset(
                                action_state[key][attribute_name], key, cur_init_pose)
                cur_action[key] = action_state[key][attribute_name]
            else:
                cur_action[key] = action_state[key]
        return cur_action
    
    def _get_delta_action(self, states, next_state_data, attribute_name = None):
        cur_action = {}
        next_state_value = {}
        for key, state in next_state_data.items():
            state_value = state if attribute_name is None else state[attribute_name]
            next_state_value[key] = state_value
        
        for key, state in states.items():
            state_value = state if attribute_name is None else state[attribute_name]
            cur_action[key] = np.array(next_state_value[key]) - np.array(state_value)
        return cur_action
    
    def get_pose_diff(self, pose1, pose2, posi_translation=True):
        """ pose1 - pose2"""
        pose_diff = np.zeros(7)
        
        rot1 = R.from_quat(pose1[3:])
        rot2 = R.from_quat(pose2[3:])
        rot2_trans = rot2.inv()
        rot = rot2_trans * rot1
        posi_diff = np.array(pose1[:3]) - np.array(pose2[:3])
        if posi_translation:
            pose_diff[:3] = rot2_trans.apply(posi_diff)
        else: pose_diff[:3] = posi_diff
        pose_diff[3:] = rot.as_quat()
        return pose_diff
    
    def convert_quat_to_euler_pose(self, all_ee_states):
        all_ee_states_euler = {}
        # @TODO: attribute name "pose"
        for key, state in all_ee_states.items():
            all_ee_states_euler[key] = np.zeros(6)
            all_ee_states_euler[key][:3] = state["pose"][:3]
            all_ee_states_euler[key][3:] = R.from_quat(state["pose"][3:]).as_euler('xyz', degrees=False)
        return all_ee_states_euler
    
    def transform_quat(self, quat1, quat2):
        rot_ab = R.from_quat(quat1)
        rot_bc = R.from_quat(quat2)
        rot_ac = rot_ab * rot_bc  # R_ac = R_ab * R_bc
        return rot_ac.as_quat()  # [qx, qy, qz, qw]
    
    def apply_rotation_offset(self, pose, key, init_data = None):
        new_pose = copy.deepcopy(pose)
        if self._rotation_transform is not None:
            if key not in self._rotation_transform:
                raise ValueError(f'Got the rotation transform but {key} not found in {self._rotation_transform}')
            new_pose[3:] = self.transform_quat(pose[3:], self._rotation_transform[key])
            if init_data:
                # calculate relative term
                new_pose = self.get_pose_diff(new_pose, init_data)
        return new_pose
        
    def _process_images(self, item_data, data_type, dir_path):
        """
        Load images for a given data_type ("colors" or "depths").
        - Colors: read BGR, convert to RGB.
        - Depths: read unchanged; expand to (H, W, 1) if needed.
        Returns (images_dict, timestamp_dict) or (None, None) on failure.
        """
        images = item_data.get(data_type, {})
        time_stamp = {}
        if images is None:
            return {}, {}

        out = {}
        for key, data in images.items():
            file_name = data.get("path") if isinstance(data, dict) else None
            if not file_name:
                return None, None
            file_path = os.path.join(dir_path, file_name)
            if not os.path.exists(file_path):
                return None, None
            if data_type == 'colors':
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if img is None:
                    return None, None
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:  # depths
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    return None, None
                # Ensure (H, W, 1) for single-channel depths to match (H,W,C) expectation
                if img.ndim == 2:
                    img = img[:, :, None]
            out[key] = img
            time_stamp[key] = data.get("time_stamp") if isinstance(data, dict) else None

        return out, time_stamp

    def _process_audio(self, item_data, data_type, episode_dir):
        audio_item = item_data.get(data_type, {})
        if audio_item is None:
            return {}
        
        audio_data = {}
        dir_path = os.path.join(episode_dir, data_type)

        for key, file_name in audio_item.items():
            if file_name:
                file_path = os.path.join(dir_path, file_name)
                if os.path.exists(file_path):
                    pass  # Handle audio data if needed
        return audio_data

if __name__ == "__main__":
    # episode_reader = RerunEpisodeReader(task_dir = unzip_file_output_dir)
    # # TEST EXAMPLE 1 : OFFLINE DATA TEST
    # episode_data6 = episode_reader.return_episode_data(6)
    # logger_mp.info("Starting offline visualization...")
    # offline_logger = RerunLogger(prefix="offline/")
    # offline_logger.log_episode_data(episode_data6)
    # logger_mp.info("Offline visualization completed.")
    
    data_folder = "dataset/data/test_now"
    cur_path = os.path.dirname(os.path.abspath(__file__))
    task_dir = os.path.join(cur_path, '../..', data_folder)
    episode_reader = RerunEpisodeReader(task_dir=task_dir, action_type=ActionType.DeltaJointPosition)
    data = episode_reader.return_episode_data(2, 1)
    print(f'data: {data}')
