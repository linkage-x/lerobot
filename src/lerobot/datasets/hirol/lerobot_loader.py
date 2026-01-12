from .reader import ActionType, ObservationType
from .data_loader_base import DataLoaderBase
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os
import numpy as np
from tqdm import tqdm
import glog as log
from PIL import Image as PILImage

class LerobotLoader(DataLoaderBase):
    def __init__(self, config, task_dir, json_file_name = "data.json", action_type = ActionType.JointPosition,
                 observation_type = ObservationType.JointPosition):
        super().__init__(config, task_dir, json_file_name, action_type, observation_type)
        # Support single task dir (str) or multiple task dirs (list/tuple of str)
        if isinstance(task_dir, (list, tuple)):
            self._task_dirs = list(task_dir)
        else:
            self._task_dirs = [task_dir]
        # Keep a canonical first task dir for backwards-compatibility with base class usage
        self._task_dir = self._task_dirs[0]
        self._observation_type = observation_type
        # self._push_to_repo = self._config.get("push_to_repo", False)
        self._robot_name = config.get("robot_name", "fr3")
        self._repo_name = config.get("repo_name", "peg_in_hole")
        # 不再支持多 task 列表，统一为单一 task 路径
        self._contain_depth = config.get(f'contain_depth', False)
        # Allow custom_prompt to be provided as a string or a list[str] in YAML.
        # Normalize to a single string (first element if list) or None.
        _cp = config.get("custom_prompt", None)
        if isinstance(_cp, list):
            _cp = _cp[0] if len(_cp) > 0 else None
        elif not (isinstance(_cp, str) or _cp is None):
            _cp = str(_cp)
        self._custom_prompt = _cp
        self._output_root_path = config.get("root_path", "../assets/data")
        cur_path = os.path.dirname(os.path.abspath(__file__))
        self._output_root_path = os.path.join(cur_path, self._output_root_path)
        self._load_fps = config.get("fps", 15) #TODO: @hph default == 15
        self._num_writer_thread = config.get(f'num_writer_thread', 1)
        self._num_writer_process = config.get(f'num_writer_process', 1)
        # Optional: batch encode videos every N episodes to amortize encoder startup costs
        self._batch_encoding_size = int(config.get("batch_encoding_size", 1))
        # Optional: force a specific video backend ("torchcodec" preferred when available)
        self._video_backend = config.get("video_backend", "torchcodec")
        
    def get_example_feature_dim(self, example_step):
        # try to confirm the images and state length,
        images = example_step.get("colors", {})
        depths = example_step.get("depths", {})
        # Track image entries with their source (colors/depths) to avoid ambiguous lookup
        self._image_keys = []  # raw keys in the episode dict, e.g. 'left', 'right'
        self._image_resolutions = []
        self._image_sources = []  # parallel list with values 'colors' or 'depths'
        for key, image in images.items():
            self._image_keys.append(key)
            # Handle both numpy array and file path (str)
            if hasattr(image, "shape"):
                shape = tuple(image.shape)
            else:
                raise TypeError(f"Unsupported image type for key '{key}': {type(image)}")
            self._image_resolutions.append(shape)
            self._image_sources.append("colors")
        if self._contain_depth:
            for key, image in depths.items():
                self._image_keys.append(key)
                if hasattr(image, "shape"):
                    shape = tuple(image.shape)
                else:
                    raise TypeError(f"Unsupported depth image type for key '{key}': {type(image)}")
                self._image_resolutions.append(shape)
                self._image_sources.append("depths")
        log.info(f'image keys: {self._image_keys}')
        self._obs_states_dim = 0; self._action_dim = 0
        self._ee_states_dim = 0
        obs_states = example_step.get("observations", {})
        actions = example_step.get("actions", {})
        for key, state in obs_states.items():
            # log.info(f'state: {state}, len {len(state)} for {key}')
            self._obs_states_dim += len(state)
        for key, action in actions.items():
            self._action_dim += len(action)
        
        log.info(f'obs state: {self._obs_states_dim}')
        log.info(f'action dim: {self._action_dim}')
    
    def convert_dataset(self):
        """
        Convert raw HIROL episodes into a LeRobotDataset.

        - States/actions dtype unified to float32 for smaller, consistent storage.
        - Frame sampling is based on timestamps from the episode metadata instead of a fixed step skip.
        """
        # Target save path for the converted LeRobot dataset
        save_path = os.path.join(self._output_root_path, self._repo_name)

        # If a dataset already exists at save_path, append new episodes instead of opening in reader mode
        # to avoid network/caching issues in restricted environments.
        if os.path.exists(save_path):
            log.info(
                f"Found existing LeRobotDataset at {save_path}, will append/continue without opening in reader mode."
            )

        # Load a reference episode without subsampling to infer feature dims and timestamp keys.
        # Use the first task_dir as template when multiple dirs are provided.
        first_task_dir = self._task_dirs[0]
        episode_dir = sorted(os.listdir(first_task_dir))[0]

        skip_nums_steps = int(30.0 / self._load_fps)
        example_episode, _ = self.load_episode(first_task_dir, episode_dir, skip_nums_steps)
        self.get_example_feature_dim(example_episode[0])
        feature_dicts = {}
        # Standardize to LeRobot naming: observation.images.<name>
        for i, key in enumerate(self._image_keys):
            feature_dicts[f"observation.images.{key}"] = dict(
                dtype="video",
                shape=self._image_resolutions[i],  # (H, W, C); lerobot will convert to (C,H,W)
                names=["height", "width", "channels"],
            )
        # observation.state and action (single, not plural)
        feature_dicts["observation.state"] = {
            "dtype": "float32",
            "shape": (self._obs_states_dim,),
            # Provide per-dimension placeholder names to match shape
            "names": [f"s{i}" for i in range(self._obs_states_dim)],
        }
        feature_dicts["action"] = {
            "dtype": "float32",
            "shape": (self._action_dim,),
            # Provide per-dimension placeholder names to match shape
            "names": [f"a{i}" for i in range(self._action_dim)],
        }
        
        save_path = os.path.join(self._output_root_path, self._repo_name)
        log.info(f'save_path: {save_path}')
        self._lerobot_dataset = LeRobotDataset.create(
            root= save_path,
            repo_id=self._repo_name,
            robot_type=self._robot_name,
            fps=self._load_fps,
            features=feature_dicts,
            image_writer_threads=self._num_writer_thread,
            image_writer_processes=self._num_writer_process,
            batch_encoding_size=self._batch_encoding_size,
            video_backend=self._video_backend,
        )

        self._lerobot_dataset.meta.metadata_buffer_size = 1
        log.info('Set metadata_buffer_size=1 for immediate meta flush per episode')
        
        # Stats for dimension mismatches and skipped episodes
        state_dismatch_list = []; action_dismatch_list = []
        state_step_list = []; action_step_list = []
        skipped_empty_episodes = []
        total_raw_episodes = 0
        total_saved_episodes = 0
        # 支持单一或多个 task 目录
        for task_dir in self._task_dirs:
            dirs = sorted(os.listdir(task_dir))
            for cur_episode_dir in tqdm(dirs, desc=f"processing episodes in {task_dir}", unit="episode"):
                total_raw_episodes += 1
                episode_data, text_info = self.load_episode(task_dir, cur_episode_dir, skip_nums_steps)
                if episode_data is None:
                    continue
                # Determine the primary timestamp key (prefer color stream; fallback to depth)
                # Note: keys are consistent across steps; derive from the first valid step
                primary_ts_key = None
                prefer_colors_ts = True
                for st in episode_data:
                    cts = st.get("colors_time_stamp", {}) or {}
                    dts = st.get("depths_time_stamp", {}) or {}
                    if len(cts) > 0:
                        # pick any color key that exists
                        primary_ts_key = next(iter(cts.keys()))
                        prefer_colors_ts = True
                        break
                    elif len(dts) > 0:
                        primary_ts_key = next(iter(dts.keys()))
                        prefer_colors_ts = False
                        break
                # Target interval in seconds between two kept frames
                target_dt = 1.0 / max(float(self._load_fps), 1e-6)
                last_kept_ts = None
                state_wrong_nums = 0; action_wrong_nums = 0
                num_valid_frames = 0
                for num_step, step in tqdm(enumerate(episode_data), desc=f"processing steps", unit="step"):
                    # Subsample by timestamps: keep first step or if elapsed >= target_dt
                    ts_dict = step.get("colors_time_stamp", {}) if prefer_colors_ts else step.get("depths_time_stamp", {})
                    ts = None
                    if isinstance(ts_dict, dict) and primary_ts_key is not None:
                        ts = ts_dict.get(primary_ts_key, None)
                    if last_kept_ts is not None and ts is not None and (ts - last_kept_ts) < target_dt:
                        continue
                    frame_feature = {}
                    # vision images: write to observation.images.<name>
                    # Ensure all required image keys exist for this step; otherwise skip this step
                    all_images_present = True
                    for idx, image_key in enumerate(self._image_keys):
                        source = self._image_sources[idx]
                        src_dict = step.get(source, {}) or {}
                        if image_key not in src_dict:
                            all_images_present = False
                            break
                    if not all_images_present:
                        # Skip this frame to keep video streams aligned and avoid KeyError
                        continue
                    for idx, image_key in enumerate(self._image_keys):
                        source = self._image_sources[idx]
                        frame_feature[f"observation.images.{image_key}"] = step[source][image_key]
                    # obs states
                    state_list = []
                    obs_states = step["observations"]
                    for key, obs_state in obs_states.items():
                        state_list.append(np.asarray(obs_state, dtype=np.float32))
                    frame_feature["observation.state"] = (
                        np.concatenate(state_list, axis=0).astype(np.float32, copy=False)
                        if len(state_list) > 0 else np.empty((0,), dtype=np.float32)
                    )
                    if len(frame_feature["observation.state"]) != self._obs_states_dim:
                        log.warn(f'{task_dir} {cur_episode_dir} has wrong state dim: {len(frame_feature["observation.state"])} in {num_step}th step')
                        state_wrong = f'{task_dir}_{cur_episode_dir}'
                        if not state_wrong in state_dismatch_list:
                            state_dismatch_list.append(state_wrong)
                        state_wrong_nums += 1
                        continue
                    # actions
                    action_list = []
                    for key, value in step["actions"].items():
                        action_list.append(np.asarray(value, dtype=np.float32))
                    frame_feature["action"] = (
                        np.concatenate(action_list, axis=0).astype(np.float32, copy=False)
                        if len(action_list) > 0 else np.empty((0,), dtype=np.float32)
                    )
                    action_dim = len(frame_feature["action"])
                    if action_dim != self._action_dim:
                        log.warn(f'{task_dir} {cur_episode_dir} has wrong action dim: {action_dim} in {num_step}th step')
                        action_wrong = f'{task_dir}_{cur_episode_dir}'
                        if action_wrong not in action_dismatch_list:
                            action_dismatch_list.append(action_wrong)
                        action_wrong_nums += 1
                        continue
                    # 单一 task 的提示词：优先使用配置中的 custom_prompt，否则使用数据中的文本；
                    # 若两者都不可用，则回退为一个占位符，避免产生 None。
                    if isinstance(self._custom_prompt, str) and len(self._custom_prompt) > 0:
                        text = self._custom_prompt
                    else:
                        text = text_info if isinstance(text_info, str) and len(text_info) > 0 else "Perform the task."
                    frame_feature["task"] = text
                    
                    self._lerobot_dataset.add_frame(frame=frame_feature)
                    num_valid_frames += 1
                    if ts is not None:
                        last_kept_ts = ts
                if state_wrong_nums != 0:
                    state_step_list.append(state_wrong_nums)
                if action_wrong_nums != 0:
                    action_step_list.append(action_wrong_nums)

                # If this episode ended up with no valid frames, treat it as invalid data and skip it
                if num_valid_frames == 0:
                    log.warning(
                        f"Skip episode {task_dir}/{cur_episode_dir} because no valid frames were added "
                        f"(state_wrong_steps={state_wrong_nums}, action_wrong_steps={action_wrong_nums})."
                    )
                    skipped_empty_episodes.append(f"{task_dir}/{cur_episode_dir}")
                    # Clear current episode buffer and delete temporary images to avoid affecting later episodes
                    if getattr(self._lerobot_dataset, 'episode_buffer', None) is not None:
                        self._lerobot_dataset.clear_episode_buffer(
                            delete_images=len(self._lerobot_dataset.meta.image_keys) > 0
                        )
                    del episode_data
                    continue

                self._lerobot_dataset.save_episode()
                total_saved_episodes += 1
                del episode_data
                
                log.info(f'Successfully processed {task_dir}/{cur_episode_dir} and saved to {save_path}')
                log.info(f'{len(self._lack_data_json_list)} lacks the data.json files: {self._lack_data_json_list}')
                log.info(f'state: {state_dismatch_list}, len: {state_step_list}')
                log.info(f'action: {action_dismatch_list}, len: {action_step_list}')
            
        log.info(f'{len(self._lack_data_json_list)} lacks the data.json files: {self._lack_data_json_list}')
        log.info(f'state: {state_dismatch_list}, len: {state_step_list}')
        log.info(f'action: {action_dismatch_list}, len: {action_step_list}')
        log.info(
            f'Finished HIROL to LeRobot conversion: raw_episodes={total_raw_episodes}, '
            f'saved_episodes={total_saved_episodes}, skipped_empty_episodes={len(skipped_empty_episodes)}'
        )
        if skipped_empty_episodes:
            # Print a subset of skipped episodes to avoid overly long logs
            max_show = 10
            show_eps = skipped_empty_episodes[:max_show]
            log.info(
                f'Skipped empty episodes (showing at most {max_show}): '
                f'{show_eps}{" ..." if len(skipped_empty_episodes) > max_show else ""}'
            )
        
        self._lerobot_dataset.finalize()
        return self._lerobot_dataset


def _parse_obs_action_from_repo_name(repo_name: str):
    """Parse observation/action types from repo_name suffix, e.g. '*_q2q'.

    Supported tokens:
      obs: 'q', 'dq', 'ee', 'dee', 'q_ee', 'mask', 'ft'
      act: 'q', 'dq', 'ee', 'dee'
    """
    try:
        suffix = repo_name.split("_")[-1].lower()
        if "2" not in suffix:
            raise ValueError
        obs_tok, act_tok = suffix.split("2", 1)
        obs_map = {
            "q": ObservationType.JointPosition,
            "dq": ObservationType.DeltaJointPosition,
            "ee": ObservationType.EEPose,
            "dee": ObservationType.DeltaEEPose,
            "q_ee": ObservationType.JP_EEPose,
            "mask": ObservationType.Mask,
            "ft": ObservationType.FT,
        }
        act_map = {
            "q": ActionType.JointPosition,
            "dq": ActionType.DeltaJointPosition,
            "ee": ActionType.EEPose,
            "dee": ActionType.DeltaEEPose,
            "cq": ActionType.CommandJointPosition,
            "cdq": ActionType.CommandDeltaJointPosition,
            "cee": ActionType.CommandEEPose,
            "cdee": ActionType.CommandDeltaEEPose,
        }
        obs_type = obs_map[obs_tok]
        act_type = act_map[act_tok]
        return obs_type, act_type
    except Exception:
        log.warn(f"Failed to parse obs/action from repo_name '{repo_name}', default to q2q")
        return ObservationType.JointPosition, ActionType.JointPosition


if __name__ == '__main__':
    import argparse, yaml
    cur_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Convert HIROL data to LeRobotDataset (single task).')
    parser.add_argument('-c', '--config', type=str,
                        default=os.path.join(cur_path, 'config', 'insert_pinboard_1113_q2q.yaml'),
                        help='Path to YAML config file (default: config/insert_pinboard.yaml relative to this file)')
    args = parser.parse_args()

    cfg_file = os.path.expanduser(args.config)
    with open(cfg_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(f'Using config: {cfg_file}')
    print(f'config: {config}')

    # data_dir 可为字符串（单一路径）或列表（多路径）
    data_dir_cfg = config.get("data_dir")
    if data_dir_cfg is None:
        raise ValueError("data_dir is required in config YAML")
    if isinstance(data_dir_cfg, (list, tuple)):
        task_dirs = list(data_dir_cfg)
    else:
        task_dirs = [data_dir_cfg]

    # 从 repo_name 后缀解析 obs/action 对，例如 q2q -> JointPosition => JointPosition
    repo_name = config.get("repo_name", "")
    obs_type, action_type = _parse_obs_action_from_repo_name(repo_name)

    lerobot_dataset = LerobotLoader(config, task_dirs, action_type=action_type,
                                    observation_type=obs_type)
    _ = lerobot_dataset.convert_dataset()  # writer instance; do not iterate on it

    # 重新以“读者”模式打开数据集再进行遍历
    repo_name = config.get("repo_name", "")
    root_base = os.path.join(cur_path, config.get("root_path", "../assets/data"))
    root = os.path.join(root_base, repo_name)
    reader_ds = LeRobotDataset(repo_id=repo_name, root=root)
    print(f'len dataset: {len(reader_ds)}')
    print(f'dataset: {reader_ds}')
    # lerobot 自动转换为 tensor
    data_sample = next(iter(reader_ds))
    print(f'data: {data_sample}')
