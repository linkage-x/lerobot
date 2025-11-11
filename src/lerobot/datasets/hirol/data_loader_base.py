import abc, os
import enum
from .reader import RerunEpisodeReader, ActionType, ObservationType
import glog as log

class DataLoaderBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, config, task_dir:str, json_file_name:str = "data.json", action_type:ActionType = ActionType.JointPosition,
                 observation_type = ObservationType.JointPosition):
        self._config = config
        self._action_prediction_step = config.get("action_prediction_step", 2)
        self._action_type = action_type
        self._obs_type = observation_type
        self._action_ori_type = config.get("action_ori_type", "euler")
        self._rotation_transform = config.get("rotation_transform", None)
        self._contain_ft = config.get(f'contain_ft', False)
        self._task_dir = task_dir
        self._json_file = json_file_name
        self._lack_data_json_list = []
    
    """
        parse for single episode given task dir and episode dir
    """
    def load_episode(self, task_dir, episode_dir, skip_steps_nums):
        self._episode_reader = RerunEpisodeReader(task_dir=task_dir,
                                                  json_file=self._json_file,
                                                  action_type=self._action_type,
                                                  action_prediction_step=self._action_prediction_step,
                                                  action_ori_type=self._action_ori_type,
                                                  observation_type=self._obs_type,
                                                  rotation_transform=self._rotation_transform,
                                                  contain_ft=self._contain_ft)
        if 'episode' in episode_dir:
            episode_number = int(episode_dir.lstrip("episode_"))
            episode_id = episode_number
            print(f'Tring to load the {episode_number}th episode data in {task_dir}')
            episode_data = self._episode_reader.return_episode_data(episode_number, skip_steps_nums)
            if episode_data is None:
                self._lack_data_json_list.append(f"{task_dir}_{episode_dir}")
                return None, None
        else:
            log.warn(f"{episode_dir} in {task_dir} does not contain episode")
            return None, None
        
        text_info = self._episode_reader.get_episode_text_info(episode_id)
        return episode_data, text_info  
        
    @abc.abstractmethod
    def convert_dataset(self):
        raise NotImplementedError
    
