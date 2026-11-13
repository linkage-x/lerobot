"""Shared native-schema recording hooks for hardware adapters."""

from typing import Any

from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.processor import make_default_processors

from .catalog import get_model


def native_recording_processors(config: Any, robot: Any, teleop: Any):
    model = get_model("robot", config.robot.type)
    if model.recording_processors:
        return model.recording_processors(config, robot, teleop)
    if set(robot.action_features) != set(teleop.action_features):
        raise ValueError(
            "Robot and teleoperator action schemas differ; register recording_processors in the hardware adapter."
        )
    return make_default_processors()


def native_recording_features(robot: Any, teleop: Any, processors: tuple, *, use_videos: bool) -> dict:
    teleop_action, _, observation = processors
    result = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action,
            initial_features=create_initial_features(action=teleop.action_features),
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=observation,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=use_videos,
        ),
    )
    timestamp_names = tuple(getattr(robot, "capture_timestamp_feature_names", ()))
    if timestamp_names:
        result["observation.device_capture_timestamp"] = {
            "dtype": "float64",
            "shape": (len(timestamp_names),),
            "names": list(timestamp_names),
        }
    return result
