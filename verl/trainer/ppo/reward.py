# Copyright 2025 Individual Contributor: Thibaut Barroyer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import os
from functools import partial

import ray
from typing import Dict, Any, Optional, Callable
from verl import DataProto
from verl.third_party.glmv5.reward import compute_rewards as compute_glmv5_rewards
from enum import Enum
import json
class RewardFunction(str, Enum):
    GLM = "glmv5"
    CUSTOM = "custom"
from verl.utils.reward_score import default_compute_score


def get_custom_reward_fn(config):
    import importlib.util
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn

def get_custom_reward_glmv5_fn(config: Dict[str, Any]) -> Optional[Callable]:
    reward_fn_config = config.get("custom_reward_function") or {}
    if not reward_fn_config:
        return None
    cot_rl_experiment = reward_fn_config.get("cot_rl_experiment", False)
    json_file_path = reward_fn_config.get("json_reward_config_path")
    output_all = reward_fn_config.get("output_all", False)
    uuid = reward_fn_config.get("uuid", None)
    if uuid == "":
        uuid = None
    print(f"json_file_path: {json_file_path}")
    with open(json_file_path, 'r') as f:
        json_file = json.load(f)
    raw_fn = compute_glmv5_rewards(config=json_file,
                                   output_all=output_all,
                                   cot_rl_experiment=cot_rl_experiment,
                                   uuid=uuid)
    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn

def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == "naive":
        from verl.workers.reward_manager import NaiveRewardManager

        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == "prime":
        from verl.workers.reward_manager import PrimeRewardManager

        reward_manager_cls = PrimeRewardManager
    elif reward_manager_name == "batch":
        from verl.workers.reward_manager import BatchRewardManager

        reward_manager_cls = BatchRewardManager
    elif reward_manager_name == "dapo":
        from verl.workers.reward_manager import DAPORewardManager

        reward_manager_cls = DAPORewardManager
    else:
        raise NotImplementedError
    custom_reward_fn_dict = config.get("custom_reward_function") or {}
    if custom_reward_fn_dict:
        if custom_reward_fn_dict.get("class_reward_fn") == RewardFunction.GLM:
            compute_score = get_custom_reward_glmv5_fn(config)
        elif custom_reward_fn_dict.get("class_reward_fn") == RewardFunction.CUSTOM:
            compute_score = get_custom_reward_fn(config)
            assert compute_score is not None, "Custom reward function is not defined"
        else:
            raise ValueError(f"Invalid reward function class: {custom_reward_fn_dict.get('class_reward_fn')}")
    else:
        compute_score = get_custom_reward_fn(config)
    print("compute_score", compute_score)
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


def compute_reward(data: DataProto, reward_fn: Callable, output_all: bool = False):
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """
    try:
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result["reward_extra_info"]
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}

    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config, tokenizer):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    return compute_reward(data, reward_fn)
