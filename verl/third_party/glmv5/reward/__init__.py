"""
This module demonstrates sample outcome-based reward functions.

Usage and Interface
-------------------
- These functions expect a `data_source`, `solution_str`, `ground_truth` and optional `extra_info` parameters.
- Each solution is evaluated against the ground truth to produce a reward score.
- The function returns a float reward value between 0 and 1.
"""

import os
import numpy as np
from typing import Dict, Any
from verl.third_party.glmv5.reward.base import REWARD_FUNCTIONS
# Import all reward modules to ensure they are registered
from verl.third_party.glmv5.reward.lmunit_not_weighted import lmunit_not_weighted_reward_batch
from verl.third_party.glmv5.reward.ingen_format import ingen_format_batch
from verl.third_party.glmv5.reward.lmunit_weighted import lmunit_weighted_reward_batch


def compute_rewards(config: Dict[str, float] = None, output_all: bool = False) -> callable:
    """
    Returns a reward function that computes weighted sum of multiple reward functions.
    
    Args:
        config: Dictionary mapping reward function names to their weights
        
    Returns:
        A function that takes (data_source, solution_str, ground_truth, extra_info) and returns
        the weighted sum of rewards from the configured reward functions.
    
    Example:
    config = {
        "lmunit": 1.0,
        "lmequivalence": 0.5,
        "lmranking": 1.0,
        "formatting": 0.25,
    }
    reward_fn = compute_rewards(config)
    reward = reward_fn(data_source, solution_str, ground_truth, extra_info)
    """
    # Validate reward function names upfront
    print(f"Reward functions Keys: {REWARD_FUNCTIONS.keys()}")
    for name in config:
        if name not in REWARD_FUNCTIONS:
            raise ValueError(f"Reward function '{name}' not found in registry. Available reward functions: {list(REWARD_FUNCTIONS.keys())}")
    
    def reward_fn(data_sources, solution_strs, ground_truths, extra_infos=None):
        total_rewards = [0.0] * len(data_sources)
        dict_rewards = {}
        
        print(f"Config: {config}")
        print(f"config.items: {config.keys()}")
        
        for name, weight in config.items():
            print(f"name: {name}, weight: {weight}")
            print(f"Length of data_sources: {len(data_sources)}")
            print(f"Length of solution_strs: {len(solution_strs)}")
            print(f"Length of ground_truths: {len(ground_truths)}")
            print(f"Length of extra_infos: {len(extra_infos)}")
            
            rewards = REWARD_FUNCTIONS[name](data_sources, solution_strs, ground_truths, extra_infos)
            dict_rewards[name] = rewards
            
            if total_rewards is None:
                total_rewards = [r * weight for r in rewards]
            else:
                total_rewards = [t + (r * weight) for t, r in zip(total_rewards, rewards)]
            dict_rewards["total_rewards"] = total_rewards
            
        if output_all:
            return dict_rewards
        else:
            return total_rewards
        
    return reward_fn
