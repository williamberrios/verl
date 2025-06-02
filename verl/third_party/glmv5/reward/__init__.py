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
from verl.third_party.glmv5.reward.factuality_weighted import factuality_weighted_reward_batch
from verl.third_party.glmv5.reward.lmunit_weighted_async import lmunit_weighted_reward_batch_async 
from verl.third_party.glmv5.reward.factuality_weighted_async import factuality_weighted_reward_batch_async 
from verl.third_party.glmv5.reward.ingen_format_async import ingen_format_batch_async
from verl.third_party.glmv5.reward.think_format_async import think_format_batch_async
from verl.third_party.glmv5.reward.inline_attribution_format_async import inline_attribution_format_batch_async

from typing import Union
import asyncio
from typing import List
def compute_rewards(config: Dict[str, float] = None,
                    output_all: bool = False,
                    cot_rl_experiment: bool = False,
                    uuid: str = None) -> callable:
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
    #rint(f"Reward functions Keys: {REWARD_FUNCTIONS.keys()}")
    run_async = False
    for name in config:
        if name not in REWARD_FUNCTIONS:
            raise ValueError(f"Reward function '{name}' not found in registry. Available reward functions: {list(REWARD_FUNCTIONS.keys())}")
        if "async" in name:
            # TODO: Make this configurable in config
            print(f"Running {name} asynchronously")
            run_async = True
    if cot_rl_experiment:
        print("================================================")
        print(f"Running in CoT RL experiment")
        print("================================================")
            
    def reward_fn(data_sources: List[str],
                  solution_strs: List[str],
                  ground_truths: List[str], 
                  extra_infos: List[Dict[str, Any]] = None) -> Union[float, Dict[str, Any]]:
        """
        Compute rewards for a list of data sources, solution strings, and ground truths.
        
        Args:
            data_sources: List of data source strings
            solution_strs: List of solution strings
            ground_truths: List of ground truth strings
        
        """
        total_rewards = [0.0] * len(data_sources)
        list_dict_rewards = [{} for _ in range(len(data_sources))]
        
        print(f"Config: {config}")
        print(f"config.items: {config.keys()}")
        
        if run_async:
            print(f"Running asynchronously")
            # Get reward functions for each configured reward
            reward_fn_names = list(config.keys())
            reward_pipelines_fn = [REWARD_FUNCTIONS[name] for name in reward_fn_names]
            
            # This is not truly parallel since asyncio.run() creates a new event loop
            # and blocks until completion. To make this truly parallel, we should:
            # 1. Use an existing event loop rather than creating a new one
            # 2. Consider using multiprocessing for CPU-bound tasks
            # 3. Ensure the reward functions themselves are async-optimized
            async def run_parallel(data_sources: List[str],
                                    solution_strs: List[str],
                                    ground_truths: List[str],
                                    extra_infos: List[Dict[str, Any]],
                                    cot_rl_experiment: bool = False,
                                    uuid: str = None):
                tasks = [
                        reward_pipeline_fn(data_sources,
                                           solution_strs,
                                           ground_truths,
                                           extra_infos,
                                           cot_rl_experiment,
                                           uuid)
                    for reward_pipeline_fn in reward_pipelines_fn
                ]
                return await asyncio.gather(*tasks)
            
            rewards = asyncio.run(run_parallel(data_sources,
                                               solution_strs,
                                               ground_truths,
                                               extra_infos,
                                               cot_rl_experiment,
                                               uuid))
            #print(f"Rewards: {rewards}")
            
            for j, reward_name in enumerate(reward_fn_names):
                weight = config[reward_name]
                for i in range(len(data_sources)):
                    list_dict_rewards[i][reward_name] = rewards[j][i]
                    if j == 0:
                        total_rewards[i] = rewards[j][i] * weight
                    else:
                        total_rewards[i] += rewards[j][i] * weight
                    list_dict_rewards[i]["score"] = total_rewards[i]
                
            print(f"list_dict_rewards: {list_dict_rewards}")
            if output_all:
                return list_dict_rewards
            else:
                return total_rewards
        else:
            for name, weight in config.items():
                #print(f"name: {name}, weight: {weight}")
                #print(f"Length of data_sources: {len(data_sources)}")
                #print(f"Length of solution_strs: {len(solution_strs)}")
                #print(f"Length of ground_truths: {len(ground_truths)}")
                #print(f"Length of extra_infos: {len(extra_infos)}")
                rewards = REWARD_FUNCTIONS[name](data_sources,
                                                    solution_strs,
                                                    ground_truths,
                                                    extra_infos,
                                                    cot_rl_experiment,
                                                    uuid)
                
                for i in range(len(data_sources)):
                    list_dict_rewards[i][name] = rewards[i]
                
                if total_rewards is None:
                    total_rewards = [r * weight for r in rewards]
                else:
                    total_rewards = [t + (r * weight) for t, r in zip(total_rewards, rewards)]
                    
                # Store total rewards in each dict
                for i in range(len(data_sources)):
                    list_dict_rewards[i]["score"] = total_rewards[i]
                
            if output_all:
                # Return list of dicts, where each dict contains the rewards for one sample
                return list_dict_rewards
            else:
                # Return a list of rewards
                # each value is a float reward from the sample
                return total_rewards
        
    return reward_fn
