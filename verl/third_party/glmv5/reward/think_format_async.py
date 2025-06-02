import os
import numpy as np
from typing import List, Any, Optional
from collections import defaultdict
import dawn.orch_v2 as orch

# Reward function registration and utilities
from verl.third_party.glmv5.reward.base import (
    reward_function,
    logger
)
from typing import Dict
import re


async def think_format_reward_async(predict_str: str) -> float:
    """
    Description:
    - You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
    - The reasoning process MUST BE enclosed within ##THINK and ##END-THINK and encapsulated between <commentary> and </commentary>.
    - The final answer MUST BE after ##FINAL-ANSWER. 
    """
    pattern = re.compile(r"##THINK.*?##END-THINK.*?##FINAL-ANSWER.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str.strip())   # TODO: strip because policy model adds a \n 
    return 1.0 if match_result else 0.0


@reward_function("think_format_batch_async")
async def think_format_batch_async(data_sources: List[str],
                                   solution_strs: List[str],
                                   ground_truths: List[str],
                                   extra_infos: List[Dict[str, Any]] = None,
                                   cot_rl_experiment: bool = False,
                                   uuid: str = None,
                                   config: Dict[str, float] = None) -> List[float]:
    """
    Computes rewards using ingen format.

    Args:
        data_sources: List of data source identifiers (not used in this function)
        solution_strs: List of solution strings to evaluate
        ground_truths: List of ground truth strings (not used in this function)
        extra_infos: List of dicts containing 'query' and 'unit_tests' for each solution
        config: Optional config dict with parameters (not used in this function)

    Returns:
        List of float scores normalized to [0,1] range

    Notes:
        - If response is missing -> reward = 0.0
        - If unit tests are missing -> reward = None
        - Uses extra_infos['query'] and extra_infos['unit_tests'] to construct test cases
        - Evaluates solutions against unit tests using LMUnit pipeline
    """
    rewards = []
    for solution in solution_strs:
        if not await think_format_reward_async(solution):
            reward = 0.0
        else:
            reward = 1.0
        rewards.append(reward)
    return rewards