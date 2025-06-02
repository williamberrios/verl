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

# Strict verification of attributions
async def verify_attributions(text):
    """
    Check if ALL attributions within <fact></fact> tags are in correct [n]()() format.
    
    CORRECT FORMAT: [n]()() (number in brackets followed by exactly two empty parentheses)
    INCORRECT FORMATS: [n], [n](), [n]()()(), [n]()()()(), etc.
    
    Args:
        text (str): Input text to check
        
    Returns:
        int: 1 if ALL attributions are correct [n]()(), 0 otherwise
    """
    # Find all <fact></fact> tags
    fact_matches = re.findall(r'<fact>(.*?)</fact>', text, re.DOTALL)
    
    if not fact_matches:
        return True
    
    # CORRECT: Only [n]()() format is valid
    correct_pattern = r'\[(\d+)\]\(\)\(\)'
    
    # ALL: Any [n] followed by any number of () pairs (including zero)
    all_attributions_pattern = r'\[(\d+)\](\(\))*'
    
    total_correct = 0
    total_all = 0
    
    for fact_content in fact_matches:
        # Count correct attributions in this fact
        correct_count = len(re.findall(correct_pattern, fact_content))
        
        # Count all attributions in this fact
        all_count = len(re.findall(all_attributions_pattern, fact_content))
        
        total_correct += correct_count
        total_all += all_count
    
    # Return 1 only if ALL attributions are correct, 0 otherwise
    if total_all == 0:
        # Means there are facts without attributions, we don't want this
        return False  # No attributions found
    
    return True if total_correct == total_all else False


@reward_function("inline_attribution_format_batch_async")
async def inline_attribution_format_batch_async(data_sources: List[str],
                                   solution_strs: List[str],
                                   ground_truths: List[str],
                                   extra_infos: List[Dict[str, Any]] = None,
                                   cot_rl_experiment: bool = False,
                                   uuid: str = None,
                                   config: Dict[str, float] = None) -> List[float]:
    """
    Computes rewards using ingen format.
    
    """
    rewards = []
    for solution in solution_strs:
        if not await verify_attributions(solution):
            reward = 0.0
        else:
            reward = 1.0
        rewards.append(reward)
    return rewards
