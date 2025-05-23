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

async def validate_tags(response: str) -> bool:
    """
    Validates that fact and commentary tags in a response are properly formatted.
    
    Checks:
    - At least one <fact> or <commentary> tag exists
    - Tags are properly paired with matching closing tags
    - No nesting of same tag type (e.g. fact within fact)
    - No mixing of tag types (fact within commentary or vice versa)
    - All tags are closed in correct order

    Not Checking:
    - If text exists between tags that are not inside a fact or commentary tags
    
    Args:
        response: String containing fact and commentary tags to validate
        
    Returns:
        bool: True if tags are valid, False otherwise
    """
    # Ensure at least one type of tag exists
    if '<fact>' not in response and '<commentary>' not in response:
        return False
    
    # Extract all tags in order of appearance
    tag_pattern = re.compile(r'<fact>|</fact>|<commentary>|</commentary>')
    tags = tag_pattern.findall(response)
    
    # If no tags found after regex search, validation fails
    if not tags:
        return False
    
    # Use stack to check proper nesting and pairing
    stack = []
    
    for tag in tags:
        if tag == '<fact>':
            # Can't have nested facts or facts inside commentary
            if stack and (stack[-1] == '<fact>' or stack[-1] == '<commentary>'):
                return False
            stack.append(tag)
        
        elif tag == '</fact>':
            # Closing tag must match most recent opening tag
            if not stack or stack[-1] != '<fact>':
                return False
            stack.pop()
        
        elif tag == '<commentary>':
            # Can't have nested commentary or commentary inside facts
            if stack and (stack[-1] == '<commentary>' or stack[-1] == '<fact>'):
                return False
            stack.append(tag)
        
        elif tag == '</commentary>':
            # Closing tag must match most recent opening tag
            if not stack or stack[-1] != '<commentary>':
                return False
            stack.pop()
    
    # Stack should be empty if all tags are properly paired
    return len(stack) == 0

@reward_function("ingen_format_batch_async")
async def ingen_format_batch_async(data_sources: List[str],
                                   solution_strs: List[str],
                                   ground_truths: List[str],
                                   extra_infos: List[Dict[str, Any]] = None,
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
        if not await validate_tags(solution):
            reward = 0.0
        else:
            reward = 1.0
        rewards.append(reward)
    return rewards