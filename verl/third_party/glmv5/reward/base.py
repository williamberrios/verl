"""
Python file for registering reward functions.

Usage:
@register_reward_function("my_reward")
def my_reward_function(batch: Any) -> np.ndarray:
    ...
"""
import os
import logging
import numpy as np
import json
import requests
from tqdm import trange
from typing import List, Dict, Any, Callable, Optional
from functools import wraps
from collections import defaultdict
try:
    import dawn.orch_v2 as orch
except ModuleNotFoundError as e:
    logging.error(f"{e}")
    logging.error("Hint: Did the start-up code manage to add the dawn path to PYTHONPATH?")


from dawn.utils.logging import get_logger, setup_logging, set_verbosity_info
logger = get_logger()
setup_logging()
set_verbosity_info()
    

# Registry for all reward functions
REWARD_FUNCTIONS: Dict[str, Callable] = {}

def register_reward_function(name: str) -> Callable:
    """
    Decorator to register a reward function.
    
    Usage:
    @register_reward_function("my_reward")
    def my_reward_function(batch: Any) -> np.ndarray:
        ...
    
    Args:
        name: Unique identifier for the reward function
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        if name in REWARD_FUNCTIONS:
            raise ValueError(f"Reward function '{name}' is already registered")
        REWARD_FUNCTIONS[name] = func
        return func
    return decorator


def reward_function(name: str) -> Callable:
    """
    Combined decorator to:
    1. Register the function in the global reward registry.
    
    Usage:
    @reward_function("my_reward")
    def my_reward(batch: Any) -> np.ndarray:
        ...
    """
    def decorator(func: Callable) -> Callable:
        wrapped = register_reward_function(name)(func)
        return wrapped
    return decorator