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
# Cache the LMUnit pipeline globally to avoid repeated initialization
lmunit_pipeline = None
LMUNIT_MAX_REWARD = 5.0

@reward_function("lmunit_weighted_reward_batch")
def lmunit_weighted_reward_batch(data_sources: List[str],
                                 solution_strs: List[str],
                                 ground_truths: List[str],
                                 extra_infos: List[Dict[str, Any]] = None,
                                 config: Dict[str, float] = None) -> List[float]:
    """
    Computes rewards using LM-based unit test evaluation in batch.

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

    global lmunit_pipeline

    # Lazy init of the evaluator
    if not lmunit_pipeline:
        lmunit_pipeline = orch.LMUnitScorerWeighted(
            inference_backend='lmunit',
            persistent_server=False, # TODO: Make this True, rn waiting never ends
            #server_uptime=24,
            use_cache=False,
            #server_url=config.server_url, # If server_url is not provided, it will launch a new server
            fail_on_invalid_data=False,
            model = "lmunit-8b"
        )

    all_ut_samples = []
    sample_boundaries = []  # Track where each solution's samples start/end
    current_idx = 0
    use_rationale = "no"
    for solution_str, extra_info in zip(solution_strs, extra_infos):
        ut_samples = []
        for ut in extra_info['unit_tests']:
            ut_samples.append({
                "query": extra_info['query'],
                "response": solution_str,
                "unit_test": ut,
                "use_rationale": use_rationale
            })
            
        all_ut_samples.extend(ut_samples)
        sample_boundaries.append((current_idx, current_idx + len(ut_samples)))
        current_idx += len(ut_samples)


    results = lmunit_pipeline(all_ut_samples)
    # Calculate scores for each solution
    final_scores = []
    for start_idx, end_idx in sample_boundaries:
        solution_results = results[start_idx:end_idx]
        scores = []
        for result in solution_results:
            if orch.is_invalid(result):
                logger.warning("Invalid Lmunit Score")
                scores.append(0.0)
            else:
                scores.append(float(result.get("score", 0.0)))
                
        valid_scores = [item for item in scores if isinstance(item, float)]
        
        if len(valid_scores) == 0:
            final_scores.append(0)
        else:
            final_scores.append(max(valid_scores) / LMUNIT_MAX_REWARD)  # Normalize to [0,1] range

    return final_scores


