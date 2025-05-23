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
from typing import Dict, List
# Cache the LMUnit pipeline globally to avoid repeated initialization
factuality_pipeline = None
FACTUALITY_MAX_REWARD = 5.0



class CalculateFactualityScoreAggFactuality(orch.ListFunctionComponent):
    async def forward(self,
                      grouped_scores: List[float],
                      fixed_knowledge: List[str],
                      grouped_claims: List[str]):
        fct_scores = []

        for scores, knowledge, claims in zip(
            grouped_scores, fixed_knowledge, grouped_claims
        ):
            if orch.is_invalid(claims):
                # No claims are found so nothing is ungrounded
                # Ideally the claim model returns an empty list, not an invalid list. This should be fixed.
                logger.warning(
                    "Found `InvalidData` in `CalculateFactualityScore`, giving the maximal score since this typically happens when no claims are found in the response. Be warned, this could also be due to your pipeline failing."
                )
                fct_scores.append(
                    {
                        "score": 1.0,
                        "metadata": {
                            "description": "No claims are found so nothing is ungrounded."
                        },
                        "status": "success",
                    }
                )

            elif len(claims) == 0:
                # No claims are found so nothing is ungrounded
                fct_scores.append(
                    {
                        "score": 1.0,
                        "metadata": {
                            "description": "No claims are found so nothing is ungrounded."
                        },
                        "status": "success",
                    }
                )

            elif (
                orch.is_invalid(knowledge)
                or "knowledge" not in knowledge
                or knowledge["knowledge"] in ["[]", [], ""]
            ):
                # There are claims but no knowledge so everything is ungrounded
                fct_scores.append(
                    {
                        "score": 0.0,
                    }
                )

            else:
                fct_scores_this_response = []
                for claim_score, claim in zip(scores, claims):
                    if orch.is_invalid(claim_score):
                        fct_scores_this_response.append(0.0)
                    else:
                        formatted_score = float(claim_score["score"])
                        fct_scores_this_response.append(formatted_score)
                        claim_score["claim"] = claim["claim"]
                fct_scores.append(
                    {
                        "score": np.mean(fct_scores_this_response),
                    }
                )
        return fct_scores


class FactualityPipeline(orch.Pipeline):
    def __init__(self):
        super().__init__()
        self.fix = orch.FixKnowledge()
        self.decompose = orch.DecomposeResponseToClaims(
            model="decompose-response-to-claims-model",
            inference_backend="online_vllm",
            persistent_server=True, 
            server_uptime=24,
            use_cache=False,
            fail_on_invalid_data=False,
        )
        self.evaluate = orch.AggfactScorerWeighted(
            model="aggfact-8b",
            inference_backend="http",
            http_backend_server_type="aggfact",
            persistent_server=True, 
            server_uptime=24,
            use_cache=False,
            fail_on_invalid_data=False,
        )
        self.use_rationale = "no"
        self.fct_score = CalculateFactualityScoreAggFactuality()

    async def forward(self, data):
        """
        Output would be a list of dicts with the following keys:
        - score: float
        """
        knowledge = orch.wrap(orch.get(data, "knowledge"), "knowledge")
        responses = orch.wrap(orch.get(data, "response"), "response")

        fixed_knowledge = self.fix(knowledge)
        claims = self.decompose(responses)
        flat_claims, response_ids = orch.concat_with_indices(claims)
        # Calling async_realize to compute all the claims and stop.
        flat_claims = await orch.realize_async(flat_claims)
        broadcast_knowledge = orch.broadcast(fixed_knowledge, response_ids)
        claims_and_knowledge = orch.merge(flat_claims, broadcast_knowledge)
        # Add use_rationale to claims_and_knowledge
        rationales = orch.map(lambda x: {"use_rationale": self.use_rationale}, claims_and_knowledge)
        claims_and_knowledge = orch.merge(claims_and_knowledge, rationales)
        flat_claim_scores = self.evaluate(claims_and_knowledge)
        grouped_scores = orch.group(flat_claim_scores, response_ids)
        return self.fct_score(grouped_scores, fixed_knowledge, claims)


@reward_function("factuality_weighted_reward_batch_async")
async def factuality_weighted_reward_batch_async(data_sources: List[str],
                                                 solution_strs: List[str],
                                                 ground_truths: List[str],
                                                 extra_infos: List[Dict[str, Any]] = None,
                                                 config: Dict[str, float] = None) -> List[float]:
    """
    Computes rewards using Factuality evaluation in batch.

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
        - Evaluates solutions against unit tests using Factuality pipeline
    """
    global factuality_pipeline
    if factuality_pipeline is None:
        factuality_pipeline = FactualityPipeline()

    # TODO: Complete the knowledge list
    knowledge_list = ["\n\n".join(f"[Chunk {i+1}]\n\n{chunk}" for i, chunk in enumerate(knowledge["chunks"])) for knowledge in extra_infos]
    data = [
        {
            "knowledge": knowledge,
            "response": response,
        }
        for knowledge, response in zip(knowledge_list, solution_strs)
    ]
    results = await factuality_pipeline.run_async(data)
    # Normalizing from 0 to 1
    rewards = [score["score"]/FACTUALITY_MAX_REWARD for score in results]
    #print(f"factuality_weighted_reward_batch_async: {rewards}")
    return rewards


