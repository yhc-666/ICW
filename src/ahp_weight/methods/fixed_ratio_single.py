"""Fixed-ratio AHP with one full consistency-triggered review."""

import asyncio
import itertools
import json
import math
from dataclasses import dataclass

import numpy as np

from ahp_weight.agents.pair_compare_agent import Judgment, PairCompareAgent
from ahp_weight.ahp import solve_ahp_weights
from schemas.rubric import Rubric


@dataclass(frozen=True)
class PairComparison:
    index_i: int
    index_j: int
    forward: Judgment
    reverse: Judgment

    @property
    def winner_id(self) -> str | None:
        if self.forward.winner_id == self.reverse.winner_id:
            return self.forward.winner_id
        return None

    def to_history(self) -> dict:
        forward = self.forward.winner_id
        reverse = self.reverse.winner_id
        if forward is None and reverse is None:
            status = "both_neutral"
        elif forward is None or reverse is None:
            status = "one_order_neutral"
        elif forward != reverse:
            status = "opposite_winners"
        else:
            status = "agreed_winner"
        return {
            "rubric_pair": [self.forward.first_id, self.forward.second_id],
            "forward": self.forward.to_history(),
            "reverse": self.reverse.to_history(),
            "status": status,
            "collapsed_winner_rubric_id": self.winner_id,
        }


async def collect_comparisons(
    *, prompt: str, rubrics: list[Rubric], agent: PairCompareAgent, feedback: str = ""
) -> list[PairComparison]:
    """Collect both display orders of every pair with the same review context."""
    async def compare_pair(index_i: int, index_j: int) -> PairComparison:
        first, second = rubrics[index_i], rubrics[index_j]
        forward, reverse = await asyncio.gather(
            agent.compare(prompt=prompt, rubrics=rubrics, first=first, second=second, feedback=feedback),
            agent.compare(prompt=prompt, rubrics=rubrics, first=second, second=first, feedback=feedback),
        )
        return PairComparison(index_i, index_j, forward, reverse)

    return await asyncio.gather(*(
        compare_pair(index_i, index_j)
        for index_i, index_j in itertools.combinations(range(len(rubrics)), 2)
    ))


def solve_comparisons(
    comparisons: list[PairComparison], rubric_ids: list[str], alpha: float
) -> tuple[dict[str, float], float]:
    """Map agreed winners to a reciprocal matrix and solve it."""
    matrix = np.ones((len(rubric_ids), len(rubric_ids)), dtype=float)
    for pair in comparisons:
        ratio = 1.0
        if pair.winner_id == rubric_ids[pair.index_i]:
            ratio = alpha
        elif pair.winner_id == rubric_ids[pair.index_j]:
            ratio = 1.0 / alpha
        matrix[pair.index_i, pair.index_j] = ratio
        matrix[pair.index_j, pair.index_i] = 1.0 / ratio
    return solve_ahp_weights(matrix, rubric_ids)


async def compute_weights(
    *, prompt: str, rubrics: list[Rubric], agent: PairCompareAgent, alpha: float, tau: float
) -> dict[str, float]:
    """Return the retained weights after at most one complete review."""
    if not math.isfinite(alpha) or alpha <= 1:
        raise ValueError("alpha must be finite and greater than one")
    if not math.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and nonnegative")
    rubric_ids = [rubric.rubric_id for rubric in rubrics]
    if not rubric_ids or len(set(rubric_ids)) != len(rubric_ids):
        raise ValueError("rubric set must be nonempty with unique IDs")
    if len(rubrics) == 1:
        return {rubric_ids[0]: 1.0}
    initial = await collect_comparisons(prompt=prompt, rubrics=rubrics, agent=agent)
    weights, initial_cr = solve_comparisons(initial, rubric_ids, alpha)
    if initial_cr > tau:
        feedback = agent.review_feedback(
            json.dumps([pair.to_history() for pair in initial], ensure_ascii=False, indent=2)
        )
        revised = await collect_comparisons(
            prompt=prompt, rubrics=rubrics, agent=agent, feedback=feedback
        )
        revised_weights, revised_cr = solve_comparisons(revised, rubric_ids, alpha)
        if revised_cr < initial_cr:
            weights = revised_weights
    return weights
