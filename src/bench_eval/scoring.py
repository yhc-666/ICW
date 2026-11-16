"""Weighted response scores and half-credit preference accuracy."""

import math


TIE_TOLERANCE = 1e-12


def score_responses(
    weights: dict[str, float], vector_a: dict[str, int], vector_b: dict[str, int]
) -> dict:
    """Score two responses using exactly the same normalized rubric weights."""
    if not weights or set(weights) != set(vector_a) or set(weights) != set(vector_b):
        raise ValueError("weights and both verifier vectors must have identical rubric IDs")
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights.values()):
        raise ValueError("weights must be positive and finite")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("weights must sum to one")
    for vector in (vector_a, vector_b):
        if any(type(bit) is not int or bit not in {0, 1} for bit in vector.values()):
            raise ValueError("verifier vectors must contain integer zero or one")
    score_a = sum(weight * vector_a[rubric_id] for rubric_id, weight in weights.items())
    score_b = sum(weight * vector_b[rubric_id] for rubric_id, weight in weights.items())
    margin = score_a - score_b
    if abs(margin) <= TIE_TOLERANCE:
        prediction = "tie"
    else:
        prediction = "A" if margin > 0 else "B"
    return {"score_a": score_a, "score_b": score_b, "prediction": prediction}


def preference_credit(prediction: str, preferred: str) -> float:
    """Give half credit to a tied prediction."""
    if prediction not in {"A", "B", "tie"} or preferred not in {"A", "B"}:
        raise ValueError("invalid preference")
    return 0.5 if prediction == "tie" else float(prediction == preferred)
