"""Convert review scores into individual quality-moderation request bodies."""

import math

from .dataset import CRITERIA
from .listing_features import LEARNED_CRITERIA


def moderation_payloads(review: dict, pass_threshold: float = 0.5) -> list[dict]:
    if not math.isfinite(pass_threshold) or not 0 <= pass_threshold <= 1:
        raise ValueError("Pass threshold must be between 0 and 1")
    payloads = []
    # These are the supported human-reviewed targets. Automatic/backend-owned
    # rules and unrelated criteria must never become moderation writes.
    for key in (*CRITERIA, *LEARNED_CRITERIA):
        result = review["criteria"].get(key, {})
        if "unavailable" in result or "pass_score" not in result:
            continue
        score = result["pass_score"]
        if (
            type(score) not in (int, float)
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError(f"Invalid pass score for {key}: {score!r}")
        payloads.append({"guideline_id": key, "passed": score >= pass_threshold})
    return payloads
