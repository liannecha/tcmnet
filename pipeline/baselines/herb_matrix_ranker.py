"""Non-neural herb ranking baseline for comparison with the active recommender.

This module preserves the original matrix/formula-based herb recommender for
ablation tests and metric comparisons. Production inference now ranks herbs
with the neural herb head.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_RECOMMENDATION_ALPHA = 0.7
HERB_RANKING_FORMULA = (
    "score = alpha * concept_similarity + (1 - alpha) * syndrome_prior"
)


@dataclass(frozen=True)
class HerbRankingResult:
    """Numerical outputs from the non-neural herb ranking baseline."""

    indices: np.ndarray
    scores: np.ndarray
    concept_similarity: np.ndarray
    syndrome_prior: np.ndarray


def rank_herbs_by_matrix_baseline(
    concept_scores: np.ndarray,
    pred_syndrome_idx: int,
    herb_concept_matrix: np.ndarray,
    syndrome_herb_prior: np.ndarray,
    top_k: int,
    alpha: float = DEFAULT_RECOMMENDATION_ALPHA,
) -> HerbRankingResult:
    """Rank herbs with the preserved non-neural matrix blending formula."""
    concept_similarity = (herb_concept_matrix @ concept_scores) / max(
        len(concept_scores), 1
    )
    prior = syndrome_herb_prior[pred_syndrome_idx]
    scores = alpha * concept_similarity + (1.0 - alpha) * prior

    used_for_syndrome = prior > 0
    if used_for_syndrome.any():
        candidate_indices = np.where(used_for_syndrome)[0]
    else:
        candidate_indices = np.arange(herb_concept_matrix.shape[0])

    ranked_indices = candidate_indices[
        np.argsort(scores[candidate_indices])[::-1]
    ][:top_k]
    return HerbRankingResult(
        indices=ranked_indices,
        scores=scores,
        concept_similarity=concept_similarity,
        syndrome_prior=prior,
    )
