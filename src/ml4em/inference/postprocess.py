"""
Postprocessing — convert raw model probabilities to Candidate results.

This module is the only fully implemented piece of the inference layer.
It has no dependency on model internals — only on types and config.

Design
------
probabilities_to_candidates() maps each (FeatureVector, probability) pair
to a Candidate by:
  1. Assigning a confidence tier ("high" / "medium" / "low") using the
     thresholds defined in InferenceConfig.confidence_thresholds.
  2. Copying identity fields (source_id, ra, dec, survey) from FeatureVector.
  3. Copying period and period_algorithm from FeatureVector (nan / "" if
     the period extractor did not run or failed).
  4. Attaching any extra metadata passed by the caller.

The result type (Candidate) is generic — this module does not assume any
science case.  The caller decides what FeatureVectors mean and what
the model's positive class represents.
"""

from __future__ import annotations

import numpy as np

from ml4em.config.schema import InferenceConfig
from ml4em.types import Candidate, Confidence, FeatureVector


def probabilities_to_candidates(
    features: list[FeatureVector],
    probabilities: np.ndarray,
    config: InferenceConfig,
) -> list[Candidate]:
    """Convert raw model probabilities to Candidate results.

    Parameters
    ----------
    features:
        One FeatureVector per source, in the same order as probabilities.
    probabilities:
        Shape (N,), dtype float — P(positive class) in [0, 1] for each source.
        Output of MLModel.predict_proba().
    config:
        InferenceConfig from PipelineConfig.inference.
        confidence_thresholds maps tier names to probability cutoffs.

    Returns
    -------
    list[Candidate]
        One Candidate per source, in the same order as input.

    Raises
    ------
    ValueError
        If len(features) != len(probabilities).
    """
    if len(features) != len(probabilities):
        raise ValueError(
            f"features and probabilities must have the same length. "
            f"Got {len(features)} features and {len(probabilities)} probabilities."
        )

    high_thresh   = config.confidence_thresholds["high"]
    medium_thresh = config.confidence_thresholds["medium"]

    candidates: list[Candidate] = []
    for fv, prob in zip(features, probabilities):
        confidence = _assign_confidence(float(prob), high_thresh, medium_thresh)
        candidates.append(
            Candidate(
                source_id        = fv.source_id,
                ra               = fv.ra,
                dec              = fv.dec,
                survey           = fv.survey,
                probability      = float(prob),
                period           = float(fv.period),
                period_algorithm = fv.period_algorithm,
                confidence       = confidence,
            )
        )

    return candidates


def _assign_confidence(
    probability: float,
    high_thresh: float,
    medium_thresh: float,
) -> Confidence:
    """Map a probability to a confidence tier.

    Parameters
    ----------
    probability:
        Model output in [0, 1].
    high_thresh:
        Minimum probability for "high" confidence.
    medium_thresh:
        Minimum probability for "medium" confidence.

    Returns
    -------
    Confidence
        "high" if probability >= high_thresh,
        "medium" if probability >= medium_thresh,
        "low" otherwise.
    """
    if probability >= high_thresh:
        return "high"
    if probability >= medium_thresh:
        return "medium"
    return "low"
