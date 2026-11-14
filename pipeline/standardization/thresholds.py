"""
Bayesian-FDR-controlled threshold for posterior exceedance probabilities.

A territory is flagged when its posterior probability of exceeding the
reference (national rate, SMR threshold, SIR threshold, ...) is above some
cutoff t. The Bayesian false discovery rate at that cutoff is the mean of
``1 - exceedance_prob`` over the flagged territories -- the expected
fraction of false positives among the discoveries.

``bayesian_fdr_threshold`` picks the lowest cutoff t >= ``start`` at
which the FDR sits at or below a target (default 5%), relaxing below
``start`` only down to a floor where the evidence is still useful. When no
cutoff meets the target it returns ``start`` itself, and the caller then
flags everything above ``start`` without FDR control. The pipeline passes
``start = floor = confidence_level`` (0.80), so in practice the rule is the
Richardson et al. (2004) D(0.8, 1) cut-off, raised only when a stricter
cutoff achieves the FDR target.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def bayesian_fdr_threshold(exc_probs: np.ndarray,
                           max_fdr: float = 0.05,
                           start: float = 0.95, step: float = 0.005,
                           ceiling: float = 0.99,
                           floor: float = 0.70) -> Tuple[float, float]:
    """
    Pick the exceedance-probability cutoff that controls the Bayesian FDR.

    Search strategy:

        1. Raise the cutoff upward from ``start`` and return the first
           (lowest) one at which the FDR of the selected territories is
           within ``max_fdr``.
        2. If no cutoff from ``start`` up to ``ceiling`` meets the target
           (or nobody is selected), relax downward from ``start`` to
           recover at least one territory, but never below ``floor``.

    Below ``floor`` (default 0.70) the posterior evidence is too weak to
    justify calling a territory a hotspot -- a 0.55 cutoff would flag near
    coin-flip signals.

    Args:
        exc_probs: Per-territory exceedance probabilities.
        max_fdr: Target false discovery rate.
        start: Initial (and preferred) cutoff.
        step: Search increment.
        ceiling: Maximum cutoff considered.
        floor: Minimum cutoff considered.

    Returns:
        ``(cutoff, achieved_fdr)``. When no cutoff satisfies the
        constraint, returns ``(start, 0.0)``: the caller still flags the
        territories above ``start``, but the FDR target was not met and the
        0.0 is a placeholder, not an achieved FDR.
    """
    # Phase 1: tighten upward.
    threshold = start
    while threshold <= ceiling:
        selected = exc_probs[exc_probs > threshold]
        if len(selected) == 0:
            break  # nobody passes this cutoff -- relax instead
        fdr = float(np.mean(1 - selected))
        if fdr <= max_fdr:
            return threshold, fdr
        threshold += step

    # Phase 2: relax downward to the floor to recover at least one
    # territory while still meeting the FDR target.
    threshold = start - step
    while threshold >= floor:
        selected = exc_probs[exc_probs > threshold]
        if len(selected) > 0:
            fdr = float(np.mean(1 - selected))
            if fdr <= max_fdr:
                return threshold, fdr
        threshold -= step

    # Nothing meets the FDR target down to the floor: report no hotspots.
    return start, 0.0
