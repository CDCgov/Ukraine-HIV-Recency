"""
Bayesian-FDR-controlled threshold for posterior exceedance probabilities.

A territory is flagged when its posterior probability of exceeding the
reference (national rate, SMR threshold, SIR threshold, ...) is above some
cutoff t. The Bayesian false discovery rate at that cutoff is the mean of
``1 - exceedance_prob`` over the flagged territories -- the expected
fraction of false positives among the discoveries.

``bayesian_fdr_threshold`` picks t so that the FDR sits at or below a
target (default 5%), preferring the highest cutoff that achieves it
(fewer, higher-confidence calls) and refusing to drop below a floor
where the evidence is too weak to be useful.
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

        1. Tighten the cutoff upward from ``start`` while the FDR stays
           within ``max_fdr`` -- prefer fewer, higher-confidence calls.
        2. If nothing passes ``start``, relax the cutoff downward to
           recover at least one territory, but never below ``floor``.

    Below ``floor`` (default 0.70) the posterior evidence is too weak to
    justify calling a territory a hotspot -- a 0.55 cutoff would flag near
    coin-flip signals -- so when no cutoff meets the FDR target down to the
    floor we report no hotspots rather than emit unreliable ones.

    Args:
        exc_probs: Per-territory exceedance probabilities.
        max_fdr: Target false discovery rate.
        start: Initial (and preferred) cutoff.
        step: Search increment.
        ceiling: Maximum cutoff considered.
        floor: Minimum cutoff considered.

    Returns:
        ``(cutoff, achieved_fdr)``. When no cutoff satisfies the
        constraint, returns ``(start, 0.0)``, meaning "no hotspots in this
        window".
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
