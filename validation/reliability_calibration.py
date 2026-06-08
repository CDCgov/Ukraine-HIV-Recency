# -*- coding: utf-8 -*-
"""
Calibration study for the per-territory reliability score (audit M5).

The pipeline rates each territory HIGH / MODERATE / LOW from a coefficient of
variation of its SMR posterior:

    cv    = (SMR_upper - SMR_lower) / (3.92 * SMR_mean)
    score = 100 * exp(-cv)
    HIGH >= 80, MODERATE >= 60, else LOW.

Those cut-offs (80 / 60) were originally picked by hand. This script grounds
them: it asks, across realistic regimes of sample size and true effect, how
the score relates to an objective, decision-relevant quantity --- whether the
95% credible interval is *decisive* about the SMR > 2 hotspot threshold (the
interval lies entirely above or entirely below 2, so the territory can be
acted on rather than being "inconclusive").

It is closed-form (Beta-Binomial conjugacy, the same shrinkage the pipeline
uses), so thousands of replicates run in a second without MCMC. Output: a
score / CI-width / decisiveness table by (true SMR, tests), and the mean
decisiveness within each reliability tier --- the evidence for keeping or
moving the cut-offs.

Run:
    python validation/reliability_calibration.py
    python validation/reliability_calibration.py --national-rate 0.02 --prior-k 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

# Project root on sys.path so this script runs from the validation/ subfolder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def reliability_cell(n_tests: int, true_smr: float, national_rate: float,
                     prior_k: float, reps: int, rng: np.random.Generator):
    """Closed-form reliability for one (true SMR, n) cell.

    Mirrors the pipeline: a Beta(prior) centred on the national rate is
    updated by Binomial recent/tested counts; the SMR posterior is the rate
    posterior divided by the national rate. Returns the mean reliability
    score, the mean SMR CI width, and the fraction of replicates whose 95% CI
    is decisive about the SMR = 2 threshold.
    """
    p_true = min(national_rate * true_smr, 0.99)
    recent = rng.binomial(n_tests, p_true, size=reps)

    a0 = national_rate * prior_k
    b0 = (1.0 - national_rate) * prior_k
    a = a0 + recent
    b = b0 + (n_tests - recent)

    p_mean = a / (a + b)
    p_lo = beta_dist.ppf(0.025, a, b)
    p_hi = beta_dist.ppf(0.975, a, b)

    cv = (p_hi - p_lo) / (3.92 * p_mean)
    score = 100.0 * np.exp(-cv)

    smr_lo = p_lo / national_rate
    smr_hi = p_hi / national_rate
    decisive = (smr_lo > 2.0) | (smr_hi < 2.0)

    smr_ci_width = (p_hi - p_lo) / national_rate
    return score, smr_ci_width, decisive


def main() -> None:
    parser = argparse.ArgumentParser(description="Reliability-score calibration study")
    parser.add_argument('--national-rate', type=float, default=0.02,
                        help='National recency proportion (default 0.02, ~ the project level)')
    parser.add_argument('--prior-k', type=float, default=20.0,
                        help='EB prior concentration K (default 20, the pipeline default)')
    parser.add_argument('--reps', type=int, default=4000, help='Replicates per cell')
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    tests_grid = [10, 20, 30, 50, 100, 200, 500]
    smr_grid = [1.0, 2.0, 3.0, 4.0]

    print("=" * 92)
    print(f"Reliability-score calibration  |  national_rate={args.national_rate}  "
          f"prior K={args.prior_k}  |  {args.reps} reps/cell")
    print("Score = 100*exp(-CV) of the SMR posterior; tiers HIGH>=80, MODERATE>=60, LOW<60.")
    print("Decisive = 95% CI lies entirely above or below the SMR=2 hotspot threshold.")
    print("=" * 92)
    header = f"{'true SMR':>9}{'tests':>8}{'mean score':>12}{'tier':>10}{'SMR CI width':>14}{'decisive':>10}"
    print(header)
    print("-" * 92)

    # Collect (score, decisive) pairs across all replicates for the tier summary.
    all_scores = []
    all_decisive = []

    for smr in smr_grid:
        for n in tests_grid:
            score, ci_w, dec = reliability_cell(n, smr, args.national_rate,
                                                args.prior_k, args.reps, rng)
            ms = float(score.mean())
            tier = "HIGH" if ms >= 80 else ("MODERATE" if ms >= 60 else "LOW")
            print(f"{smr:>9.1f}{n:>8}{ms:>12.1f}{tier:>10}{ci_w.mean():>14.2f}{dec.mean():>10.2f}")
            all_scores.append(score)
            all_decisive.append(dec)
        print("-" * 92)

    scores = np.concatenate(all_scores)
    decisive = np.concatenate(all_decisive)

    print("\nDecisiveness within each reliability tier (pooled over the grid):")
    for tier, lo, hi in [("HIGH", 80, 100.01), ("MODERATE", 60, 80), ("LOW", 0, 60)]:
        mask = (scores >= lo) & (scores < hi)
        frac = mask.mean()
        dec = decisive[mask].mean() if mask.any() else float('nan')
        print(f"  {tier:<9}: {frac*100:5.1f}% of cells   mean decisiveness = {dec:.2f}")

    print("\nReading the table:")
    print("  - the score rises monotonically with sample size, as it should;")
    print("  - if HIGH cells are decisive far more often than LOW cells, the 80/60")
    print("    cut-offs separate actionable from non-actionable estimates well;")
    print("  - at this national rate the recent-event counts are tiny, so HIGH may")
    print("    require hundreds of tests — a property of the data, not the cut-offs.")


if __name__ == '__main__':
    main()
