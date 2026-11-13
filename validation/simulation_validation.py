# -*- coding: utf-8 -*-
"""
Simulation study for the HIV recent-infection hotspot detector (level axis).

Purpose
-------
Quantify the operating characteristics of the detection rule -- sensitivity,
specificity and the empirical (realised) false discovery rate -- on synthetic
data with KNOWN injected hotspots. This is the validation a methods reviewer
expects: it shows the rule recovers a known ground truth and reports its error
rates across realistic scenarios (small samples, different effect sizes,
different national recency levels).

What is being validated
------------------------
The production rule for the LEVEL axis ("elevated vs national"): a territory is
elevated when the posterior probability that its recency proportion exceeds
the leave-one-out national proportion times ``detection.smr_threshold``
(parity, 1.0) is above a cutoff. The cutoff is the pipeline's own
``bayesian_fdr_threshold`` called exactly as ``_finalize_classification`` calls
it: ``start = floor = detection.confidence_level`` (0.80, the Richardson et al.
2004 D(0.8, 1) rule), raised only when a stricter cutoff brings the Bayesian
FDR within 5%. A hotspot must also pass the presence gate
(``MIN_HOTSPOT_CURRENT_EVENTS`` recent events in the window). Thresholds are
read from ``config.json`` and the gate from ``taxonomy.py``, so the simulated
and production rules cannot drift apart.

The trend axis and the joint two-period model are validated separately
(``two_period_validation.py``, ``threshold_simulation.py``); the full label
taxonomy by ``taxonomy_validation.py``.

Why it is self-contained
------------------------
The per-territory recency probability is summarised in closed form using the
Beta-Binomial conjugacy: a Beta prior centred on the national proportion
combined with Binomial recent/tested counts yields a Beta posterior.
Exceedance probabilities are exact Beta tail areas, so thousands of replicates
run in seconds without MCMC. The fixed Beta prior is a stand-in for the
production hierarchical prior and is deliberately conservative: when there is
no real between-territory variation the hierarchical model learns that and
shrinks every territory to the national rate, whereas a fixed prior cannot, so
this script OVERSTATES false positives (about 5-7% of null territories flagged
per window here, against about 1% for the full two-period model in
``threshold_simulation.py``; a stronger prior narrows the gap only partly).
Read its numbers as an upper bound on false calls and as the shape of the
sensitivity/sample-size trade-off; the operating characteristics of the
production model come from ``threshold_simulation.py``.

Note on the measured quantity: like the pipeline, this validates the *proportion
of recent infections among the newly diagnosed*, not incidence.

Run:
    python validation/simulation_validation.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.stats import beta as beta_dist

# Project root on sys.path so this script runs from the validation/ subfolder.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.classification.taxonomy import MIN_HOTSPOT_CURRENT_EVENTS  # noqa: E402
from pipeline.standardization import bayesian_fdr_threshold  # noqa: E402

_DETECTION = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))['detection']
SMR_THRESHOLD = float(_DETECTION['smr_threshold'])
CONFIDENCE = float(_DETECTION['confidence_level'])


# ----------------------------------------------------------------------------
# Scenario definition and one simulated window
# ----------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    n_territories: int          # number of territories per window
    national_prop: float        # current national recency proportion
    avg_tests: int              # mean number of tests per territory
    hotspot_multiplier: float   # true recency = national * this, for hotspots
    frac_hotspots: float        # fraction of territories that are true hotspots


def _beta_prior(national_prop: float, prior_strength: float = 20.0) -> Tuple[float, float]:
    """
    Weakly informative Beta prior centred on the national recency proportion.

    prior_strength is the prior "pseudo-count" (a + b): larger values pull small
    territories more strongly toward the national mean (empirical-Bayes style
    shrinkage), which stabilises estimates where few people were tested.
    """
    a0 = national_prop * prior_strength
    b0 = (1.0 - national_prop) * prior_strength
    return a0, b0


def simulate_window(scn: Scenario, rng: np.random.Generator) -> Tuple[float, float, float, int, bool]:
    """
    Simulate one analysis window and apply the production level-axis rule.

    Steps:
      1. Assign a true recency proportion to each territory: hotspots sit at
         national * hotspot_multiplier, the rest at the national proportion.
      2. Draw the number of tests per territory (Poisson around avg_tests) and
         the observed recent count ~ Binomial(tests, true_prop).
      3. Form the Beta posterior for each territory and the leave-one-out
         national proportion from the other territories' observed counts.
      4. Exceedance probability P(p_i > SMR_THRESHOLD * national_loo_i).
      5. Cutoff from ``bayesian_fdr_threshold(start=floor=CONFIDENCE)``; flag
         territories above it that also pass the presence gate.
      6. Compare flags with the known hotspot labels.

    Returns (sensitivity, specificity, empirical_fdr, n_flagged, fdr_met),
    where fdr_met says whether a cutoff meeting the 5% FDR target was found.
    """
    n = scn.n_territories
    n_hot = int(round(scn.frac_hotspots * n))
    if scn.frac_hotspots > 0:
        n_hot = max(1, n_hot)
    is_hotspot = np.zeros(n, dtype=bool)
    is_hotspot[:n_hot] = True
    rng.shuffle(is_hotspot)

    true_prop = np.where(is_hotspot,
                         np.minimum(scn.national_prop * scn.hotspot_multiplier, 0.99),
                         scn.national_prop)

    # Tests per territory: at least 1, Poisson-dispersed around the mean.
    tests = np.maximum(1, rng.poisson(scn.avg_tests, size=n))
    recent = rng.binomial(tests, true_prop)

    # Beta posterior per territory (conjugate update of the shared prior).
    a0, b0 = _beta_prior(scn.national_prop)
    post_a = a0 + recent
    post_b = b0 + (tests - recent)

    # Leave-one-out national proportion, floored as in compute_smr_sir.
    loo = (recent.sum() - recent) / np.maximum(tests.sum() - tests, 1)
    ref = np.minimum(np.maximum(loo, 1e-3) * SMR_THRESHOLD, 0.999)
    exc = 1.0 - beta_dist.cdf(ref, post_a, post_b)

    cutoff, achieved = bayesian_fdr_threshold(exc, start=CONFIDENCE, floor=CONFIDENCE)
    fdr_met = achieved > 0 or not (exc > cutoff).any()
    flagged = (exc > cutoff) & (recent >= MIN_HOTSPOT_CURRENT_EVENTS)

    tp = int(np.sum(flagged & is_hotspot))
    fp = int(np.sum(flagged & ~is_hotspot))
    fn = int(np.sum(~flagged & is_hotspot))
    tn = int(np.sum(~flagged & ~is_hotspot))

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    emp_fdr = fp / (tp + fp) if (tp + fp) else 0.0
    return sensitivity, specificity, emp_fdr, tp + fp, fdr_met


def run_scenario(scn: Scenario, n_reps: int, seed: int) -> dict:
    """Average the operating characteristics over many simulated windows."""
    rng = np.random.default_rng(seed)
    sens, spec, fdr, flagged, met = [], [], [], [], []
    for _ in range(n_reps):
        s, sp, f, nf, m = simulate_window(scn, rng)
        sens.append(s)
        spec.append(sp)
        fdr.append(f)
        flagged.append(nf)
        met.append(m)
    return {
        'scenario': scn.name,
        'sensitivity': np.nanmean(sens) if not np.all(np.isnan(sens)) else np.nan,
        'specificity': np.nanmean(spec),
        'empirical_fdr': np.mean(fdr),
        'mean_flagged': np.mean(flagged),
        'fdr_met': np.mean(met),
    }


def main():
    n_reps = 500
    seed = 42

    # Scenarios span the regimes seen in the real data: low vs higher national
    # recency, sparse vs well-sampled territories, and moderate vs strong true
    # elevation. frac_hotspots is kept small, as real hotspots are rare. The
    # "real-data-like" row mirrors an H3 res3 window: ~27 active cells, ~65
    # tests per cell, national recency ~1.3%, one strongly elevated cell.
    scenarios = [
        Scenario("real-data-like (27 cells, 1 hotspot x4)",
                 n_territories=27, national_prop=0.013, avg_tests=65,
                 hotspot_multiplier=4.0, frac_hotspots=1 / 27),
        Scenario("low national, sparse tests, strong effect",
                 n_territories=35, national_prop=0.01, avg_tests=25,
                 hotspot_multiplier=4.0, frac_hotspots=0.1),
        Scenario("low national, sparse tests, moderate effect",
                 n_territories=35, national_prop=0.01, avg_tests=25,
                 hotspot_multiplier=2.5, frac_hotspots=0.1),
        Scenario("higher national, sparse tests, strong effect",
                 n_territories=35, national_prop=0.04, avg_tests=25,
                 hotspot_multiplier=4.0, frac_hotspots=0.1),
        Scenario("higher national, well sampled, strong effect",
                 n_territories=35, national_prop=0.04, avg_tests=150,
                 hotspot_multiplier=4.0, frac_hotspots=0.1),
        Scenario("null (no true hotspots)",
                 n_territories=35, national_prop=0.02, avg_tests=50,
                 hotspot_multiplier=1.0, frac_hotspots=0.0),
    ]

    results = [run_scenario(s, n_reps, seed) for s in scenarios]

    print("=" * 108)
    print(f"Hotspot detector simulation (level axis)  |  {n_reps} replicates per scenario  "
          f"|  SMR threshold = {SMR_THRESHOLD:g} (LOO)  |  confidence = {CONFIDENCE:.2f}  "
          f"|  presence gate >= {MIN_HOTSPOT_CURRENT_EVENTS}")
    print("=" * 108)
    header = (f"{'scenario':<46}{'sensitivity':>13}{'specificity':>13}{'emp.FDR':>10}"
              f"{'flagged':>10}{'FDR<=5% met':>14}")
    print(header)
    print("-" * 108)
    for r in results:
        print(f"{r['scenario']:<46}{r['sensitivity']:>13.3f}{r['specificity']:>13.3f}"
              f"{r['empirical_fdr']:>10.3f}{r['mean_flagged']:>10.2f}{r['fdr_met']:>14.2f}")
    print("-" * 108)
    print("Reading the table:")
    print("  - 'FDR<=5% met' is the share of windows where a cutoff >= the confidence level")
    print("    achieved the 5% Bayesian FDR target; otherwise the rule falls back to the")
    print("    plain confidence-level cutoff and the realised FDR can exceed 5%;")
    print("  - sensitivity falls when tests are sparse or the effect is small")
    print("    (small numerators give wide posteriors and few confident calls);")
    print("  - in the null scenario few territories should be flagged; the fixed prior")
    print("    used here overstates false positives relative to the hierarchical model")
    print("    (see the module docstring and threshold_simulation.py).")


if __name__ == '__main__':
    main()
