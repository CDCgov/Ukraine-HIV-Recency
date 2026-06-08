# -*- coding: utf-8 -*-
"""
Simulation study for the HIV recent-infection hotspot detector.

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
The pipeline flags a territory as a hotspot when the posterior probability that
its standardized recency ratio exceeds an epidemiological threshold (here the
Standardized Morbidity-style Ratio SMR > 2.0, i.e. twice the current national
recency proportion) is itself above an automatically chosen cutoff. That cutoff
is selected to control the Bayesian false discovery rate (FDR) among the flagged
territories. This script reproduces that rule and checks whether the realised
FDR matches the nominal target and how much true signal is recovered.

Why it is self-contained
------------------------
The per-territory recency probability is summarised in closed form using the
Beta-Binomial conjugacy: a Beta prior combined with Binomial recent/ tested
counts yields a Beta posterior. Exceedance probabilities are then exact Beta
tail areas, so thousands of replicates run in seconds without MCMC. The
detection logic imports the pipeline's own FDR-controlled exceedance
threshold (``pipeline.standardization.bayesian_fdr_threshold``), so the
simulated and production rules are guaranteed identical.

Note on the measured quantity: like the pipeline, this validates the *proportion
of recent infections among the newly diagnosed*, not incidence.

Run:
    python simulation_validation.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.stats import beta as beta_dist

# Project root on sys.path so this script runs from the validation/ subfolder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Use the *production* FDR-controlled cutoff rule rather than a private copy,
# so the simulated detector and the pipeline cannot drift apart. The function
# tightens the exceedance cutoff upward from 0.95 while the Bayesian FDR stays
# within target, then relaxes to a 0.70 floor; see its docstring.
from pipeline.standardization import bayesian_fdr_threshold


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
    smr_threshold: float = 2.0  # SMR cutoff defining an epidemiological hotspot


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


def simulate_window(scn: Scenario, rng: np.random.Generator) -> Tuple[float, float, float, int]:
    """
    Simulate one analysis window and apply the detection rule.

    Steps:
      1. Assign a true recency proportion to each territory: hotspots sit at
         national * hotspot_multiplier, the rest at the national proportion.
      2. Draw the number of tests per territory (Poisson around avg_tests) and
         the observed recent count ~ Binomial(tests, true_prop).
      3. Form the Beta posterior for each territory and compute the exceedance
         probability P(p_i > smr_threshold * national_prop) -- the posterior
         probability that the SMR exceeds the epidemiological threshold.
      4. Pick the FDR-controlled cutoff and flag territories above it.
      5. Compare flags with the known hotspot labels.

    Returns (sensitivity, specificity, empirical_fdr, n_flagged).
    """
    n = scn.n_territories
    n_hot = max(1, int(round(scn.frac_hotspots * n)))
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

    # Exceedance probability that the SMR exceeds the threshold, i.e. that the
    # true recency proportion exceeds smr_threshold * national. 1 - CDF is the
    # upper tail of the Beta posterior.
    ref = min(scn.smr_threshold * scn.national_prop, 0.999)
    exc = 1.0 - beta_dist.cdf(ref, post_a, post_b)

    cutoff, _ = bayesian_fdr_threshold(exc)
    flagged = exc > cutoff

    tp = int(np.sum(flagged & is_hotspot))
    fp = int(np.sum(flagged & ~is_hotspot))
    fn = int(np.sum(~flagged & is_hotspot))
    tn = int(np.sum(~flagged & ~is_hotspot))

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    emp_fdr = fp / (tp + fp) if (tp + fp) else 0.0
    return sensitivity, specificity, emp_fdr, tp + fp


def run_scenario(scn: Scenario, n_reps: int, seed: int) -> dict:
    """Average the operating characteristics over many simulated windows."""
    rng = np.random.default_rng(seed)
    sens, spec, fdr, flagged = [], [], [], []
    for _ in range(n_reps):
        s, sp, f, nf = simulate_window(scn, rng)
        sens.append(s)
        spec.append(sp)
        fdr.append(f)
        flagged.append(nf)
    return {
        'scenario': scn.name,
        'sensitivity': np.nanmean(sens),
        'specificity': np.nanmean(spec),
        'empirical_fdr': np.mean(fdr),
        'mean_flagged': np.mean(flagged),
    }


def main():
    n_reps = 500
    seed = 42

    # Scenarios span the regimes seen in the real data: low vs higher national
    # recency, sparse vs well-sampled territories, and moderate vs strong true
    # elevation. frac_hotspots is kept small, as real hotspots are rare.
    scenarios = [
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

    print("=" * 100)
    print(f"Hotspot detector simulation  |  {n_reps} replicates per scenario  "
          f"|  SMR threshold = 2.0  |  nominal FDR = 0.05")
    print("=" * 100)
    header = f"{'scenario':<46}{'sensitivity':>13}{'specificity':>13}{'emp.FDR':>10}{'flagged':>10}"
    print(header)
    print("-" * 100)
    for r in results:
        print(f"{r['scenario']:<46}{r['sensitivity']:>13.3f}{r['specificity']:>13.3f}"
              f"{r['empirical_fdr']:>10.3f}{r['mean_flagged']:>10.2f}")
    print("-" * 100)
    print("Reading the table:")
    print("  - empirical FDR should sit at or below the 0.05 nominal target;")
    print("  - sensitivity falls when tests are sparse or the effect is small")
    print("    (small numerators give wide posteriors and few confident calls);")
    print("  - in the null scenario almost nothing should be flagged.")


if __name__ == '__main__':
    main()
