# -*- coding: utf-8 -*-
"""
Simulation comparison: joint two-period model vs the two-step SIR trend axis.

Purpose
-------
The pipeline detects a *trend* (a territory rising versus its own past) in two
ways:

  * the legacy two-step SIR: fit the current window, then take a ratio against a
    separately empirical-Bayes-shrunk history;
  * the joint two-period model (``BayesianAnalyzer.run_two_period_model``):
    fit both windows at once and read the trend directly from a hierarchical
    per-territory current-period change ``delta_i``.

This script simulates two-period data with a KNOWN trend on a fraction of
territories and measures how well each method recovers it -- sensitivity,
specificity and the realised false discovery rate -- across scenarios that span
the sparse-event regime of the real data. It is the evidence a methods reviewer
needs before the two-period model replaces the SIR axis as the default.

Design
------
For tractability the two-period model is fit once per scenario on one large
simulated dataset (a single MCMC run), and the SIR baseline is computed on the
SAME data in closed form via the pipeline's own ``compute_smr_sir`` /
``bayesian_fdr_threshold`` helpers, so the two rules cannot drift from
production. The trend "truth" is whether a territory's current rate was raised
above its history by at least the flagged multiplier.

Run:
    python validation/two_period_validation.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline.bootstrap  # noqa: F401 -- env/warning setup before heavy imports

import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from pipeline.standardization import bayesian_fdr_threshold
from pipeline.standardization.smr_sir import compute_smr_sir

SIR_THRESHOLD = 1.5  # a "trend" is a >=1.5x current-vs-history rise
LOG_SIR = float(np.log(SIR_THRESHOLD))


@dataclass
class Scenario:
    name: str
    n_territories: int
    national_prop: float      # baseline (history) national recency proportion
    avg_tests: int            # mean tests per territory per window
    trend_multiplier: float   # true current = history * this, for trend territories
    frac_trend: float         # fraction of territories with a true rise


def simulate_two_period(scn: Scenario, rng: np.random.Generator) -> pd.DataFrame:
    """One simulated dataset: history + current counts with a known trend flag."""
    n = scn.n_territories
    n_trend = int(round(scn.frac_trend * n))
    is_trend = np.zeros(n, dtype=bool)
    is_trend[:n_trend] = True
    rng.shuffle(is_trend)

    # Territory baseline rate: national proportion with lognormal territory
    # variation, so the intercept a_i is non-trivial to recover.
    base = scn.national_prop * rng.lognormal(mean=0.0, sigma=0.4, size=n)
    base = np.clip(base, 1e-4, 0.5)
    curr = np.where(is_trend, np.clip(base * scn.trend_multiplier, 1e-4, 0.8), base)

    n_h = np.maximum(1, rng.poisson(scn.avg_tests, size=n))
    y_h = rng.binomial(n_h, base)
    n_c = np.maximum(1, rng.poisson(scn.avg_tests, size=n))
    y_c = rng.binomial(n_c, curr)

    return pd.DataFrame({
        'recent_count_hist': y_h, 'all_tested_hist': n_h,
        'recent_count_curr': y_c, 'all_tested_curr': n_c,
        'is_trend': is_trend,
    })


def fit_two_period_delta(df: pd.DataFrame, national_rate: float,
                         seed: int) -> np.ndarray:
    """Fit the joint two-period model and return P(delta_i > log(1.5)) per territory.

    This mirrors ``BayesianAnalyzer.run_two_period_model`` in compact form (a
    lightweight sampler configuration for the simulation).
    """
    y = df['recent_count_curr'].to_numpy().astype(int)
    n = df['all_tested_curr'].to_numpy().astype(int)
    y_h = df['recent_count_hist'].to_numpy().astype(int)
    n_h = df['all_tested_hist'].to_numpy().astype(int)
    n_terr = len(df)
    clip = float(np.clip(national_rate, 1e-3, 0.5))
    prior_mu = float(np.log(clip / (1.0 - clip)))

    with pm.Model():
        mu_hist = pm.Normal('mu_hist', mu=prior_mu, sigma=1.5)
        mu_curr = pm.Normal('mu_curr', mu=prior_mu, sigma=1.5)
        sigma_a = pm.HalfNormal('sigma_a', sigma=2)
        mu_delta = pm.Normal('mu_delta', mu=0.0, sigma=1.0)
        sigma_delta = pm.HalfNormal('sigma_delta', sigma=1.0)
        a = pm.Deterministic('a', sigma_a * pm.Normal('a_off', 0, 1, shape=n_terr))
        delta = pm.Deterministic('delta', mu_delta + sigma_delta * pm.Normal('d_off', 0, 1, shape=n_terr))
        p_hist = pm.math.invlogit(mu_hist + a)
        p_curr = pm.math.invlogit(mu_curr + a + delta)
        kappa = pm.Gamma('kappa', alpha=3, beta=0.2)
        pm.BetaBinomial('y_curr', alpha=p_curr * kappa, beta=(1 - p_curr) * kappa, n=n, observed=y)
        bb_h = pm.BetaBinomial.dist(alpha=p_hist * kappa, beta=(1 - p_hist) * kappa, n=np.maximum(n_h, 1))
        pm.Potential('y_hist', (pm.logp(bb_h, pt.as_tensor_variable(y_h))
                                * pt.as_tensor_variable((n_h > 0).astype(float))).sum())
        trace = pm.sample(draws=800, tune=600, chains=2, cores=2, target_accept=0.95,
                          random_seed=seed, progressbar=False,
                          idata_kwargs={'log_likelihood': False})

    delta_samples = trace.posterior['delta'].values.reshape(-1, n_terr)
    return (delta_samples > LOG_SIR).mean(axis=0)


def sir_exceedance(df: pd.DataFrame, national_rate: float,
                   n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """P(SIR > 1.5) per territory from the two-step SIR axis, on the same data.

    Draws current-rate posterior samples from the conjugate Beta posterior (a
    weakly-informative national-centred prior) and feeds them to the pipeline's
    ``compute_smr_sir`` so the SIR definition is identical to production.
    """
    prior = 20.0
    a0 = national_rate * prior
    b0 = (1 - national_rate) * prior
    post_a = a0 + df['recent_count_curr'].to_numpy()
    post_b = b0 + (df['all_tested_curr'].to_numpy() - df['recent_count_curr'].to_numpy())
    p_samples = rng.beta(post_a[None, :], post_b[None, :], size=(n_draws, len(df)))
    p_list = [p_samples[:, i] for i in range(len(df))]
    out = compute_smr_sir(p_list, df.reset_index(drop=True), national_rate,
                          sir_threshold=SIR_THRESHOLD)
    return np.asarray(out['exc_prob_sir'])


def _metrics(flagged: np.ndarray, truth: np.ndarray) -> Tuple[float, float, float, int]:
    tp = int(np.sum(flagged & truth)); fp = int(np.sum(flagged & ~truth))
    fn = int(np.sum(~flagged & truth)); tn = int(np.sum(~flagged & ~truth))
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    fdr = fp / (tp + fp) if (tp + fp) else 0.0
    return sens, spec, fdr, tp + fp


def run_scenario(scn: Scenario, seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    df = simulate_two_period(scn, rng)
    truth = df['is_trend'].to_numpy()
    national = float(df['recent_count_curr'].sum() / df['all_tested_curr'].sum())

    # Joint two-period model.
    exc_tp = fit_two_period_delta(df, national, seed)
    cut_tp, _ = bayesian_fdr_threshold(exc_tp)
    tp_metrics = _metrics(exc_tp > cut_tp, truth)

    # Two-step SIR baseline on the same data.
    exc_sir = sir_exceedance(df, national, n_draws=4000, rng=rng)
    cut_sir, _ = bayesian_fdr_threshold(exc_sir)
    sir_metrics = _metrics(exc_sir > cut_sir, truth)

    return {'scenario': scn.name, 'two_period': tp_metrics, 'sir': sir_metrics}


def main():
    seed = 2026
    scenarios = [
        Scenario("null (no true trend)", 120, 0.02, 40, 1.0, 0.0),
        Scenario("moderate trend, sparse", 120, 0.02, 30, 2.0, 0.15),
        Scenario("strong trend, sparse", 120, 0.02, 30, 4.0, 0.15),
        Scenario("strong trend, well sampled", 120, 0.02, 120, 4.0, 0.15),
    ]
    results = [run_scenario(s, seed + i) for i, s in enumerate(scenarios)]

    print("=" * 104)
    print("Trend detection: joint two-period model vs two-step SIR  |  trend = >=1.5x current-vs-history rise")
    print("=" * 104)
    hdr = f"{'scenario':<32}{'method':<14}{'sensitivity':>13}{'specificity':>13}{'emp.FDR':>10}{'flagged':>10}"
    print(hdr)
    print("-" * 104)
    for r in results:
        for method in ('two_period', 'sir'):
            s, sp, f, nf = r[method]
            label = 'two-period' if method == 'two_period' else 'SIR (legacy)'
            print(f"{r['scenario']:<32}{label:<14}{s:>13.3f}{sp:>13.3f}{f:>10.3f}{nf:>10d}")
        print("-" * 104)
    print("Reading the table: higher sensitivity at equal/controlled FDR is better; in the")
    print("null scenario both methods should flag almost nothing.")


if __name__ == '__main__':
    main()
