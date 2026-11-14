# -*- coding: utf-8 -*-
"""
Does the joint two-period model detect local trends at all, and when?

Simulates two-window recency data with a KNOWN truth over a grid of scenarios
and fits the production two-period model (same priors as
``BayesianAnalyzer.run_two_period_model``; the testing-intensity term is
omitted because tests per month are held constant between windows, and
production switches the term off in that case). The trend call is the
production rule: P(delta_i > 0) above the Bayesian-FDR cut-off (start and floor
0.80), with the presence gate of at least two current recent events.

To separate a data limit from a model defect, every dataset is also analysed by
a reference test with no pooling at all: a one-sided Fisher exact test of
current vs history per territory with Benjamini-Hochberg control at 5%. Where
the reference also fails, the events are too few for any method; where the
reference succeeds and the model does not, the model is losing information.

Scenarios vary
  * tests per territory in the current window (history = 4x, a 12-month
    baseline against a 3-month window at a steady testing rate);
  * the share of territories that truly rise;
  * the size of the rise (2x, 3x, 5x, mixed within each dataset);
  * a null scenario, a national-shift scenario (every territory rises 2x, so a
    local trend flag is a false positive), and overdispersed data.

Metrics per scenario: sensitivity by rise size, false-positive rate among
territories without a local rise, empirical FDR, 95% interval coverage of the
true local log change, and convergence (max R-hat).

Run:
    python validation/trend_power_simulation.py [out_dir]
Environment: TPS_REPS (replicates per scenario, default 5), TPS_K (territories,
default 60), TPS_ONLY (comma list of scenario names).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline.bootstrap  # noqa: F401 -- env/warning setup before heavy imports

import arviz as az
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from pipeline.standardization import bayesian_fdr_threshold

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
REPS = int(os.environ.get('TPS_REPS', '5'))
K = int(os.environ.get('TPS_K', '60'))
P_BASE = 0.015                 # national history recency proportion
HETERO_SD = 0.4                # between-territory lognormal spread of the base rate
EFFECTS = (2.0, 3.0, 5.0)      # local rises, assigned in equal shares to risers


@dataclass
class Scenario:
    name: str
    n_curr: int                # mean tests per territory, current window
    frac_rise: float = 0.15    # share of territories with a local rise
    national_shift: float = 1.0
    true_kappa: float | None = None   # None = Binomial data; a number = Beta-Binomial


SCENARIOS = [
    Scenario('null_n120', 120, frac_rise=0.0),
    Scenario('n30', 30),
    Scenario('n120', 120),
    Scenario('n500', 500),
    Scenario('n2000', 2000),
    Scenario('n500_many_risers', 500, frac_rise=0.40),
    Scenario('n500_national_shift', 500, frac_rise=0.15, national_shift=2.0),
    Scenario('n500_overdispersed', 500, true_kappa=50.0),
]


def simulate(scn: Scenario, rng: np.random.Generator) -> pd.DataFrame:
    n_rise = int(round(scn.frac_rise * K))
    effect = np.ones(K)
    idx = rng.permutation(K)[:n_rise]
    effect[idx] = np.resize(np.array(EFFECTS), n_rise)
    base = np.clip(P_BASE * rng.lognormal(0.0, HETERO_SD, K), 1e-4, 0.3)
    p_h = base
    p_c = np.clip(base * effect * scn.national_shift, 1e-4, 0.9)
    n_c = np.maximum(1, rng.poisson(scn.n_curr, K))
    n_h = np.maximum(1, rng.poisson(4 * scn.n_curr, K))
    if scn.true_kappa:
        kap = scn.true_kappa
        p_h = rng.beta(p_h * kap, (1 - p_h) * kap)
        p_c = rng.beta(p_c * kap, (1 - p_c) * kap)
    return pd.DataFrame({
        'y_c': rng.binomial(n_c, p_c), 'n_c': n_c,
        'y_h': rng.binomial(n_h, p_h), 'n_h': n_h,
        'effect': effect,
        # true local log change beyond the national change (logit scale ~ log at low p)
        'true_local': np.log(effect),
    })


def fit_model(df: pd.DataFrame, seed: int):
    y, n = df['y_c'].to_numpy(), df['n_c'].to_numpy()
    y_h, n_h = df['y_h'].to_numpy(), df['n_h'].to_numpy()
    nat = float(np.clip(y.sum() / n.sum(), 1e-3, 0.5))
    prior_mu = float(np.log(nat / (1 - nat)))
    with pm.Model():
        mu_hist = pm.Normal('mu_hist', mu=prior_mu, sigma=1.5)
        mu_curr = pm.Normal('mu_curr', mu=prior_mu, sigma=1.5)
        sigma_a = pm.HalfNormal('sigma_a', sigma=2)
        sigma_delta = pm.HalfNormal('sigma_delta', sigma=1.0)
        a = pm.Deterministic('a', sigma_a * pm.Normal('a_offset', 0, 1, shape=K))
        delta = pm.Deterministic('delta', sigma_delta * pm.Normal('d_offset', 0, 1, shape=K))
        p_h = pm.math.invlogit(mu_hist + a)
        p_c = pm.math.invlogit(mu_curr + a + delta)
        kappa = pm.Gamma('kappa', alpha=2, beta=0.01)
        pm.BetaBinomial('y_obs', alpha=p_c * kappa, beta=(1 - p_c) * kappa, n=n, observed=y)
        bb_h = pm.BetaBinomial.dist(alpha=p_h * kappa, beta=(1 - p_h) * kappa, n=n_h)
        pm.Potential('y_hist_ll', pm.logp(bb_h, pt.as_tensor_variable(y_h)).sum())
        idata = pm.sample(draws=1000, tune=1000, chains=2, cores=1, target_accept=0.95,
                          random_seed=seed, progressbar=False)
    dl = idata.posterior['delta'].values.reshape(-1, K)
    rhat = float(az.rhat(idata, var_names=['mu_hist', 'mu_curr', 'sigma_a', 'sigma_delta',
                                           'kappa', 'delta']).to_array().max())
    return dl, rhat, float(idata.posterior['sigma_delta'].mean()), float(idata.posterior['kappa'].mean())


def fisher_bh(df: pd.DataFrame) -> np.ndarray:
    p = np.array([fisher_exact([[yc, nc - yc], [yh, nh - yh]], alternative='greater')[1]
                  for yc, nc, yh, nh in df[['y_c', 'n_c', 'y_h', 'n_h']].to_numpy()])
    return multipletests(p, alpha=0.05, method='fdr_bh')[0]


def score(flag: np.ndarray, df: pd.DataFrame) -> dict:
    truth = df['effect'].to_numpy() > 1
    out = {'fp_rate': float(flag[~truth].mean()) if (~truth).any() else np.nan,
           'flagged': int(flag.sum()),
           'false': int((flag & ~truth).sum())}
    for e in EFFECTS:
        m = df['effect'].to_numpy() == e
        out[f'sens_{e:.0f}x'] = float(flag[m].mean()) if m.any() else np.nan
    return out


def main() -> None:
    only = set(filter(None, os.environ.get('TPS_ONLY', '').split(',')))
    rows = []
    for si, scn in enumerate(SCENARIOS):
        if only and scn.name not in only:
            continue
        for rep in range(REPS):
            seed = 7000 + 100 * si + rep
            rng = np.random.default_rng(seed)
            df = simulate(scn, rng)
            dl, rhat, sd_delta, kap = fit_model(df, seed)
            exc = (dl > 0).mean(axis=0)
            cut = bayesian_fdr_threshold(exc, start=0.80, floor=0.80)[0]
            flag_model = (exc > cut) & (df['y_c'].to_numpy() >= 2)
            flag_ref = fisher_bh(df) & (df['y_c'].to_numpy() >= 2)
            lo, hi = np.percentile(dl, [2.5, 97.5], axis=0)
            cover = float(((df['true_local'] >= lo) & (df['true_local'] <= hi)).mean())
            base = dict(scenario=scn.name, rep=rep, rhat_max=rhat, sigma_delta=sd_delta,
                        kappa=kap, coverage95=cover,
                        events_curr_median=float(np.median(df['y_c'])),
                        events_hist_median=float(np.median(df['y_h'])))
            rows.append({**base, 'method': 'two-period model', **score(flag_model, df)})
            rows.append({**base, 'method': 'Fisher + BH (no pooling)', **score(flag_ref, df)})
            pd.DataFrame(rows).to_csv(OUT / 'trend_power_raw.csv', index=False)
            print(f'[{scn.name} rep {rep + 1}/{REPS}] rhat={rhat:.3f} cut={cut:.2f} '
                  f'model flagged={int(flag_model.sum())} ref flagged={int(flag_ref.sum())}', flush=True)

    res = pd.DataFrame(rows)
    cols = ['sens_2x', 'sens_3x', 'sens_5x', 'fp_rate', 'flagged', 'false',
            'coverage95', 'rhat_max', 'sigma_delta', 'events_curr_median', 'events_hist_median']
    summ = res.groupby(['scenario', 'method'], sort=False)[cols].mean().round(3)
    summ['emp_FDR'] = (res.groupby(['scenario', 'method'], sort=False)
                       .apply(lambda g: g['false'].sum() / max(g['flagged'].sum(), 1))).round(3)
    text = summ.to_string()
    print('\n' + text)
    (OUT / 'trend_power_summary.txt').write_text(
        f'K={K} territories, {REPS} replicates per scenario, base rate {P_BASE}, '
        f'history = 4x current tests\n\n' + text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
