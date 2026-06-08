"""
Formal overdispersion test: Binomial vs Beta-Binomial via LOO-IC.

Audit C3-B / Mo2. The production detector always uses a Beta-Binomial
likelihood, which nests the Binomial as the concentration ``kappa -> infinity``.
This diagnostic answers the separate question *"is the extra-binomial
variation actually supported by the data, or is kappa essentially
unidentified on these small counts?"* by fitting both a Binomial and a
Beta-Binomial version of the **same** exchangeable mean structure on the
active territories and comparing their expected log pointwise predictive
density (ELPD) by leave-one-out cross-validation (Vehtari, Gelman & Gabry,
2017).

The two fits here are small, dedicated and **separate** from the production
fit: they exist only to produce the comparison and never feed the
classification, reliability or maps, so they do not perturb the main
numerical results. Run only when the per-level ``use_loo_ic`` flag is set,
because each call is two extra MCMC fits.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

logger = logging.getLogger(__name__)


def _fit_mean_structure(y: np.ndarray, n: np.ndarray, hist: np.ndarray,
                        prior_mu: float, seed: int, overdispersed: bool):
    """Fit the exchangeable mean structure with a Binomial or Beta-Binomial
    likelihood and return the trace (with pointwise log-likelihood for LOO).

    The mean structure mirrors the production model after audit C1: a
    partially-pooled random intercept plus a single shared slope on the
    historical proportion. The only difference between the two calls is the
    likelihood, so the LOO contrast isolates the overdispersion question.
    """
    n_t = len(y)
    with pm.Model():
        mu_alpha = pm.Normal('mu_alpha', mu=prior_mu, sigma=1.0)
        sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=2.0)
        beta = pm.Normal('beta', mu=0.0, sigma=1.0)
        alpha_offset = pm.Normal('alpha_offset', mu=0.0, sigma=1.0, shape=n_t)
        alpha = mu_alpha + sigma_alpha * alpha_offset
        p = pm.Deterministic('p', pm.math.invlogit(alpha + beta * hist))
        if overdispersed:
            kappa = pm.Gamma('kappa', alpha=3, beta=0.2)
            pm.BetaBinomial('y_obs', alpha=p * kappa, beta=(1 - p) * kappa,
                            n=n, observed=y)
        else:
            pm.Binomial('y_obs', n=n, p=p, observed=y)
        trace = pm.sample(draws=800, tune=800, chains=2, cores=2,
                          target_accept=0.95, return_inferencedata=True,
                          idata_kwargs={"log_likelihood": True},
                          random_seed=seed, progressbar=False)
    return trace


def compare_binomial_betabinomial(df: pd.DataFrame, national_rate: float,
                                  cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fit Binomial and Beta-Binomial on the active territories and compare
    them by LOO-IC.

    Returns a dict with the best model by ELPD, the ELPD difference and its
    standard error, a boolean ``overdispersion_supported`` (Beta-Binomial
    wins by more than ~2 SE of the difference), the posterior mean of kappa,
    and a Pareto-k reliability flag. On failure returns the keys with ``None``
    plus an ``error`` string -- the caller treats this as "comparison
    unavailable", never as a hard error.
    """
    y = df['recent_count_curr'].values.astype(int)
    n = df['all_tested_curr'].values.astype(int)
    hist = df['recent_proportion_hist'].values.astype(float)
    hist = np.where(np.isnan(hist), national_rate, hist)
    seed = int(cfg.get('random_seed', 42))
    nr = float(np.clip(national_rate, 1e-4, 1.0 - 1e-4))
    prior_mu = float(np.log(nr / (1.0 - nr)))

    logger.info("LOO model comparison: fitting Binomial and Beta-Binomial on "
                f"{len(df)} active territories (two dedicated diagnostic fits)...")
    try:
        tr_binom = _fit_mean_structure(y, n, hist, prior_mu, seed, overdispersed=False)
        tr_bb = _fit_mean_structure(y, n, hist, prior_mu, seed, overdispersed=True)

        comparison = az.compare({'binomial': tr_binom, 'beta_binomial': tr_bb}, ic='loo')
        best = str(comparison.index[0])
        # Second row holds the difference of the runner-up from the best model.
        elpd_diff = float(comparison.iloc[1]['elpd_diff'])
        dse = float(comparison.iloc[1].get('dse', float('nan')))
        kappa_mean = float(tr_bb.posterior['kappa'].mean().values)

        overdispersion_supported = bool(
            best == 'beta_binomial' and np.isfinite(dse) and dse > 0
            and elpd_diff > 2.0 * dse
        )

        result: Dict[str, Any] = {
            'loo_best_model': best,
            'loo_elpd_diff': elpd_diff,
            'loo_dse': dse,
            'overdispersion_supported': overdispersion_supported,
            'kappa_posterior_mean': kappa_mean,
        }

        logger.info(
            f"LOO comparison: best={best}, elpd_diff={elpd_diff:.2f} (dse={dse:.2f}), "
            f"kappa~{kappa_mean:.1f} -> overdispersion "
            f"{'SUPPORTED' if overdispersion_supported else 'not clearly supported'}"
        )

        # Pareto-k reliability of the LOO approximation (Vehtari et al. 2017).
        try:
            khat = az.loo(tr_bb, pointwise=True).pareto_k.values
            n_bad = int((khat > 0.7).sum())
            if n_bad:
                logger.warning(f"LOO: {n_bad}/{len(khat)} Pareto k > 0.7 for Beta-Binomial "
                               "-- LOO approximation may be unreliable for those points")
                result['loo_pareto_k_bad'] = n_bad
        except (ValueError, KeyError, AttributeError):
            pass

        return result

    except (ValueError, RuntimeError, KeyError, FloatingPointError) as e:
        logger.error(f"LOO model comparison failed: {e}")
        return {
            'loo_best_model': None,
            'overdispersion_supported': None,
            'error': str(e),
        }
