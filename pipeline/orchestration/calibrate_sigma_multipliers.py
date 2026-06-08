"""
Grid search for the optimal ``sigma_hyperprior`` multipliers.

The Bayesian hierarchical model uses one global ``sigma`` for the
prior on territory-level log-odds, scaled by two multipliers --
``small_sample`` (for thin test counts) and ``local_density`` (for
medium counts). This routine fits the Beta-Binomial model on a 6x6
grid of multiplier values, ranks them by LOO-IC's ELPD, and returns
the best combination. The ``se_*`` multipliers are left at their
defaults because they rarely trigger in production data.

Falls back to the literature defaults (0.7/0.85/1.3/1.15) if every
grid cell raises during sampling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from pipeline.constants import ANALYSIS_CONSTANTS

logger = logging.getLogger(__name__)


def calibrate_sigma_multipliers(gdf: pd.DataFrame, level_name: str,
                                national_rate: float, national_se: float,
                                start, end, b_start, b_end,
                                output_dir: Path,
                                random_seed: int = 42) -> Dict[str, float]:
    """Grid-search the sigma multipliers; return the LOO-best combination."""
    logger.info("\n" + "=" * 60)
    logger.info("SIGMA MULTIPLIER CALIBRATION")
    logger.info("=" * 60)


    # Only calibrate small_sample and local_density (most impactful)
    small_sample_mults = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    local_density_mults = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0]

    results = []

    for ss_mult in small_sample_mults:
        for ld_mult in local_density_mults:
            try:
                # Temporarily override config
                orig_ss = ANALYSIS_CONSTANTS.get('sigma_hyperprior_small_sample_mult', {}).get('value', 0.7)
                orig_ld = ANALYSIS_CONSTANTS.get('sigma_hyperprior_local_density_mult', {}).get('value', 0.85)

                # Compute sigma with these multipliers
                n_events = gdf['recent_count_curr'].sum()
                n_territories = len(gdf[gdf['all_tested_curr'] > 0])
                avg_tests = gdf.loc[gdf['all_tested_curr'] > 0, 'all_tested_curr'].mean()

                sigma_national = 0.3 if n_events < 10 else 0.5 if n_events < 20 else 0.7
                local_density_factor = ss_mult if avg_tests < 15 else ld_mult if avg_tests < 30 else 1.0
                sigma_hyperprior = sigma_national * local_density_factor

                # Fit model with this sigma and compute LOO
                df_active = gdf[gdf['all_tested_curr'] > 0].copy()
                y = df_active['recent_count_curr'].values.astype(float)
                n = df_active['all_tested_curr'].values.astype(float)

                with pm.Model() as model:
                    prior_mu = pm.math.logit(np.clip(
                national_rate,
                ANALYSIS_CONSTANTS['prior_mu_logit_clip_min']['value'],
                ANALYSIS_CONSTANTS['prior_mu_logit_clip_max']['value']))
                    mu_alpha = pm.Normal('mu_alpha', mu=prior_mu, sigma=sigma_hyperprior)
                    sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=0.5)
                    # Non-centered parametrization (matches run_model — reduces divergences)
                    alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=len(y))
                    alpha = pm.Deterministic('alpha', mu_alpha + sigma_alpha * alpha_offset)
                    p = pm.math.sigmoid(alpha)
                    # Match the analysis likelihood (Beta-Binomial) so the
                    # sigma chosen here is calibrated under the same model.
                    kappa = pm.Gamma('kappa', alpha=3, beta=0.2)
                    y_obs = pm.BetaBinomial('y_obs', alpha=p * kappa, beta=(1 - p) * kappa,
                                            n=n, observed=y)
                    trace = pm.sample(500, tune=250, chains=4, cores=1,
                                     return_inferencedata=True,
                                     idata_kwargs={"log_likelihood": True},
                                     random_seed=random_seed, progressbar=False)

                loo = az.loo(trace)
                results.append({
                    'ss_mult': ss_mult,
                    'ld_mult': ld_mult,
                    'sigma': sigma_hyperprior,
                    'elpd': loo.elpd_loo,
                    'se': loo.se
                })

            except Exception as e:
                logger.debug(f"  Calibration failed for ss={ss_mult}, ld={ld_mult}: {e}")
                continue

    if not results:
        logger.warning("Sigma calibration failed — using defaults")
        return {'small_sample_mult': 0.7, 'local_density_mult': 0.85,
                'se_high_mult': 1.3, 'se_moderate_mult': 1.15}

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('elpd', ascending=False)

    best = df_results.iloc[0]
    default = df_results[(df_results['ss_mult'] == 0.7) & (df_results['ld_mult'] == 0.85)]

    logger.info(f"\n  OPTIMAL SIGMA MULTIPLIERS:")
    logger.info(f"    small_sample_mult: {best['ss_mult']}")
    logger.info(f"    local_density_mult: {best['ld_mult']}")
    logger.info(f"    sigma = {best['sigma']:.3f}")
    logger.info(f"    ELPD = {best['elpd']:.2f} ± {best['se']:.2f}")
    if len(default) > 0:
        logger.info(f"    Default (0.7/0.85) ELPD = {default.iloc[0]['elpd']:.2f}")
        logger.info(f"    Improvement: {best['elpd'] - default.iloc[0]['elpd']:.2f}")

    logger.info(f"\n  TOP 5 COMBINATIONS:")
    logger.info(f"  {'ss_mult':<10} {'ld_mult':<10} {'sigma':<10} {'ELPD':<12}")
    for _, row in df_results.head(5).iterrows():
        logger.info(f"  {row['ss_mult']:<10} {row['ld_mult']:<10} {row['sigma']:<10.3f} {row['elpd']:<12.2f}")

    # Save calibration report
    cal_file = output_dir / f'Sigma_Calibration_{level_name}.xlsx'
    df_results.to_excel(cal_file, index=False)
    logger.info(f"\n  Calibration report saved: {cal_file}")

    optimal = {
        'small_sample_mult': float(best['ss_mult']),
        'local_density_mult': float(best['ld_mult']),
        'se_high_mult': 1.3,  # Keep default — rarely triggered
        'se_moderate_mult': 1.15  # Keep default
    }

    return optimal
