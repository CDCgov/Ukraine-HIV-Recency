"""
Pure-Python helpers for :class:`BayesianAnalyzer.run_model`.

These functions cover the work that surrounds the
``with pm.Model() as model:`` block -- input validation and adaptive
prior calibration before sampling, and posterior summary extraction
(CIs, exceedance probabilities, SMR/SIR) after sampling. The
sampling block itself is intentionally kept inline in the method:
every variable registered inside ``pm.Model()`` enters a PyTensor
symbolic graph whose registration order also feeds the seeded RNG, so
moving it would silently shift numerical results even when ``--test``
still passes.

The helpers here are deterministic numpy / pandas only, with no PyMC
context dependency.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from pipeline.diagnostics import DataQualityChecker
from pipeline.validators import validate_geodataframe

logger = logging.getLogger(__name__)


def prepare_bayesian_inputs(cfg: Dict[str, Any],
                            gdf_admin: gpd.GeoDataFrame,
                            level_name: str,
                            national_rate: float,
                            national_se: float,
                            parametrization: str) -> Optional[Dict[str, Any]]:
    """Validate inputs, run the data-quality pre-check, auto-pick
    parametrization, and compute the adaptive ``sigma_hyperprior``.

    Returns a dict with ``df``, ``y``, ``n``, ``hist_prop``,
    ``parametrization``, ``sigma_hyperprior``. Returns ``None`` when
    the data-quality check rejects the input or there are fewer than
    3 active territories -- the caller should bail out with
    ``(gdf_admin, None)`` in that case.
    """
    validate_geodataframe(gdf_admin, "gdf_admin", required_columns=['all_tested_curr', 'recent_count_curr'])

    if not level_name:
        raise ValueError("level_name cannot be empty")

    if parametrization not in ['centered', 'non_centered']:
        raise ValueError(f"parametrization must be 'centered' or 'non_centered', got '{parametrization}'")

    df = gdf_admin[gdf_admin['all_tested_curr'] > 0].copy()

    logger.info("\n--- Data Quality Pre-Check ---")
    quality_results = DataQualityChecker.check_data_quality(df, min_tests=3)
    DataQualityChecker.print_quality_report(quality_results)

    if not quality_results['proceed']:
        logger.error("[FAIL] Data quality check FAILED - skipping this level")
        return None

    if len(df) < 3:
        logger.error(f"Insufficient data for Bayesian {level_name} (need 3+ territories)")
        return None

    n_territories = len(df)
    avg_tests = df['all_tested_curr'].mean()

    auto_select = cfg.get('bayesian', {}).get('auto_select_parametrization', True)
    threshold_territories = cfg.get('bayesian', {}).get('non_centered_threshold_territories', 50)
    threshold_tests = cfg.get('bayesian', {}).get('non_centered_threshold_avg_tests', 20)

    if auto_select:
        original_param = parametrization
        if n_territories < threshold_territories or avg_tests < threshold_tests:
            parametrization = 'non_centered'
            if original_param != parametrization:
                logger.info(f"Auto-selected non_centered parametrization (n={n_territories}, avg_tests={avg_tests:.1f})")
        else:
            parametrization = 'centered'
            if original_param != parametrization:
                logger.info(f"Auto-selected centered parametrization (n={n_territories}, avg_tests={avg_tests:.1f})")

    logger.info(f"Running Bayesian hierarchical model for {level_name} ({len(df)} territories)")
    logger.info(f"Parametrization: {parametrization}")

    logger.info("Using EXCHANGEABLE model (hierarchical random effects, no spatial structure)")
    logger.info("Recommended for facility-based surveillance data")

    y = df['recent_count_curr'].values.astype(int)
    n = df['all_tested_curr'].values.astype(int)
    hist_prop = df['recent_proportion_hist'].values
    # NaN historical proportion -> fill with national rate so the linear
    # predictor stays defined for brand-new sites with no baseline data.
    hist_prop = np.where(np.isnan(hist_prop), national_rate, hist_prop)

    sigma_hyperprior, prior_strength, aggregation_level = compute_sigma_hyperprior(
        cfg, y, n, n_territories, level_name, national_se, national_rate,
    )

    national_events = int(y.sum())
    logger.info(
        f"Adaptive prior: sigma={sigma_hyperprior:.2f} ({prior_strength}) based on "
        f"{national_events} national events, {n_territories} territories, {aggregation_level} level"
    )

    logger.info("Likelihood: Beta-Binomial (Binomial recovered as kappa -> infinity)")

    return {
        'df': df,
        'y': y,
        'n': n,
        'hist_prop': hist_prop,
        'parametrization': parametrization,
        'sigma_hyperprior': sigma_hyperprior,
    }


def compute_sigma_hyperprior(cfg: Dict[str, Any],
                              y: np.ndarray,
                              n: np.ndarray,
                              n_territories: int,
                              level_name: str,
                              national_se: float,
                              national_rate: float) -> Tuple[float, str, str]:
    """Compute the adaptive ``sigma_hyperprior`` plus the labels used
    for the diagnostic log line.

    Returns ``(sigma_hyperprior, prior_strength, aggregation_level)``.
    The two label strings are kept so the existing log shape stays
    byte-identical to the inline version.
    """
    national_events = int(y.sum())

    if national_events >= 50:
        sigma_national = 1.0
        prior_strength = "weak"
    elif national_events >= 20:
        sigma_national = 0.7
        prior_strength = "moderate"
    elif national_events >= 10:
        sigma_national = 0.5
        prior_strength = "informative"
    else:
        sigma_national = 0.3
        prior_strength = "strong"

    avg_tests = float(n.mean())

    if avg_tests < 20:
        local_density_factor = 0.7
        logger.info(f"[WARN] Low average sample size ({avg_tests:.1f} tests/territory) - applying 0.7x local density factor")
    elif avg_tests < 50:
        local_density_factor = 0.85
        logger.info(f"Moderate sample size ({avg_tests:.1f} tests/territory) - applying 0.85x local density factor")
    else:
        local_density_factor = 1.0
        logger.info(f"Good sample size ({avg_tests:.1f} tests/territory) - no local density adjustment")

    sigma_hyperprior = sigma_national * local_density_factor

    se_relative = national_se / max(national_rate, 0.001)
    if se_relative > 0.5:
        sigma_hyperprior *= 1.3
        logger.info(f"High relative SE ({se_relative:.2f}) — widening sigma by 1.3x")
    elif se_relative > 0.3:
        sigma_hyperprior *= 1.15
        logger.info(f"Moderate relative SE ({se_relative:.2f}) — widening sigma by 1.15x")

    if n_territories < 10:
        sigma_hyperprior *= 0.7
        logger.info(f"[WARN] Very few territories (n={n_territories}) - applying 0.7x multiplier to sigma")

    # Optional, explicit per-resolution prior tightening from config. The
    # previous multiplier keyed off substrings of the level *name* and was
    # silently inert for the actual H3 level names ("Hex_Res4" never matched
    # the "Hex_Res_4" test), so it never fired for the hex-only pipeline.
    # Replaced by an opt-in config map (audit Mi1); absent config -> 1.0x.
    res_mult = cfg.get('bayesian', {}).get('resolution_sigma_multiplier', {})
    mult = float(res_mult.get(level_name, 1.0))
    if mult != 1.0:
        sigma_hyperprior *= mult
        logger.info(f"Applying configured sigma multiplier {mult}x for {level_name}")
    aggregation_level = level_name

    return sigma_hyperprior, prior_strength, aggregation_level


def extract_posterior_summaries(cfg: Dict[str, Any],
                                trace: Any,
                                df: pd.DataFrame,
                                hist_prop: np.ndarray,
                                n: np.ndarray,
                                y: np.ndarray,
                                national_rate: float,
                                compute_smr_sir_fn) -> Tuple[pd.DataFrame, list]:
    """Pull posterior summaries (point estimates, CIs, exceedance,
    SMR/SIR) off the sampled ``trace`` and write them onto ``df``.

    The seeded RNG used to draw count posterior predictives is
    instantiated here so the same seed in ``cfg['random_seed']``
    yields reproducible ``count_lower``/``count_upper`` columns across
    runs.

    Returns ``(df, p_samples)`` -- ``p_samples`` is the per-territory
    list of posterior-probability arrays, needed by the SMR/SIR call.
    """
    sigma_alpha_post = float(trace.posterior['sigma_alpha'].mean().values)
    mu_alpha_post = float(trace.posterior['mu_alpha'].mean().values)
    beta_post = float(trace.posterior['beta'].mean().values)
    logger.info(f"  Posterior sigma_alpha={sigma_alpha_post:.3f}, "
                f"mu_alpha={mu_alpha_post:.3f} (logit scale), beta={beta_post:.3f} (shared slope)")
    if 'kappa' in trace.posterior:
        kappa_post = float(trace.posterior['kappa'].mean().values)
        logger.info(f"  Posterior kappa={kappa_post:.1f} (Beta-Binomial concentration)")

    # Per-territory recency probability read directly from the model's
    # Deterministic ``p``. The point estimate is the posterior mean of p
    # (E[p]), not invlogit(E[alpha]+E[beta]*x): pushing posterior means
    # through the logit link is biased by Jensen's inequality and was
    # inconsistent with the exceedance / SMR numbers (audit M3). Reading
    # ``p`` directly also sidesteps reconstructing it from alpha/beta, which
    # changed shape once the per-territory random slope was removed (C1).
    p_da = trace.posterior['p']
    posterior_p = p_da.mean(dim=['chain', 'draw']).values
    p_samples = p_da.values.reshape(-1, len(df))   # (n_samples, n_territories)

    df['predicted_prob'] = posterior_p
    df['predicted'] = posterior_p * n
    df['residual'] = y - df['predicted']

    df['prob_lower'] = np.percentile(p_samples, 2.5, axis=0)
    df['prob_upper'] = np.percentile(p_samples, 97.5, axis=0)

    count_samples = []
    rng = np.random.default_rng(cfg.get('random_seed', 42))
    for i in range(len(df)):
        count_samples.append(rng.binomial(n[i], p_samples[:, i]))

    df['count_lower'] = [np.percentile(c, 2.5) for c in count_samples]
    df['count_upper'] = [np.percentile(c, 97.5) for c in count_samples]

    df['exceedance_prob'] = (p_samples > national_rate).mean(axis=0)
    logger.info(f"Calculated exceedance probabilities (P(theta > {national_rate:.4f}))")

    _dt = (cfg or {}).get('detection', {})
    _smr_sir = compute_smr_sir_fn(
        p_samples, df, national_rate,
        smr_threshold=float(_dt.get('smr_threshold', 2.0)),
        sir_threshold=float(_dt.get('sir_threshold', 1.5)),
    )
    df['national_rate_curr'] = _smr_sir['national_rate_curr']
    df['baseline_rate_eb'] = _smr_sir['baseline_rate_eb']
    df['smr_mean'] = _smr_sir['smr_mean']
    df['smr_median'] = _smr_sir['smr_median']
    df['smr_lower'] = _smr_sir['smr_lower']
    df['smr_upper'] = _smr_sir['smr_upper']
    df['sir_mean'] = _smr_sir['sir_mean']
    df['sir_lower'] = _smr_sir['sir_lower']
    df['sir_upper'] = _smr_sir['sir_upper']
    df['exc_prob_smr'] = _smr_sir['exc_prob_smr']
    df['exc_prob_sir'] = _smr_sir['exc_prob_sir']
    df['exc_prob_smr_low'] = _smr_sir['exc_prob_smr_low']
    df['exc_prob_sir_low'] = _smr_sir['exc_prob_sir_low']
    logger.info(
        f"SMR/SIR computed: national_rate_curr={_smr_sir['national_rate_curr']:.4f}, "
        f"EB concentration K={_smr_sir['eb_concentration']:.1f}"
    )

    # SIR informativeness flag (audit D2). The SIR axis compares a territory to
    # its OWN history, but on a thin baseline the Empirical-Bayes prior (the
    # national rate) dominates and SIR merely echoes the national comparison.
    # A territory's own history carries >=50% of the weight only when its
    # historical tests exceed the fitted EB concentration K
    # (weight = tested_hist / (tested_hist + K)). Below that the SIR call is
    # national-dominated, not own-history, and is flagged accordingly.
    eb_K = float(_smr_sir.get('eb_concentration', 0.0) or 0.0)
    df['sir_informative'] = df['all_tested_hist'].fillna(0).astype(float) > eb_K
    logger.info(
        f"SIR informativeness: {int(df['sir_informative'].sum())}/{len(df)} territories "
        f"own-history-driven (tested_hist > K={eb_K:.1f}); the rest are national-dominated"
    )

    exc_sorted = df.nlargest(10, 'exceedance_prob')
    logger.info(f"  Top 10 exceedance_prob values:")
    for _, r in exc_sorted.iterrows():
        rate = r.get('recent_proportion_curr', 0)
        tests = int(r.get('all_tested_curr', 0))
        recent = int(r.get('recent_count_curr', 0))
        logger.info(f"    exc_prob={r['exceedance_prob']:.3f}  rate={rate:.4f}  "
                    f"({recent}/{tests} recent/tests)")

    return df, p_samples
