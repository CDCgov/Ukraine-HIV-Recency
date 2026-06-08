"""
Pure-Python helpers for :class:`BayesianCovariatesAnalyzer.run_model`.

Mirrors :mod:`pipeline.analyzers._bayesian_runtime` for the
covariates variant: input validation, covariate variance check,
adaptive ``sigma_hyperprior`` calibration (reused from the standard
helper since the rule is identical), and posterior summary
extraction.

As with the standard analyzer, the ``with pm.Model() as model:``
block stays inside the method on purpose -- registration order of
variables there feeds the seeded RNG and any reshuffle silently
shifts numerical results.

The covariates pre-model has one branch the standard analyzer does
not: when ``proportion_high_risk`` has near-zero variance the model
cannot identify ``beta_risk`` at all, and the caller is told to
fall back to the standard covariate-free fit. That decision is
reported back through the ``action`` field of the return tuple
(``'fallback'``) so the caller can route to
:class:`BayesianAnalyzer` without re-deriving the variance check.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.analyzers._bayesian_runtime import compute_sigma_hyperprior
from pipeline.validators import validate_geodataframe

logger = logging.getLogger(__name__)


def prepare_covariates_inputs(cfg: Dict[str, Any],
                              gdf_admin: Any,
                              df_stratified: Optional[pd.DataFrame],
                              level_name: str,
                              national_rate: float,
                              national_se: float,
                              parametrization: str,
                              check_multicollinearity_fn) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Validate inputs, check covariate variance, prepare arrays,
    compute ``sigma_hyperprior``.

    Returns ``(action, payload)``:

    * ``('proceed', prep_dict)`` -- normal path; ``prep_dict`` holds
      the arrays + parametrisation + ``sigma_hyperprior`` needed by
      the PyMC block, plus the labels used in the existing log line
      and the multicollinearity diagnostic for downstream storage.
    * ``('abort', None)`` -- stratified data is missing or below the
      3-territory minimum; caller returns ``(gdf_admin, None)``.
    * ``('fallback', None)`` -- ``proportion_high_risk`` has near-
      zero variance; caller should fall back to the covariate-free
      :class:`BayesianAnalyzer`.
    """
    validate_geodataframe(gdf_admin, "gdf_admin")

    if not level_name:
        raise ValueError("level_name cannot be empty")

    if parametrization not in ['centered', 'non_centered']:
        raise ValueError(f"parametrization must be 'centered' or 'non_centered', got '{parametrization}'")

    if df_stratified is None or len(df_stratified) == 0:
        logger.error(f"No stratified data available for {level_name}")
        return 'abort', None

    if len(df_stratified) < 3:
        logger.error(f"Insufficient stratified data for {level_name} (need 3+ territories)")
        return 'abort', None

    n_territories = len(df_stratified)
    avg_tests = df_stratified['all_tested_curr'].mean()

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

    logger.info(f"Running SOFT STRATIFIED Bayesian model for {level_name}")
    logger.info(f"  {len(df_stratified)} territories with proportion_high_risk covariate")
    logger.info(f"  Parametrization: {parametrization}")

    y = df_stratified['recent_count_curr'].values.astype(int)
    n = df_stratified['all_tested_curr'].values.astype(int)
    hist_prop = df_stratified['recent_proportion_hist'].values
    # NaN historical proportion -> fill with national rate so the linear
    # predictor stays defined for brand-new sites with no baseline data.
    hist_prop = np.where(np.isnan(hist_prop), national_rate, hist_prop)
    proportion_high_risk = df_stratified['proportion_high_risk'].values

    logger.info(f"  Mean proportion_high_risk: {proportion_high_risk.mean():.2f}")
    logger.info(f"  Range: [{proportion_high_risk.min():.2f}, {proportion_high_risk.max():.2f}]")

    covariate_cols = ['proportion_high_risk']
    multicollinearity_diagnostics = check_multicollinearity_fn(df_stratified, covariate_cols)

    covariate_std = proportion_high_risk.std()
    if covariate_std < 0.01:
        logger.error("="*80)
        logger.error("[WARN] CRITICAL: COVARIATE HAS ZERO VARIANCE")
        logger.error(f"[WARN] proportion_high_risk std: {covariate_std:.6f} (threshold: 0.01)")
        logger.error("[WARN] All territories have essentially the same proportion_high_risk")
        logger.error("[WARN] Covariate model cannot estimate effect - FALLING BACK to standard Bayesian model")
        logger.error("="*80)
        return 'fallback', None

    logger.info(f"[OK] Covariate variance check passed (std: {covariate_std:.3f})")

    sigma_hyperprior, prior_strength, aggregation_level = compute_sigma_hyperprior(
        cfg, y, n, n_territories, level_name, national_se, national_rate,
    )

    national_events = int(y.sum())
    logger.info(
        f"Adaptive prior: sigma={sigma_hyperprior:.2f} ({prior_strength}) based on "
        f"{national_events} national events, {n_territories} territories, {aggregation_level} level"
    )

    logger.info("Using EXCHANGEABLE model for covariates (no spatial structure)")
    logger.info("Likelihood: Beta-Binomial (Binomial recovered as kappa -> infinity)")

    return 'proceed', {
        'df_stratified': df_stratified,
        'y': y,
        'n': n,
        'hist_prop': hist_prop,
        'proportion_high_risk': proportion_high_risk,
        'parametrization': parametrization,
        'sigma_hyperprior': sigma_hyperprior,
        'prior_strength': prior_strength,
        'aggregation_level': aggregation_level,
        'multicollinearity_diagnostics': multicollinearity_diagnostics,
        'avg_tests': float(avg_tests),
        'n_territories': n_territories,
        'national_events': national_events,
    }


def extract_covariates_posterior_summaries(cfg: Dict[str, Any],
                                           trace: Any,
                                           df_stratified: pd.DataFrame,
                                           n: np.ndarray,
                                           national_rate: float,
                                           compute_smr_sir_fn) -> Tuple[pd.DataFrame, np.ndarray]:
    """Pull posterior summaries off the covariates ``trace`` and
    write them onto a fresh copy of ``df_stratified``.

    Unlike the standard analyzer, ``p`` is exposed as a Deterministic
    in the covariates model so the per-territory probabilities are
    read directly off ``trace.posterior['p']`` rather than rebuilt
    from ``alpha`` and the linear predictor.

    Returns ``(df_stratified_with_results, p_samples)`` -- ``p_samples``
    is the ``(n_samples, n_territories)`` array of posterior
    probabilities used both here for the per-territory CIs and by
    the SMR/SIR helper.
    """
    posterior_p = trace.posterior['p'].mean(dim=['chain', 'draw']).values

    df_stratified = df_stratified.copy()
    df_stratified['predicted_prob'] = posterior_p
    df_stratified['predicted'] = df_stratified['predicted_prob'] * df_stratified['all_tested_curr']

    p_samples = trace.posterior['p'].values.reshape(-1, len(df_stratified))
    df_stratified['prob_lower'] = np.percentile(p_samples, 2.5, axis=0)
    df_stratified['prob_upper'] = np.percentile(p_samples, 97.5, axis=0)

    count_samples = []
    rng = np.random.default_rng(cfg.get('random_seed', 42))
    for i in range(len(df_stratified)):
        p_i = p_samples[:, i]
        n_i = n[i]
        count_samples_i = rng.binomial(n_i, p_i)
        count_samples.append(count_samples_i)

    df_stratified['count_lower'] = [np.percentile(c, 2.5) for c in count_samples]
    df_stratified['count_upper'] = [np.percentile(c, 97.5) for c in count_samples]

    exceedance_probs = []
    for i in range(len(df_stratified)):
        p_i = p_samples[:, i]
        exceedance_prob = (p_i > national_rate).mean()
        exceedance_probs.append(exceedance_prob)

    df_stratified['exceedance_prob'] = exceedance_probs
    logger.info(f"Calculated exceedance probabilities for {len(df_stratified)} territories")

    _dt = (cfg or {}).get('detection', {})
    _smr_sir = compute_smr_sir_fn(
        p_samples, df_stratified, national_rate,
        smr_threshold=float(_dt.get('smr_threshold', 2.0)),
        sir_threshold=float(_dt.get('sir_threshold', 1.5)),
    )
    df_stratified['national_rate_curr'] = _smr_sir['national_rate_curr']
    df_stratified['baseline_rate_eb'] = _smr_sir['baseline_rate_eb']
    df_stratified['smr_mean'] = _smr_sir['smr_mean']
    df_stratified['smr_median'] = _smr_sir['smr_median']
    df_stratified['smr_lower'] = _smr_sir['smr_lower']
    df_stratified['smr_upper'] = _smr_sir['smr_upper']
    df_stratified['sir_mean'] = _smr_sir['sir_mean']
    df_stratified['sir_lower'] = _smr_sir['sir_lower']
    df_stratified['sir_upper'] = _smr_sir['sir_upper']
    df_stratified['exc_prob_smr'] = _smr_sir['exc_prob_smr']
    df_stratified['exc_prob_sir'] = _smr_sir['exc_prob_sir']
    df_stratified['exc_prob_smr_low'] = _smr_sir['exc_prob_smr_low']
    df_stratified['exc_prob_sir_low'] = _smr_sir['exc_prob_sir_low']
    logger.info(
        f"SMR/SIR computed: national_rate_curr={_smr_sir['national_rate_curr']:.4f}, "
        f"EB concentration K={_smr_sir['eb_concentration']:.1f}"
    )

    return df_stratified, p_samples


def build_detailed_analysis_text(
    level_name: str,
    kappa_mean: float, kappa_lower: float, kappa_upper: float,
    beta_risk_mean: float, beta_risk_lower: float, beta_risk_upper: float,
    OR_risk: float, OR_risk_lower: float, OR_risk_upper: float,
    beta_intensity_mean: float, beta_intensity_lower: float, beta_intensity_upper: float,
    OR_intensity: float, OR_intensity_lower: float, OR_intensity_upper: float,
    territory_analysis_sorted: List[Dict[str, Any]],
) -> str:
    """Render the human-readable per-territory analysis report for the
    HARD-stratified covariates fit.

    Builds the plain-text report stored as ``diagnostics['detailed_analysis']``:
    a header with the fitted Beta-Binomial overdispersion (``kappa``) and the
    risk-group / testing-intensity coefficients, followed by one block per
    territory describing the high- and low-risk group levels, the testing-shift
    artifact check, and the resulting outbreak classification.

    This is pure presentation -- it reads the already-computed posterior
    summaries and the per-territory ``territory_analysis_sorted`` dicts and
    returns the joined string; it performs no model fitting and touches no RNG.
    """
    lines: List[str] = []
    lines.append("="*80)
    lines.append(f"DETAILED TERRITORY ANALYSIS - BAYESIAN COVARIATES MODEL")
    lines.append(f"Level: {level_name}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("="*80)
    lines.append("")

    # Report the fitted overdispersion (kappa) of the Beta-Binomial.
    lines.append(f"LIKELIHOOD: Beta-Binomial")
    lines.append(f"  Concentration parameter (kappa): {kappa_mean:.2f} [{kappa_lower:.2f}, {kappa_upper:.2f}]")
    lines.append(f"  Variance inflation factor: {1 + 1/kappa_mean:.2f}x relative to Binomial")
    lines.append(f"  Interpretation: kappa quantifies extra-binomial variation (clustering, heterogeneity)")
    lines.append("")

    lines.append(f"RISK GROUP EFFECT:")
    lines.append(f"  Coefficient (high vs low): {beta_risk_mean:.3f} [{beta_risk_lower:.3f}, {beta_risk_upper:.3f}]")
    lines.append(f"  Odds Ratio: {OR_risk:.2f} [{OR_risk_lower:.2f}, {OR_risk_upper:.2f}]")
    lines.append("")
    lines.append(f"TESTING INTENSITY EFFECT:")
    lines.append(f"  Coefficient (log intensity): {beta_intensity_mean:.3f} [{beta_intensity_lower:.3f}, {beta_intensity_upper:.3f}]")
    lines.append(f"  Odds Ratio (1 SD increase): {OR_intensity:.2f} [{OR_intensity_lower:.2f}, {OR_intensity_upper:.2f}]")
    lines.append("")

    if beta_risk_lower > 0:
        lines.append("[OK] RISK GROUP: High-risk group has SIGNIFICANTLY higher infection rate")
        lines.append(f"   → High-risk group has {OR_risk:.1f}x higher risk than low-risk")
    elif beta_risk_upper < 0:
        lines.append("[OK] RISK GROUP: High-risk group has SIGNIFICANTLY lower infection rate")
        lines.append(f"   → Unexpected result, requires data verification")
    else:
        lines.append("[WARN] RISK GROUP: NO significant difference between high and low groups")
        lines.append(f"   → Both groups have similar infection rates")

    lines.append("")

    if beta_intensity_lower > 0:
        lines.append("[OK] TESTING INTENSITY: Higher testing intensity → SIGNIFICANTLY higher observed rate")
        lines.append(f"   → 1 SD increase in log(intensity) increases odds by {(OR_intensity - 1) * 100:.1f}%")
        lines.append(f"   → May indicate better detection OR testing artifact")
    elif beta_intensity_upper < 0:
        lines.append("[OK] TESTING INTENSITY: Higher testing intensity → SIGNIFICANTLY lower observed rate")
        lines.append(f"   → May indicate saturation effect or selection bias")
    else:
        lines.append("[WARN] TESTING INTENSITY: NO significant effect of testing intensity")
        lines.append(f"   → Observed rate does not vary with testing effort changes")

    lines.append("")
    lines.append("="*80)
    lines.append(f"DETAILED ANALYSIS OF ALL TERRITORIES ({len(territory_analysis_sorted)} territories)")
    lines.append("="*80)
    lines.append("")

    for i, terr in enumerate(territory_analysis_sorted, 1):
        lines.append("="*80)
        lines.append(f"{i}. TERRITORY: {terr['territory_name']}")
        lines.append("="*80)
        lines.append("")

        lines.append("HIGH GROUP (high-risk group):")
        lines.append(f"   • Current level: {terr['high_observed_curr']:.1%}")
        lines.append(f"   • Expected range (95% credibility): [{terr.get('high_ci_lower', 0):.1%}, {terr.get('high_ci_upper', 0):.1%}]")

        if terr['high_outbreak']:
            lines.append(f"   [OK] OUTBREAK in HIGH group (95% statistical confidence)")
            lines.append(f"   → Current level ABOVE expected range")
        else:
            lines.append(f"   ⚪ Stable level in HIGH group")
            lines.append(f"   → Current level within expected range")

        lines.append("")
        lines.append("LOW GROUP (low-risk group):")
        lines.append(f"   • Current level: {terr['low_observed_curr']:.1%}")
        lines.append(f"   • Expected range (95% credibility): [{terr.get('low_ci_lower', 0):.1%}, {terr.get('low_ci_upper', 0):.1%}]")

        if terr['low_outbreak']:
            lines.append(f"   [OK] OUTBREAK in LOW group (95% statistical confidence)")
            lines.append(f"   → Current level ABOVE expected range")
        else:
            lines.append(f"   ⚪ Stable level in LOW group")
            lines.append(f"   → Current level within expected range")

        lines.append("")
        lines.append("TESTING STRATEGY CHANGE ANALYSIS:")
        lines.append(f"   • Change in high/low proportion: {terr['testing_shift']:+.1%}")

        if terr['testing_artifact']:
            lines.append(f"   [WARN] TESTING ARTIFACT DETECTED")
            lines.append(f"   → Real levels in both groups are stable")
            lines.append(f"   → Overall increase due to testing proportion change")
            lines.append(f"   → Approximately {terr['artifact_contribution']:.0f}% of increase is artifact")
        else:
            lines.append(f"   [OK] Testing artifact NOT detected")
            lines.append(f"   → Testing proportion change is negligible or absent")

        lines.append("")
        lines.append("CONCLUSION FOR TERRITORY:")

        if terr['high_outbreak'] and not terr['low_outbreak'] and not terr['testing_artifact']:
            lines.append(f"   TYPE: REAL OUTBREAK IN HIGH-RISK GROUP")
            lines.append(f"   • Significant increase detected in high-risk")
            lines.append(f"   • Low-risk group stable")
            lines.append(f"   • Testing artifact absent")
        elif terr['low_outbreak'] and not terr['high_outbreak'] and not terr['testing_artifact']:
            lines.append(f"   TYPE: REAL OUTBREAK IN LOW-RISK GROUP")
            lines.append(f"   • Significant increase detected in low-risk")
            lines.append(f"   • High-risk group stable")
            lines.append(f"   • Testing artifact absent")
        elif terr['high_outbreak'] and terr['low_outbreak']:
            lines.append(f"   TYPE: OUTBREAK IN BOTH GROUPS")
            lines.append(f"   • Significant increase detected in both groups")
            lines.append(f"   • Critical situation")
        elif terr['testing_artifact']:
            lines.append(f"   TYPE: TESTING ARTIFACT")
            lines.append(f"   • Real infection levels are stable")
            lines.append(f"   • Increase due to testing strategy change")
            lines.append(f"   → Approximately {terr['artifact_contribution']:.0f}% of increase is artifact")
        else:
            lines.append(f"   TYPE: STABLE SITUATION")
            lines.append(f"   • Indicators within expected variation")

        lines.append("")

    return '\n'.join(lines)
