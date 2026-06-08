"""
:class:`BayesianCovariatesAnalyzer` -- the explanatory model that
augments the standard hierarchical fit with the ``proportion_high_risk``
covariate and, in the HARD variant, a per-territory pair of fits for
the high- and low-risk subpopulations.

This is *not* a replacement for :class:`BayesianAnalyzer`; the crude
detector remains the primary classifier and the covariate model sits
beside it. Adjusting for risk-group composition here is descriptive,
not causal -- treating the risk mix as a confounder would mask the
very signal surveillance is meant to catch (see Hernan & Robins 2020
on over-adjustment for mediators).

When the SOFT stratified fit cannot be built (no stratification rows,
multicollinearity, or PyMC fit failure), the analyzer falls back to a
standard :class:`BayesianAnalyzer` run so the caller still gets a fit
on the same data.

``run_hurdle_model`` here delegates to ``BayesianAnalyzer.run_hurdle_model``
on a temporary instance: the truncated-Binomial path on active sites
does not change with covariates, so reusing the standard implementation
keeps the two branches in sync.
"""

from __future__ import annotations

import gc
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import arviz as az
import geopandas as gpd
import numpy as np
import pandas as pd
import pymc as pm

from pipeline.analyzers.base import BaseHotspotAnalyzer
from pipeline.analyzers.bayesian import BayesianAnalyzer
from pipeline.analyzers._covariates_runtime import (
    build_detailed_analysis_text,
    extract_covariates_posterior_summaries,
    prepare_covariates_inputs,
)
from pipeline.classification import (
    HOTSPOT_LABELS,
    add_smr_sir_counts as _add_smr_sir_counts,
    classify_with_exceedance as _classify_with_exceedance,
    classify_with_smr_sir as _classify_with_smr_sir,
)
from pipeline.constants import ANALYSIS_CONSTANTS
from pipeline.diagnostics import (
    BayesianDiagnosticsFixed,
    DataQualityChecker,
    DiagnosticInterpreter,
    ReliabilityScoreCalculator,
    calculate_covariates_diagnostics as _calculate_covariates_diagnostics,
    calculate_covariates_diagnostics_stratified as _calculate_covariates_diagnostics_stratified,
    check_multicollinearity as _check_multicollinearity_fn,
)
from pipeline.exceptions import (
    DataValidationError,
    InsufficientDataError,
    ModelConvergenceError,
)
from pipeline.history import HistoricalComparison
from pipeline.models import ParallelSamplingConfig, SamplingProgressBar
from pipeline.standardization import bayesian_fdr_threshold
from pipeline.standardization.smr_sir import compute_smr_sir, eb_baseline_rate
from pipeline.standardization.z_scores import calculate_z_scores as _calculate_z_scores
from pipeline.validators import validate_geodataframe

logger = logging.getLogger(__name__)


class BayesianCovariatesAnalyzer(BaseHotspotAnalyzer):
    """Bayesian Hierarchical Model with covariates -- the **explanatory layer**.

    This analyzer answers a different question than the crude detector:
    "where is the recent-infection proportion higher than the risk-group
    composition predicts?". It runs in parallel and its output sits next to
    the crude classification rather than overriding it.

    Adjusting for ``proportion_high_risk`` here is descriptive, not causal.
    The risk-group composition lies on the causal pathway from local
    environment to recent infection, so treating it as a confounder would
    mask the very signal the surveillance system is meant to catch
    (Hernán & Robins, *Causal Inference: What If*, 2020, Chs 7--9 on
    over-adjustment for mediators). ``testing_intensity``, when added,
    behaves differently -- it shifts only how many people are tested, not
    the underlying recency proportion -- and remains a legitimate
    adjustment.
    """

    MODEL_TYPE = "bayesian_covariates"

    def __init__(self, config: Dict[str, Any], mode_suffix: str = 'admin', orchestrator=None):
        """Initialise like the base analyzer plus the stratified-data slots.

        ``df_stratified`` holds the per-territory covariate frame (one row
        per territory under SOFT stratification) and ``df_hard_stratified``
        the HARD-stratified frame (two rows per territory: high- and
        low-risk groups). Both are populated lazily during ``run_model``.
        """
        super().__init__(config, mode_suffix, orchestrator)
        self.df_stratified = None
        self.df_hard_stratified = None

    def _check_multicollinearity(self, df: pd.DataFrame, covariate_cols: List[str]) -> dict:
        """Thin wrapper around :func:`pipeline.diagnostics.check_multicollinearity`."""
        return _check_multicollinearity_fn(df, covariate_cols)

    def run_model(self, gdf_admin: gpd.GeoDataFrame, level_name: str,
                  national_rate: float, national_se: float,
                  parametrization: str = 'non_centered') -> Tuple[gpd.GeoDataFrame, dict]:
        """Run STRATIFIED Bayesian hierarchical model with risk_group covariate.

        This model analyzes high and low risk groups SEPARATELY (2 observations per territory)
        to directly answer: "Is there a difference in recent infection rate between high and low risk groups?"

        Args:
            parametrization: 'centered' or 'non_centered' (default: 'non_centered')
                - non_centered: Reduces divergences for small samples (RECOMMENDED)
                - centered: Standard parametrization (only for large samples)
        """
        action, prep = prepare_covariates_inputs(
            self.cfg, gdf_admin, self.df_stratified, level_name,
            national_rate, national_se, parametrization,
            self._check_multicollinearity,
        )
        if action == 'abort':
            return gdf_admin, None
        if action == 'fallback':
            logger.info("Falling back to BayesianAnalyzer instead of BayesianCovariatesAnalyzer")
            standard_analyzer = BayesianAnalyzer(self.cfg, self.gdf_cases, self.analysis_start, self.analysis_end)
            return standard_analyzer.run_model(gdf_admin, level_name, national_rate, national_se, parametrization)

        df_stratified = prep['df_stratified']
        y = prep['y']
        n = prep['n']
        hist_prop = prep['hist_prop']
        proportion_high_risk = prep['proportion_high_risk']
        parametrization = prep['parametrization']
        sigma_hyperprior = prep['sigma_hyperprior']
        prior_strength = prep['prior_strength']
        aggregation_level = prep['aggregation_level']
        avg_tests = prep['avg_tests']
        n_territories = prep['n_territories']
        national_events = prep['national_events']
        self._multicollinearity_diagnostics = prep['multicollinearity_diagnostics']

        # Audit trail: prior strength + likelihood family. Kept here in the
        # method (not in the prep helper) so the orchestrator-coupled audit
        # state stays out of the pure-Python prep path.
        if hasattr(self, 'orchestrator') and self.orchestrator and hasattr(self.orchestrator, 'current_audit_trail'):
            audit_trail = self.orchestrator.current_audit_trail
            if audit_trail:
                audit_trail.add_decision(
                    test_name="Prior Strength Selection - Covariates Model",
                    test_type="threshold",
                    result=f"National events={national_events}, Avg tests={avg_tests:.1f}, Territories={n_territories}",
                    decision=f"Use {prior_strength} priors (sigma={sigma_hyperprior:.2f})",
                    reason=f"Adaptive prior based on sample size. Aggregation level: {aggregation_level}.",
                    impact="Stronger priors provide regularization for small samples, preventing overfitting. "
                           "Weaker priors allow data to dominate when sample is large.",
                    details={
                        'national_events': national_events,
                        'avg_tests': avg_tests,
                        'n_territories': n_territories,
                        'sigma_hyperprior': float(sigma_hyperprior),
                        'prior_strength': prior_strength,
                        'aggregation_level': aggregation_level,
                    },
                )
                audit_trail.add_decision(
                    test_name="Likelihood family - Covariates Model",
                    test_type="modeling",
                    result="Beta-Binomial selected unconditionally",
                    decision="Use Beta-Binomial likelihood",
                    reason="Beta-Binomial nests the Binomial (kappa -> infinity) and "
                           "absorbs any overdispersion without a separate, error-prone test.",
                    impact="Credible intervals widen automatically when extra-binomial "
                           "variation is present and are unaffected when it is absent.",
                    details={'use_beta_binomial': True},
                )

        try:
            with pm.Model() as model:
                # Hyperpriors for territory-specific intercepts
                # center mu_alpha on national baseline rate (logit scale)
                prior_mu = pm.math.logit(np.clip(
                    national_rate,
                    ANALYSIS_CONSTANTS['prior_mu_logit_clip_min']['value'],
                    ANALYSIS_CONSTANTS['prior_mu_logit_clip_max']['value']))
                mu_alpha = pm.Normal('mu_alpha', mu=prior_mu, sigma=sigma_hyperprior)
                sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=2)

                # Exchangeable model: hierarchical random effects without spatial structure
                if parametrization == 'non_centered':
                    alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=n_territories)
                    alpha = pm.Deterministic('alpha', mu_alpha + sigma_alpha * alpha_offset)
                else:
                    alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, shape=n_territories)

                # Effect of historical proportion (shared across territories)
                beta_hist = pm.Normal('beta_hist', mu=0, sigma=1)

                # Effect of proportion_high_risk (KEY: continuous covariate 0-1)
                # Positive beta_risk means higher proportion of high-risk → higher infection rate
                beta_risk = pm.Normal('beta_risk', mu=0, sigma=1)

                # Effect of testing intensity (log-transformed)
                # Positive beta_intensity means higher testing intensity → higher observed rate
                # (could indicate better detection OR testing artifact)
                beta_intensity = pm.Normal('beta_intensity', mu=0, sigma=1)

                # Prepare log(testing_intensity) covariate
                # Add small constant to avoid log(0)
                testing_intensity_curr = df_stratified['testing_intensity_curr'].values
                log_intensity = np.log(testing_intensity_curr + 0.1)

                # Standardize log_intensity for better sampling
                log_intensity_mean = log_intensity.mean()
                log_intensity_std = log_intensity.std()
                if log_intensity_std > 0:
                    log_intensity_standardized = (log_intensity - log_intensity_mean) / log_intensity_std
                else:
                    log_intensity_standardized = log_intensity - log_intensity_mean

                # Linear predictor with proportion_high_risk and log(testing_intensity) as covariates
                # No territory indexing needed - one row per territory
                logit_p = (alpha +
                          beta_hist * hist_prop +
                          beta_risk * proportion_high_risk +
                          beta_intensity * log_intensity_standardized)

                p = pm.Deterministic('p', pm.math.invlogit(logit_p))

                # Beta-Binomial likelihood. kappa is the Beta concentration:
                # large kappa -> near-Binomial, small kappa -> strong
                # overdispersion. Gamma(3, 0.2) is a weakly informative prior.
                kappa = pm.Gamma('kappa', alpha=3, beta=0.2)
                obs = pm.BetaBinomial('obs', alpha=p * kappa, beta=(1 - p) * kappa,
                                      n=n, observed=y)

                # Sample with adaptive target_accept
                # ETA estimation for large grids (H3 res5)
                n_territories = len(df_stratified)
                draws = 1000
                tune = 2000
                if n_territories > 1000:
                    # Rough estimate: ~0.5-1.5 sec per territory for tune+draw
                    estimated_minutes = (n_territories * 1.0 * (tune + draws) / 1000) / 60
                    logger.info(f"Sampling from posterior (large grid: {n_territories} territories, ETA: {estimated_minutes:.0f}-{estimated_minutes*2:.0f} min)...")
                else:
                    logger.info("Sampling from posterior (this may take a few minutes)...")

                trace, sampling_info = ParallelSamplingConfig.adaptive_sample(
                    model=model,
                    initial_target_accept=0.95,
                    draws=draws,
                    tune=tune,
                    chains=4,
                    cores=4,
                    random_seed=self.cfg.get('random_seed', 42),
                    progressbar=False
                )

                # Log sampling adaptation info
                if sampling_info['adapted']:
                    logger.info(f"[OK] Adaptive sampling: {sampling_info['n_attempts']} attempts, "
                              f"final target_accept={sampling_info['final_target_accept']:.2f}, "
                              f"divergences={sampling_info['divergence_pct']:.1f}%")
                else:
                    logger.info(f"[OK] Sampling completed without adaptation (divergences={sampling_info['divergence_pct']:.1f}%)")

                # CRITICAL CHECK: Convergence failure detection
                # If divergences > 5% after all adaptive attempts, posterior geometry is unhealthy
                # Results are unreliable and should be flagged as FATAL
                convergence_fatal = False
                if sampling_info['divergence_pct'] > 5.0:
                    logger.error("="*80)
                    logger.error("[WARN] CRITICAL: MODEL CONVERGENCE FAILED")
                    logger.error(f"[WARN] Divergences: {sampling_info['divergence_pct']:.1f}% (threshold: 5%)")
                    logger.error(f"[WARN] Attempts: {sampling_info.get('n_attempts', 1)}")
                    logger.error(f"[WARN] Final target_accept: {sampling_info.get('final_target_accept', 'N/A')}")
                    logger.error("[WARN] POSTERIOR GEOMETRY IS UNHEALTHY - RESULTS UNRELIABLE")
                    logger.error("[WARN] Reliability Score will be set to FATAL")
                    logger.error("="*80)
                    convergence_fatal = True

                # Record convergence decision in audit trail (Covariates Model)
                if hasattr(self, 'orchestrator') and self.orchestrator and hasattr(self.orchestrator, 'current_audit_trail'):
                    audit_trail = self.orchestrator.current_audit_trail
                    if audit_trail:
                        audit_trail.add_decision(
                            test_name="Convergence Diagnostics - Covariates Model",
                            test_type="diagnostic",
                            result=f"Divergences: {sampling_info['divergence_pct']:.1f}%, " +
                                   f"Attempts: {sampling_info.get('n_attempts', 1)}, " +
                                   f"Target accept: {sampling_info.get('final_target_accept', 'N/A')}",
                            decision="Accept results" if not convergence_fatal else "Flag as UNRELIABLE",
                            reason=f"Threshold: 5% divergences. Current: {sampling_info['divergence_pct']:.1f}%. " +
                                   ("Healthy posterior geometry - MCMC converged successfully" if not convergence_fatal
                                    else "Unhealthy posterior geometry - MCMC failed to converge"),
                            impact="Results are reliable and can be used for inference" if not convergence_fatal
                                   else "Results are unreliable - consider non-centered parametrization or simpler model",
                            details={
                                'divergence_pct': sampling_info['divergence_pct'],
                                'n_attempts': sampling_info.get('n_attempts', 1),
                                'final_target_accept': sampling_info.get('final_target_accept'),
                                'convergence_fatal': convergence_fatal,
                                'threshold': 5.0
                            }
                        )

            # IMPORTANT: Save model separately for diagnostics
            saved_model = model

            # Extract beta_risk (the key parameter!)
            beta_risk_mean = float(trace.posterior['beta_risk'].mean().values)
            beta_risk_hdi = az.hdi(trace, var_names=['beta_risk'])['beta_risk'].values
            beta_risk_lower = float(beta_risk_hdi[0])
            beta_risk_upper = float(beta_risk_hdi[1])

            # Extract beta_intensity (testing effort effect!)
            beta_intensity_mean = float(trace.posterior['beta_intensity'].mean().values)
            beta_intensity_hdi = az.hdi(trace, var_names=['beta_intensity'])['beta_intensity'].values
            beta_intensity_lower = float(beta_intensity_hdi[0])
            beta_intensity_upper = float(beta_intensity_hdi[1])

            # Extract the fitted Beta-Binomial concentration (kappa).
            kappa_mean = float(trace.posterior['kappa'].mean().values)
            kappa_hdi = az.hdi(trace, var_names=['kappa'])['kappa'].values
            kappa_lower = float(kappa_hdi[0])
            kappa_upper = float(kappa_hdi[1])

            # Convert to odds ratio
            # Interpretation: OR for 100% high-risk vs 0% high-risk
            OR_risk = np.exp(beta_risk_mean)
            OR_risk_lower = np.exp(beta_risk_lower)
            OR_risk_upper = np.exp(beta_risk_upper)

            # Interpretation: OR for 1 SD increase in log(testing_intensity)
            OR_intensity = np.exp(beta_intensity_mean)
            OR_intensity_lower = np.exp(beta_intensity_lower)
            OR_intensity_upper = np.exp(beta_intensity_upper)

            logger.info(f"\n" + "="*80)
            logger.info(f"COVARIATE EFFECTS (Soft Stratification)")
            logger.info(f"="*80)

            # Report the fitted overdispersion (kappa) of the Beta-Binomial.
            logger.info(f"Likelihood: Beta-Binomial")
            logger.info(f"  Concentration parameter (kappa): {kappa_mean:.2f} [{kappa_lower:.2f}, {kappa_upper:.2f}]")
            logger.info(f"  Lower kappa = more overdispersion")
            logger.info(f"  Variance inflation factor: {1 + 1/kappa_mean:.2f}x relative to Binomial")
            logger.info(f"")
            logger.info(f"Risk Group Effect:")
            logger.info(f"  Coefficient for proportion_high_risk: {beta_risk_mean:.3f} [{beta_risk_lower:.3f}, {beta_risk_upper:.3f}]")
            logger.info(f"  Odds Ratio (100% high-risk vs 0%): {OR_risk:.2f} [{OR_risk_lower:.2f}, {OR_risk_upper:.2f}]")
            logger.info(f"")
            logger.info(f"Testing Intensity Effect:")
            logger.info(f"  Coefficient for log(testing_intensity): {beta_intensity_mean:.3f} [{beta_intensity_lower:.3f}, {beta_intensity_upper:.3f}]")
            logger.info(f"  Odds Ratio (1 SD increase): {OR_intensity:.2f} [{OR_intensity_lower:.2f}, {OR_intensity_upper:.2f}]")
            logger.info(f"  Statistical confidence: 95%")
            logger.info(f"")

            if beta_risk_lower > 0:
                logger.info("[OK] RISK GROUP: Higher proportion of high-risk testing → SIGNIFICANTLY higher infection rate")
                logger.info(f"   → Territory with 100% high-risk has {OR_risk:.1f}x higher risk than 0% high-risk")
                logger.info(f"   → Each 10% increase in high-risk proportion increases odds by {(np.exp(beta_risk_mean * 0.1) - 1) * 100:.1f}%")
            elif beta_risk_upper < 0:
                logger.info("[OK] RISK GROUP: Higher proportion of high-risk testing → SIGNIFICANTLY lower infection rate")
                logger.info(f"   → Unexpected result, may indicate testing artifact or data quality issue")
            else:
                logger.info("[WARN] RISK GROUP: NO significant effect of risk group composition")
                logger.info(f"   → Infection rate does not vary with proportion of high-risk testing")

            if beta_intensity_lower > 0:
                logger.info("[OK] TESTING INTENSITY: Higher testing intensity → SIGNIFICANTLY higher observed rate")
                logger.info(f"   → 1 SD increase in log(intensity) increases odds by {(OR_intensity - 1) * 100:.1f}%")
                logger.info(f"   → May indicate better detection OR testing artifact (network expansion)")
            elif beta_intensity_upper < 0:
                logger.info("[OK] TESTING INTENSITY: Higher testing intensity → SIGNIFICANTLY lower observed rate")
                logger.info(f"   → May indicate saturation effect or selection bias in expanded networks")
            else:
                logger.info("[WARN] TESTING INTENSITY: NO significant effect of testing intensity")
                logger.info(f"   → Observed rate does not vary with testing effort changes")

            logger.info(f"="*80)

            df_stratified, p_samples = extract_covariates_posterior_summaries(
                self.cfg, trace, df_stratified, n, national_rate,
                BaseHotspotAnalyzer._compute_smr_sir,
            )

            # HYBRID APPROACH: Use HARD stratification where possible, SOFT as fallback
            # Check if HARD stratification data is available
            df_hard = self.df_hard_stratified
            use_hybrid = (df_hard is not None and len(df_hard) > 0)

            if use_hybrid:
                logger.info(f"\nHYBRID STRATIFICATION: Using HARD where possible, SOFT as fallback")
                n_can_use_hard = df_hard[df_hard['risk_group'] == 'high']['can_use_hard'].sum()
                n_must_use_soft = len(df_stratified) - n_can_use_hard
                logger.info(f"   HARD stratification: {n_can_use_hard} territories (testing artifact detection enabled)")
                logger.info(f"   SOFT stratification: {n_must_use_soft} territories (insufficient data for HARD)")
            else:
                logger.info(f"\n[WARN] NOTE: HARD stratification not available - using SOFT for all territories")
                logger.info(f"   Outbreak detection flags will be disabled")
                logger.info(f"   high_outbreak, low_outbreak, testing_artifact will be False for all territories")

            # Create territory_analysis list
            territory_analysis = []
            for idx, row in df_stratified.iterrows():
                territory_idx = row['territory_idx']

                # Try HARD stratification first
                if use_hybrid and df_hard is not None:
                    # Check if this territory can use HARD
                    terr_hard = df_hard[df_hard['territory_idx'] == territory_idx]
                    if len(terr_hard) == 2:
                        can_use_hard = terr_hard.iloc[0]['can_use_hard']
                        if can_use_hard:
                            # Use HARD stratification - detect outbreak and artifact
                            hard_result = self.detect_outbreak_and_artifact(territory_idx, df_hard, national_rate)

                            territory_analysis.append({
                                'territory_idx': territory_idx,
                                'territory_name': row['territory_name'],
                                'high_observed_curr': hard_result['high_observed_curr'],
                                'high_ci_lower': hard_result['high_ci_lower'],
                                'high_ci_upper': hard_result['high_ci_upper'],
                                'high_outbreak': hard_result['high_outbreak'],
                                'low_observed_curr': hard_result['low_observed_curr'],
                                'low_ci_lower': hard_result['low_ci_lower'],
                                'low_ci_upper': hard_result['low_ci_upper'],
                                'low_outbreak': hard_result['low_outbreak'],
                                'testing_shift': hard_result['testing_shift'],
                                'testing_artifact': hard_result['testing_artifact'],
                                'artifact_contribution': hard_result['artifact_contribution'],
                                'outbreak_type': hard_result['outbreak_type'],
                                'explanation': hard_result['explanation'],
                                'stratification_method': 'HARD',
                                'overall_change': row.get('predicted_prob', 0.0) - row.get('recent_proportion_hist', 0.0)
                            })
                            continue

                # Fallback to SOFT stratification
                territory_analysis.append({
                    'territory_idx': territory_idx,
                    'territory_name': row['territory_name'],
                    'high_observed_curr': 0.0,
                    'high_ci_lower': 0.0,
                    'high_ci_upper': 0.0,
                    'high_outbreak': False,
                    'low_observed_curr': 0.0,
                    'low_ci_lower': 0.0,
                    'low_ci_upper': 0.0,
                    'low_outbreak': False,
                    'testing_shift': 0.0,
                    'testing_artifact': False,
                    'artifact_contribution': 0.0,
                    'outbreak_type': 'INSUFFICIENT DATA',
                    'explanation': 'Insufficient data for HARD stratification (need ≥3 tests in both groups)',
                    'stratification_method': 'SOFT',
                    'overall_change': row.get('predicted_prob', 0.0) - row.get('recent_proportion_hist', 0.0)
                })

            # Simplified analysis for soft stratification
            # No separate high/low outbreak detection - just overall territory analysis
            logger.info(f"\n" + "="*80)
            logger.info(f"TERRITORY ANALYSIS WITH RISK GROUP ADJUSTMENT")
            logger.info(f"="*80)

            # Add proportion_high_risk to output for transparency
            df_stratified['risk_group_composition'] = df_stratified['proportion_high_risk'].apply(
                lambda x: f"{x*100:.0f}% high-risk"
            )

            # Sort by overall_change (descending) and analyze ALL territories
            territory_analysis_sorted = sorted(territory_analysis, key=lambda x: abs(x['overall_change']), reverse=True)

            logger.info(f"\nDETAILED ANALYSIS OF ALL TERRITORIES ({len(territory_analysis_sorted)} territories):")
            logger.info(f"")

            for i, terr in enumerate(territory_analysis_sorted, 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"{i}. TERRITORY: {terr['territory_name']}")
                logger.info(f"   Stratification method: {terr.get('stratification_method', 'SOFT')}")
                logger.info(f"{'='*80}")

                # Show detailed analysis only for HARD stratification
                if terr.get('stratification_method') == 'HARD':
                    logger.info(f"\nHIGH GROUP (high-risk group):")
                    logger.info(f"   • Current level: {terr['high_observed_curr']:.1%}")
                    logger.info(f"   • Expected range (95% credibility): [{terr.get('high_ci_lower', 0):.1%}, {terr.get('high_ci_upper', 0):.1%}]")

                    if terr['high_outbreak']:
                        logger.info(f"   [OK] OUTBREAK in HIGH group (95% statistical confidence)")
                        logger.info(f"   → Current level ABOVE expected range")
                    else:
                        logger.info(f"   ⚪ Stable level in HIGH group")
                        logger.info(f"   → Current level within expected range")

                    logger.info(f"\nLOW GROUP (low-risk group):")
                    logger.info(f"   • Current level: {terr['low_observed_curr']:.1%}")
                    logger.info(f"   • Expected range (95% credibility): [{terr.get('low_ci_lower', 0):.1%}, {terr.get('low_ci_upper', 0):.1%}]")

                    if terr['low_outbreak']:
                        logger.info(f"   [OK] OUTBREAK in LOW group (95% statistical confidence)")
                        logger.info(f"   → Current level ABOVE expected range")
                    else:
                        logger.info(f"   ⚪ Stable level in LOW group")
                        logger.info(f"   → Current level within expected range")

                    logger.info(f"\nTESTING STRATEGY CHANGE ANALYSIS:")
                    logger.info(f"   • Change in high/low proportion: {terr['testing_shift']:+.1%}")

                    if terr['testing_artifact']:
                        logger.info(f"   [WARN] TESTING ARTIFACT DETECTED")
                        logger.info(f"   → Approximately {terr['artifact_contribution']:.0f}% of increase is artifact")
                    else:
                        logger.info(f"   [OK] Testing artifact NOT detected")

                    # Final conclusion for this territory
                    logger.info(f"\nCONCLUSION:")
                    logger.info(f"   TYPE: {terr.get('outbreak_type', 'UNKNOWN')}")
                    logger.info(f"\n   EXPLANATION:")
                    # Split explanation into lines for better readability
                    explanation_lines = terr.get('explanation', '').split('. ')
                    for line in explanation_lines:
                        if line.strip():
                            logger.info(f"   {line.strip()}.")
                else:
                    # SOFT stratification - show simplified info
                    logger.info(f"\n[WARN] SOFT STRATIFICATION (insufficient data for detailed analysis)")
                    logger.info(f"   {terr.get('explanation', 'Insufficient data for separate risk group analysis')}")

            # Build the human-readable per-territory analysis report.
            detailed_analysis = build_detailed_analysis_text(
                level_name,
                kappa_mean, kappa_lower, kappa_upper,
                beta_risk_mean, beta_risk_lower, beta_risk_upper,
                OR_risk, OR_risk_lower, OR_risk_upper,
                beta_intensity_mean, beta_intensity_lower, beta_intensity_upper,
                OR_intensity, OR_intensity_lower, OR_intensity_upper,
                territory_analysis_sorted,
            )

            # Calculate residuals (already have predicted from lines 5645-5677)
            df_stratified['residual'] = df_stratified['recent_count_curr'] - df_stratified['predicted']

            # Aggregate back to territory level (combine high and low)
            # Ensure all required columns exist before aggregation
            for _col, _default in [
                ('recent_count_hist', 0), ('all_tested_hist', 0),
                ('count_lower', 0.0), ('count_upper', 0.0),
            ]:
                if _col not in df_stratified.columns:
                    df_stratified[_col] = _default

            # Build the territory-level frame from the stratified results.
            # Counts sum; proportions, posterior summaries and exceedance
            # probabilities are averaged across the strata that make up the
            # territory (a territory is typically split into two risk-group
            # strata, so the average is across those).
            df_territory = df_stratified.groupby('territory_idx').agg({
                'territory_name': 'first',
                'all_tested_curr': 'sum',
                'recent_count_curr': 'sum',
                'recent_proportion_curr': 'mean',
                'recent_proportion_hist': 'mean',
                'all_tested_hist': 'sum',
                'recent_count_hist': 'sum',
                'predicted': 'sum',
                'residual': 'sum',
                'exceedance_prob': 'mean',
                'count_lower': 'sum',
                'count_upper': 'sum',
                # SIR/SMR taxonomy inputs, aggregated to the territory level.
                'smr_mean': 'mean', 'smr_median': 'mean', 'smr_lower': 'mean', 'smr_upper': 'mean',
                'sir_mean': 'mean', 'sir_lower': 'mean', 'sir_upper': 'mean',
                'exc_prob_smr': 'mean', 'exc_prob_sir': 'mean',
                'exc_prob_smr_low': 'mean', 'exc_prob_sir_low': 'mean',
                'national_rate_curr': 'first',
                'baseline_rate_eb': 'mean',
            }).reset_index()

            # Calculate z-scores at territory level using unified method
            df_territory['national_baseline'] = national_rate
            # Percent deviation from the current national rate, via SMR.
            df_territory['deviation_pct'] = (df_territory['smr_mean'] - 1.0) * 100

            # Use unified calculate_z_scores method
            df_territory = self.calculate_z_scores(df_territory, national_rate)

            # Shared FDR-controlled SMR/SIR classification (audit M2 — same
            # post-fit step as the crude and hurdle fits).
            df_territory = self._finalize_classification(df_territory, national_rate)

            # IMPORTANT: Classification based on Z-score, NOT on component interpretation
            # If Z-score is high → territory marked as hotspot
            # Regardless of whether it's a real outbreak or testing artifact
            # Detailed component analysis helps USER make the decision

            # Merge back to gdf_admin (including outbreak flags)
            for idx, row in df_territory.iterrows():
                territory_idx = row['territory_idx']
                for col in ['predicted', 'residual', 'national_baseline', 'deviation_pct',
                           'z_national', 'z_residual', 'combined_z', 'exceedance_prob', 'classification',
                           # Combined burden + rate watch-list (add_watchlist).
                           'on_watchlist', 'watch_reason', 'watch_rank',
                           'burden_rank', 'rate_rank', 'burden_share_pct',
                           'burden_high', 'rate_high']:
                    if col in row.index:
                        gdf_admin.loc[territory_idx, col] = row[col]

            # Add outbreak flags from territory_analysis
            for terr in territory_analysis:
                territory_idx = terr['territory_idx']
                gdf_admin.loc[territory_idx, 'high_outbreak'] = terr['high_outbreak']
                gdf_admin.loc[territory_idx, 'low_outbreak'] = terr['low_outbreak']
                gdf_admin.loc[territory_idx, 'testing_artifact'] = terr['testing_artifact']
                gdf_admin.loc[territory_idx, 'high_observed_curr'] = terr['high_observed_curr']
                gdf_admin.loc[territory_idx, 'high_ci_upper'] = terr['high_ci_upper']
                gdf_admin.loc[territory_idx, 'low_observed_curr'] = terr['low_observed_curr']
                gdf_admin.loc[territory_idx, 'low_ci_upper'] = terr['low_ci_upper']

            # DEBUG: Verify columns were added
            logger.debug(f"After adding outbreak flags, gdf_admin columns: {list(gdf_admin.columns)}")
            logger.debug(f"high_outbreak in gdf_admin: {'high_outbreak' in gdf_admin.columns}")
            if 'high_outbreak' in gdf_admin.columns:
                logger.debug(f"high_outbreak values: {gdf_admin['high_outbreak'].tolist()}")

            # Generate PPC once for reuse in diagnostics and plotting
            logger.info("Generating posterior predictive samples...")
            with saved_model:
                ppc = pm.sample_posterior_predictive(trace, progressbar=False, random_seed=self.cfg.get('random_seed', 42))

            # Calculate diagnostics (using territory-level aggregated data)
            diagnostics = self._calculate_diagnostics_stratified(
                trace, df_territory, df_stratified,
                level_name, national_rate,
                beta_risk_mean, beta_risk_lower, beta_risk_upper,
                OR_risk, OR_risk_lower, OR_risk_upper,
                territory_analysis,
                ppc,
                convergence_fatal
            )

            # Store model, trace, and PPC for visualization
            diagnostics['model'] = saved_model  # [OK] Save PyMC Model
            diagnostics['trace'] = trace
            diagnostics['y_obs'] = y
            diagnostics['ppc'] = ppc  # [OK] Save PPC for reuse
            diagnostics['detailed_analysis'] = detailed_analysis  # Save detailed territory analysis

            return gdf_admin, diagnostics

        except (ValueError, RuntimeError, KeyError) as e:
            logger.error(f"Stratified Bayesian model failed: {e}")
            logger.error(traceback.format_exc())
            return gdf_admin, None

    def _calculate_diagnostics_stratified(self, trace, df_territory, df_stratified, level_name, national_rate,
                                          beta_risk_mean, beta_risk_lower, beta_risk_upper,
                                          OR_risk, OR_risk_lower, OR_risk_upper,
                                          territory_analysis, ppc=None, convergence_fatal=False) -> dict:
        """Thin wrapper around :func:`pipeline.diagnostics.calculate_covariates_diagnostics_stratified`."""
        return _calculate_covariates_diagnostics_stratified(
            trace, df_territory, df_stratified, level_name, national_rate,
            beta_risk_mean, beta_risk_lower, beta_risk_upper,
            OR_risk, OR_risk_lower, OR_risk_upper,
            territory_analysis, ppc=ppc, convergence_fatal=convergence_fatal,
        )

    def _calculate_diagnostics(self, trace, df, level_name, national_rate) -> dict:
        """Thin wrapper around :func:`pipeline.diagnostics.calculate_covariates_diagnostics`."""
        return _calculate_covariates_diagnostics(trace, df, level_name, national_rate)

    def run_hurdle_model(self, gdf_admin: gpd.GeoDataFrame, level_name: str,
                         national_rate: float) -> Tuple[gpd.GeoDataFrame, dict]:
        """
        Run Hurdle Binomial model with covariates for sparse data.

        Note: This is a simplified version that doesn't include risk_group covariate.
        For full covariate support, use the standard run_model() method.

        Args:
            gdf_admin: GeoDataFrame with all territories
            level_name: Administrative level name
            national_rate: National recency rate

        Returns:
            Tuple of (updated GeoDataFrame, diagnostics dict)
        """
        logger.info(f"\n--- Hurdle Binomial Model (Covariates) for {level_name} ---")
        logger.warning("[WARN] Hurdle model for BayesianCovariatesAnalyzer uses simplified version without risk_group")
        logger.warning("[WARN] For full covariate support, use standard model with lower structural zero threshold")

        # Delegate to BayesianAnalyzer's hurdle model
        # Create temporary BayesianAnalyzer instance
        temp_analyzer = BayesianAnalyzer(self.cfg, self.mode_suffix, self.orchestrator)

        # Copy necessary attributes
        temp_analyzer.gdf_cases = self.gdf_cases if hasattr(self, 'gdf_cases') else None

        # Run hurdle model
        gdf_result, diagnostics = temp_analyzer.run_hurdle_model(gdf_admin, level_name, national_rate)

        if diagnostics:
            diagnostics['model_name'] = 'Hurdle Binomial (Covariates - Simplified)'

        return gdf_result, diagnostics
