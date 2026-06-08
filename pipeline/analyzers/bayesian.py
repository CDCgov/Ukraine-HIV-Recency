"""
:class:`BayesianAnalyzer` -- the standard Bayesian hotspot detector.

Hierarchical Beta-Binomial with exchangeable random effects on the
logit scale. Hyperprior strength adapts to the national event count
and local sample size; aggregation level (ADM3/ADM2/ADM1, fine/medium/
coarse hex) tightens the prior further to control variance inflation
at finer resolutions. Sampling uses :class:`ParallelSamplingConfig`'s
adaptive ``target_accept`` retry loop, and a convergence-fatal gate
flips on when divergences exceed 5% after every adaptive attempt so
unhealthy posteriors do not silently propagate to downstream maps and
reports.

``run_hurdle_model`` is the Truncated Binomial branch for sparse data
with structural zeros: when at least ``hurdle_threshold`` percent of
sites are structurally inactive, the fit is restricted to the active
sites only and the inactive ones are reported as structural zeros
without being passed through the latent recency model.
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
from pipeline.analyzers._bayesian_runtime import (
    extract_posterior_summaries,
    prepare_bayesian_inputs,
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
    calculate_bayesian_diagnostics as _calculate_bayesian_diagnostics,
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


class BayesianAnalyzer(BaseHotspotAnalyzer):
    """Bayesian Hierarchical Model -- the **primary crude detector**.

    This analyzer answers "where is the recent-infection proportion
    higher than the national current rate?" without adjusting for the
    composition of who walks in the door. That is intentional: programme
    targeting cares about absolute burden, not about residual burden after
    risk-mix adjustment.

    Outputs from this model drive the hotspot list, the maps and the
    recommendations. The covariate model (:class:`BayesianCovariatesAnalyzer`)
    is a parallel explanatory layer -- it asks the different question
    "where is the burden higher than risk composition predicts?" and is
    reported alongside, not used to override the crude classification.
    """

    MODEL_TYPE = "bayesian"

    def run_model(self, gdf_admin: gpd.GeoDataFrame, level_name: str,
                  national_rate: float, national_se: float,
                  parametrization: str = 'non_centered') -> Tuple[gpd.GeoDataFrame, dict]:
        """Run Bayesian hierarchical model with exchangeable random effects.

        Args:
            parametrization: 'centered' or 'non_centered' (default: 'non_centered')
                - non_centered: Reduces divergences for small samples (RECOMMENDED)
                - centered: Standard parametrization (only for large samples >50 territories)
        """
        prep = prepare_bayesian_inputs(
            self.cfg, gdf_admin, level_name, national_rate, national_se, parametrization,
        )
        if prep is None:
            return gdf_admin, None
        df = prep['df']
        y = prep['y']
        n = prep['n']
        hist_prop = prep['hist_prop']
        parametrization = prep['parametrization']
        sigma_hyperprior = prep['sigma_hyperprior']

        try:
            with pm.Model() as model:
                # Hyperpriors for hierarchical structure (partial pooling)
                # center mu_alpha on national baseline rate (logit scale)
                # instead of mu=0 (which = 50% probability — unrealistic for HIV)
                prior_mu = pm.math.logit(np.clip(
                    national_rate,
                    ANALYSIS_CONSTANTS['prior_mu_logit_clip_min']['value'],
                    ANALYSIS_CONSTANTS['prior_mu_logit_clip_max']['value']))
                mu_alpha = pm.Normal('mu_alpha', mu=prior_mu, sigma=sigma_hyperprior)
                sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=2)

                # Shared population-level slope on the historical proportion.
                # A single coefficient is identifiable from all territories;
                # a per-territory random slope is NOT (one Binomial observation
                # per territory), so it only added a posterior ridge and extra
                # divergences without buying information. Removed per audit C1:
                # alpha stays a partially-pooled random intercept, beta is one
                # fixed population-level coefficient.
                beta = pm.Normal('beta', mu=0, sigma=sigma_hyperprior)

                # Exchangeable model: hierarchical random intercept, no spatial structure
                if parametrization == 'non_centered':
                    alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=len(df))
                    alpha = pm.Deterministic('alpha', mu_alpha + sigma_alpha * alpha_offset)
                else:
                    alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, shape=len(df))

                # Linear predictor (logit link)
                logit_p = alpha + beta * hist_prop

                # Per-territory recency probability on the natural scale.
                # Exposed as a Deterministic so it is stored in the trace and
                # available to posterior predictive checks and to the
                # standardized-ratio computations downstream.
                p = pm.Deterministic('p', pm.math.invlogit(logit_p))

                # Beta-Binomial likelihood. kappa is the concentration of the
                # Beta mixing distribution: large kappa -> near-Binomial,
                # small kappa -> strong overdispersion. Gamma(3, 0.2) is a
                # weakly informative prior that keeps kappa positive.
                kappa = pm.Gamma('kappa', alpha=3, beta=0.2)
                y_obs = pm.BetaBinomial('y_obs', alpha=p * kappa, beta=(1 - p) * kappa,
                                        n=n, observed=y)

                # [WARN] IMPROVEMENT 4: Parallel sampling configuration
                sampling_config = ParallelSamplingConfig.get_sampling_config(
                    n_territories=len(df),
                    fast_mode=False,
                    cores_override=self.cfg.get('sampling', {}).get('cores'),
                )
                draws = sampling_config['draws']
                tune = sampling_config['tune']
                chains = sampling_config['chains']
                cores = sampling_config['cores']
                target_accept = sampling_config['target_accept']
                logger.info(f"Using optimized sampling: {chains} chains, {draws} draws, {cores} cores")

                # [OK] IMPROVEMENT 6: Progress bar
                progress_callback = SamplingProgressBar.create_progress_callback()

                # Sample from posterior with adaptive target_accept
                # ETA estimation for large grids (H3 res5)
                n_territories = len(df)
                if n_territories > 1000:
                    # Rough estimate: ~0.5-1.5 sec per territory for tune+draw
                    estimated_minutes = (n_territories * 1.0 * (tune + draws) / 1000) / 60
                    logger.info(f"Sampling from posterior (large grid: {n_territories} territories, ETA: {estimated_minutes:.0f}-{estimated_minutes*2:.0f} min)...")
                else:
                    logger.info("Sampling from posterior (this may take a few minutes)...")

                trace, sampling_info = ParallelSamplingConfig.adaptive_sample(
                    model=model,
                    initial_target_accept=target_accept,
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    cores=cores,
                    random_seed=self.cfg.get('random_seed', 42),
                    progressbar=False,
                    callback=progress_callback if progress_callback else None
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

            # IMPORTANT: Save model separately for diagnostics (trace doesn't have .model attribute)
            saved_model = model

            logger.info("[OK] Bayesian model converged")

            df, p_samples = extract_posterior_summaries(
                self.cfg, trace, df, hist_prop, n, y, national_rate,
                BaseHotspotAnalyzer._compute_smr_sir,
            )

            # Z-scores, then the shared FDR-controlled SMR/SIR classification
            # (audit M2 — identical post-fit step in the hurdle and covariates
            # fits, centralised in BaseHotspotAnalyzer._finalize_classification).
            df = self.calculate_z_scores(df, national_rate)
            df = self._finalize_classification(df, national_rate)

            # Join results back
            result_cols = ['predicted', 'predicted_prob', 'prob_lower', 'prob_upper', 'residual',
                          'exceedance_prob', 'z_national', 'z_residual', 'combined_z', 'classification',
                          'national_baseline', 'deviation_pct',
                          # SIR/SMR taxonomy outputs:
                          # point summaries and 95% credible intervals for each
                          # ratio, the four exceedance probabilities used by the
                          # taxonomy, the taxonomy label itself, the
                          # new-site flag, and the in-window national rate /
                          # EB-shrunken historical rate kept for traceability.
                          'smr_mean', 'smr_median', 'smr_lower', 'smr_upper',
                          'sir_mean', 'sir_lower', 'sir_upper',
                          'exc_prob_smr', 'exc_prob_sir',
                          'exc_prob_smr_low', 'exc_prob_sir_low',
                          'classification_smr_sir', 'is_new_site',
                          'national_rate_curr', 'baseline_rate_eb',
                          # Combined burden + rate watch-list (add_watchlist).
                          'on_watchlist', 'watch_reason', 'watch_rank',
                          'burden_rank', 'rate_rank', 'burden_share_pct',
                          'burden_high', 'rate_high']

            for col in result_cols:
                if col in df.columns:
                    # Use index-based merge instead of .values to avoid misalignment
                    gdf_admin.loc[gdf_admin['all_tested_curr'] > 0, col] = gdf_admin.loc[gdf_admin['all_tested_curr'] > 0].index.map(df[col])

            # D2: SIR informativeness flag (own-history-driven vs national-dominated)
            if 'sir_informative' in df.columns:
                gdf_admin.loc[gdf_admin['all_tested_curr'] > 0, 'sir_informative'] = \
                    gdf_admin.loc[gdf_admin['all_tested_curr'] > 0].index.map(df['sir_informative'])

            # Generate PPC once for reuse in diagnostics and plotting
            logger.info("Generating posterior predictive samples...")
            with saved_model:
                ppc = pm.sample_posterior_predictive(trace, progressbar=False, random_seed=self.cfg.get('random_seed', 42))

            # Diagnostics
            diagnostics = self._calculate_diagnostics(trace, df, level_name, national_rate, saved_model, ppc, convergence_fatal)

            # Store model, trace, and PPC for visualization
            diagnostics['model'] = saved_model  # [OK] Save PyMC Model
            diagnostics['trace'] = trace
            diagnostics['y_obs'] = y
            diagnostics['ppc'] = ppc  # [OK] Save PPC for reuse

            return gdf_admin, diagnostics

        except (ValueError, RuntimeError, KeyError) as e:
            logger.error(f"Bayesian model failed: {e}")
            logger.error(traceback.format_exc())
            return gdf_admin, None

    def _calculate_diagnostics(self, trace, df, level_name, national_rate, model=None, ppc=None, convergence_fatal=False) -> dict:
        """Thin wrapper around :func:`pipeline.diagnostics.calculate_bayesian_diagnostics`."""
        return _calculate_bayesian_diagnostics(trace, df, level_name, national_rate, model=model, ppc=ppc, convergence_fatal=convergence_fatal)

    def run_hurdle_model(self, gdf_admin: gpd.GeoDataFrame, level_name: str,
                         national_rate: float) -> Tuple[gpd.GeoDataFrame, dict]:
        """
        Run Hurdle Binomial model for sparse data with many structural zeros.

        Two-stage model:
        Stage 1: Logistic regression for P(site_present=1)
        Stage 2: Binomial model for active sites only

        This model is appropriate when:
        - High proportion of structural zeros (>70%)
        - Clear distinction between sites with/without testing
        - Facility-based surveillance data

        Args:
            gdf_admin: GeoDataFrame with all territories
            level_name: Administrative level name
            national_rate: National recency rate

        Returns:
            Tuple of (updated GeoDataFrame, diagnostics dict)
        """
        logger.info(f"\n--- Hurdle Binomial Model for {level_name} ---")

        # Check if site_present flag exists
        if 'site_present' not in gdf_admin.columns:
            logger.error("site_present column not found - cannot run Hurdle model")
            return gdf_admin, None

        # Count structural zeros
        n_total = len(gdf_admin)
        n_active = gdf_admin['site_present'].sum()
        n_structural_zeros = n_total - n_active
        pct_structural = (n_structural_zeros / n_total) * 100

        logger.info(f"Total territories: {n_total}")
        logger.info(f"Active sites: {n_active} ({(n_active/n_total)*100:.1f}%)")
        logger.info(f"Structural zeros: {n_structural_zeros} ({pct_structural:.1f}%)")

        if n_active < 3:
            logger.error(f"Need at least 3 active sites for the truncated-Binomial model, got {n_active}")
            return gdf_admin, None

        # Despite the historical function name ``run_hurdle_model`` (kept for
        # backwards compatibility with the CLI flag and config key), this is
        # a truncated Binomial analysis on the active sites only, not a true
        # two-stage Hurdle. ``site_present`` is determined deterministically
        # from the testing-sites registry (a site counts as "present" in a
        # window iff the registry shows it was operating then), so adding
        # a separate logistic stage for site presence would contribute no
        # information -- it would just be observing the registry twice.
        df_active = gdf_admin[gdf_admin['site_present'] == True].copy()

        logger.info(f"Truncated Binomial (active sites): {len(df_active)} active sites")

        # Prepare data
        y = df_active['recent_count_curr'].values.astype(int)
        n = df_active['all_tested_curr'].values.astype(int)
        hist_prop = df_active['recent_proportion_hist'].values
        # Fill missing historical proportions (new sites) with the national
        # recency proportion to avoid NaN propagating into the model.
        hist_prop = np.where(np.isnan(hist_prop), national_rate, hist_prop)

        # Check for valid data
        if len(y) == 0 or n.sum() == 0:
            logger.error("No valid data for active sites")
            return gdf_admin, None

        # Adaptive prior based on sample size
        national_events = y.sum()
        avg_tests = n.mean()

        if national_events >= 50:
            sigma_hyperprior = 1.0
            prior_strength = "weak"
        elif national_events >= 20:
            sigma_hyperprior = 0.7
            prior_strength = "moderate"
        elif national_events >= 10:
            sigma_hyperprior = 0.5
            prior_strength = "informative"
        else:
            sigma_hyperprior = 0.3
            prior_strength = "strong"

        if avg_tests < 20:
            sigma_hyperprior *= 0.7
            logger.info(f"[WARN] Low average sample size ({avg_tests:.1f}) - tightening priors")

        logger.info(f"Prior strength: {prior_strength} (sigma={sigma_hyperprior:.2f})")
        logger.info(f"National events: {national_events}, Avg tests: {avg_tests:.1f}")

        # Likelihood choice: always Beta-Binomial (Binomial recovered as
        # kappa -> infinity). See BayesianAnalyzer.run_model for the rationale.
        logger.info("Likelihood: Beta-Binomial (Binomial recovered as kappa -> infinity)")

        try:
            with pm.Model() as model:
                # Hyperpriors
                # center mu_alpha on national baseline rate (logit scale)
                prior_mu = pm.math.logit(np.clip(
                    national_rate,
                    ANALYSIS_CONSTANTS['prior_mu_logit_clip_min']['value'],
                    ANALYSIS_CONSTANTS['prior_mu_logit_clip_max']['value']))
                mu_alpha = pm.Normal('mu_alpha', mu=prior_mu, sigma=2)
                mu_beta = pm.Normal('mu_beta', mu=0, sigma=2)
                sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=sigma_hyperprior)
                sigma_beta = pm.HalfNormal('sigma_beta', sigma=sigma_hyperprior)

                # Non-centered parametrization for better sampling
                alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=len(df_active))
                beta_offset = pm.Normal('beta_offset', mu=0, sigma=1, shape=len(df_active))

                alpha = pm.Deterministic('alpha', mu_alpha + sigma_alpha * alpha_offset)
                beta = pm.Deterministic('beta', mu_beta + sigma_beta * beta_offset)

                # Logit model
                logit_p = alpha + beta * hist_prop

                # Per-territory recency probability (Deterministic so it is in
                # the trace for posterior predictive checks). Beta-Binomial
                # likelihood; kappa is the Beta concentration (large -> near
                # Binomial). Gamma(3, 0.2) is a weakly informative prior.
                p = pm.Deterministic('p', pm.math.invlogit(logit_p))
                kappa = pm.Gamma('kappa', alpha=3, beta=0.2)
                y_obs = pm.BetaBinomial('y_obs', alpha=p * kappa, beta=(1 - p) * kappa,
                                        n=n, observed=y)

                # Sample
                logger.info("Sampling from posterior...")
                trace = pm.sample(
                    draws=1000,
                    tune=500,
                    chains=2,
                    cores=1,
                    target_accept=0.95,
                    return_inferencedata=True,
                    idata_kwargs={"log_likelihood": True},
                    random_seed=self.cfg.get('random_seed', 42),
                    progressbar=False
                )

            logger.info("[OK] Sampling completed")

            # Extract posterior samples
            alpha_samples = trace.posterior['alpha'].values.reshape(-1, len(df_active))
            beta_samples = trace.posterior['beta'].values.reshape(-1, len(df_active))

            # Calculate predictions for active sites
            p_samples = []
            for i in range(len(df_active)):
                logit_p_samples = alpha_samples[:, i] + beta_samples[:, i] * hist_prop[i]
                p_samples.append(1 / (1 + np.exp(-logit_p_samples)))

            df_active['predicted_prob'] = [np.mean(p) for p in p_samples]
            df_active['predicted'] = df_active['predicted_prob'] * df_active['all_tested_curr']
            df_active['residual'] = df_active['recent_count_curr'] - df_active['predicted']
            df_active['prob_lower'] = [np.percentile(p, 2.5) for p in p_samples]
            df_active['prob_upper'] = [np.percentile(p, 97.5) for p in p_samples]

            # Calculate exceedance probability
            exceedance_probs = []
            for i in range(len(df_active)):
                p_i = p_samples[i]
                exceedance_prob = (p_i > national_rate).mean()
                exceedance_probs.append(exceedance_prob)

            df_active['exceedance_prob'] = exceedance_probs

            # Standardized comparison ratios for the active-sites subset
            # (additive; the new taxonomy consumes these columns later).
            _dt = (self.cfg or {}).get('detection', {}) if isinstance(self.cfg, dict) else {}
            _smr_sir = BaseHotspotAnalyzer._compute_smr_sir(
                p_samples, df_active, national_rate,
                smr_threshold=float(_dt.get('smr_threshold', 2.0)),
                sir_threshold=float(_dt.get('sir_threshold', 1.5)),
            )
            df_active['national_rate_curr'] = _smr_sir['national_rate_curr']
            df_active['baseline_rate_eb'] = _smr_sir['baseline_rate_eb']
            df_active['smr_mean'] = _smr_sir['smr_mean']
            df_active['smr_median'] = _smr_sir['smr_median']
            df_active['smr_lower'] = _smr_sir['smr_lower']
            df_active['smr_upper'] = _smr_sir['smr_upper']
            df_active['sir_mean'] = _smr_sir['sir_mean']
            df_active['sir_lower'] = _smr_sir['sir_lower']
            df_active['sir_upper'] = _smr_sir['sir_upper']
            df_active['exc_prob_smr'] = _smr_sir['exc_prob_smr']
            df_active['exc_prob_sir'] = _smr_sir['exc_prob_sir']
            df_active['exc_prob_smr_low'] = _smr_sir['exc_prob_smr_low']
            df_active['exc_prob_sir_low'] = _smr_sir['exc_prob_sir_low']
            logger.info(
                f"SMR/SIR computed: national_rate_curr={_smr_sir['national_rate_curr']:.4f}, "
                f"EB concentration K={_smr_sir['eb_concentration']:.1f}"
            )

            # Z-scores, then the shared FDR-controlled SMR/SIR classification
            # (audit M2 — see BaseHotspotAnalyzer._finalize_classification).
            df_active = self.calculate_z_scores(df_active, national_rate)
            df_active = self._finalize_classification(df_active, national_rate)

            # Join results back to full GeoDataFrame
            result_cols = ['predicted', 'predicted_prob', 'prob_lower', 'prob_upper', 'residual',
                          'exceedance_prob', 'z_national', 'z_residual', 'combined_z', 'classification',
                          'national_baseline', 'deviation_pct',
                          # SIR/SMR taxonomy outputs:
                          # point summaries and 95% credible intervals for each
                          # ratio, the four exceedance probabilities used by the
                          # taxonomy, the taxonomy label itself, the
                          # new-site flag, and the in-window national rate /
                          # EB-shrunken historical rate kept for traceability.
                          'smr_mean', 'smr_median', 'smr_lower', 'smr_upper',
                          'sir_mean', 'sir_lower', 'sir_upper',
                          'exc_prob_smr', 'exc_prob_sir',
                          'exc_prob_smr_low', 'exc_prob_sir_low',
                          'classification_smr_sir', 'is_new_site',
                          'national_rate_curr', 'baseline_rate_eb',
                          # Combined burden + rate watch-list (add_watchlist).
                          'on_watchlist', 'watch_reason', 'watch_rank',
                          'burden_rank', 'rate_rank', 'burden_share_pct',
                          'burden_high', 'rate_high']

            for col in result_cols:
                if col in df_active.columns:
                    # Use index-based merge instead of .values to avoid misalignment
                    gdf_admin.loc[gdf_admin['site_present'] == True, col] = gdf_admin.loc[gdf_admin['site_present'] == True].index.map(df_active[col])

            # Mark structural zeros as "No Data"
            gdf_admin.loc[gdf_admin['site_present'] == False, 'classification'] = 'No Data'

            # Generate diagnostics
            ppc = pm.sample_posterior_predictive(trace, model=model, progressbar=False, random_seed=self.cfg.get('random_seed', 42))
            diagnostics = self._calculate_diagnostics(trace, df_active, level_name, national_rate, model, ppc, False)
            diagnostics['model_type'] = 'Hurdle Binomial'
            diagnostics['n_structural_zeros'] = int(n_structural_zeros)
            diagnostics['pct_structural_zeros'] = float(pct_structural)
            diagnostics['n_active_sites'] = int(n_active)

            logger.info(f"[OK] Hurdle model completed: {n_active} active sites, {n_structural_zeros} structural zeros")

            return gdf_admin, diagnostics

        except Exception as e:
            logger.error(f"Hurdle model failed: {e}")
            logger.error(traceback.format_exc())
            return gdf_admin, None


# =============================================================================
# BAYESIAN WITH COVARIATES ANALYZER
