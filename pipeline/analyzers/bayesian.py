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

``run_two_part_model`` is the zero-inflated Beta-Binomial branch for
sparse data with structural zeros (Fu 2023 style): a logistic presence
sub-model separates "structurally absent" territories (many tests, zero
recent events) from low-count sampling zeros, so the presence-weighted
rate cannot inflate into a false rise. It replaces the retired Truncated
Binomial ("Hurdle") model, which fit a weakly-identified per-territory
random slope under a less robust sampling configuration.
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
import pytensor.tensor as pt

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
                # Weakly-informative concentration: let the data pick kappa,
                # including near-Binomial (large kappa). The previous
                # Gamma(3, 0.2) (mean 15) forced overdispersion even on
                # Binomial-like data, weakening the likelihood so strong signals
                # were over-shrunk toward the national rate (a 45/250 hex read as
                # SMR ~1.3, i.e. Normal). See validation/overdispersion notes.
                kappa = pm.Gamma('kappa', alpha=2, beta=0.01)
                y_obs = pm.BetaBinomial('y_obs', alpha=p * kappa, beta=(1 - p) * kappa,
                                        n=n, observed=y)

                # [WARN] IMPROVEMENT 4: Parallel sampling configuration
                sampling_config = ParallelSamplingConfig.get_sampling_config(
                    n_territories=len(df),
                    fast_mode=bool(self.cfg.get('fast_sampling', False)),
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

    def run_two_part_model(self, gdf_admin: gpd.GeoDataFrame, level_name: str,
                           national_rate: float, national_se: float,
                           parametrization: str = 'non_centered') -> Tuple[gpd.GeoDataFrame, dict]:
        """Two-part (zero-inflated Beta-Binomial) detector -- Fu 2023 style.

        Adds a *presence* sub-model on top of the standard Beta-Binomial
        *intensity* model, so a territory's zeros are separated into
        "structurally absent" (no recency signal at all) and "present but a
        low / sampling-zero count". This follows the two-part decomposition of
        Fu et al. (Spatial and seasonal determinants of Lyme borreliosis
        incidence in France, 2016-2021. Euro Surveill. 2023;28(14):2200581),
        who split a zero-heavy outcome into a logistic *presence* part and an
        *intensity* part; here both parts stay in our PyMC Beta-Binomial
        framework (no INLA), and the intensity outcome is the recency
        proportion rather than a Gamma incidence.

        Part 1 (presence): ``logit(pi_i)`` hierarchical across territories.
        Part 2 (intensity): ``logit(p_i) = alpha_i + beta * hist_prop`` exactly
            as in :meth:`run_model`, Beta-Binomial concentration ``kappa``.
        Likelihood: zero-inflated Beta-Binomial, marginalised over the latent
            presence indicator (``pm.CustomDist`` so PPC / diagnostics work).

        The presence-weighted marginal rate ``theta_i = pi_i * p_i`` feeds the
        SMR/SIR taxonomy in place of the raw intensity, so a territory with
        many tests and zero recent events collapses to ``theta ~ 0`` and can no
        longer surface as a false 'Emerging hotspot' (the zero-count
        SIR-inflation failure mode). ``presence_prob`` = E[pi_i] is reported.
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

        # Historical-window counts feed the SHARED presence sub-model (v2): a
        # structurally-absent territory shows zeros in BOTH windows, so pooling
        # the two windows is what identifies presence -- the current window
        # alone (often 0 on a handful of tests) cannot.
        y_h = np.nan_to_num(df['recent_count_hist'].to_numpy(), nan=0.0).astype(int)
        n_h = np.nan_to_num(df['all_tested_hist'].to_numpy(), nan=0.0).astype(int)
        has_hist = n_h > 0

        logger.info(f"\n--- Two-Part (zero-inflated Beta-Binomial) Model for {level_name} ---")
        ever_pos = (y > 0) | (y_h > 0)
        pres_frac = float(np.clip(ever_pos.mean(), 0.05, 0.95))
        logger.info(f"Presence sub-model (both windows): {int(ever_pos.sum())}/{len(df)} territories "
                    f"have >=1 recent event in either window (empirical presence {pres_frac:.2f})")

        def _zibb_logp(value, pi_, p_, kappa_, n_):
            bb = pm.BetaBinomial.dist(alpha=p_ * kappa_, beta=(1 - p_) * kappa_, n=n_)
            logp_bb = pm.logp(bb, value)
            log_pi = pt.log(pi_)
            log1m_pi = pt.log1p(-pi_)
            # y>0 must come from the "present" component; y==0 is a mixture of
            # structural absence (prob 1-pi) and a present-but-zero draw.
            return pt.where(
                pt.gt(value, 0),
                log_pi + logp_bb,
                pt.logaddexp(log1m_pi, log_pi + logp_bb),
            )

        def _zibb_random(pi_, p_, kappa_, n_, rng=None, size=None):
            # Fully vectorised (no float()/int() scalar coercion): PPC vectorises
            # this over posterior draws, so every param arrives as an ndarray and
            # numpy broadcasting must carry the shapes through.
            pi_ = np.asarray(pi_, dtype=float)
            p_ = np.asarray(p_, dtype=float)
            kappa_ = np.asarray(kappa_, dtype=float)
            n_ = np.asarray(n_)
            a = p_ * kappa_
            b = (1.0 - p_) * kappa_
            p_draw = rng.beta(np.maximum(a, 1e-6), np.maximum(b, 1e-6))
            present = rng.binomial(1, np.clip(pi_, 0.0, 1.0))
            counts = rng.binomial(n_.astype(int), p_draw)
            return present * counts

        try:
            with pm.Model() as model:
                prior_mu = pm.math.logit(np.clip(
                    national_rate,
                    ANALYSIS_CONSTANTS['prior_mu_logit_clip_min']['value'],
                    ANALYSIS_CONSTANTS['prior_mu_logit_clip_max']['value']))

                # --- Part 2: intensity given presence (mirrors run_model) ---
                mu_alpha = pm.Normal('mu_alpha', mu=prior_mu, sigma=sigma_hyperprior)
                sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=2)
                beta = pm.Normal('beta', mu=0, sigma=sigma_hyperprior)
                if parametrization == 'non_centered':
                    alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=len(df))
                    alpha = pm.Deterministic('alpha', mu_alpha + sigma_alpha * alpha_offset)
                else:
                    alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, shape=len(df))
                logit_p = alpha + beta * hist_prop
                p = pm.Deterministic('p', pm.math.invlogit(logit_p))
                # Weakly-informative concentration: let the data pick kappa,
                # including near-Binomial (large kappa). The previous
                # Gamma(3, 0.2) (mean 15) forced overdispersion even on
                # Binomial-like data, weakening the likelihood so strong signals
                # were over-shrunk toward the national rate (a 45/250 hex read as
                # SMR ~1.3, i.e. Normal). See validation/overdispersion notes.
                kappa = pm.Gamma('kappa', alpha=2, beta=0.01)

                # Separate historical-window intensity: shares the hyperprior
                # (mu_alpha, sigma_alpha) but has its OWN per-territory offsets,
                # so pooling history for presence does not force the current rate
                # to equal the past -- the current-vs-past change SMR/SIR detects
                # is preserved.
                alpha_h_offset = pm.Normal('alpha_h_offset', mu=0, sigma=1, shape=len(df))
                alpha_h = pm.Deterministic('alpha_h', mu_alpha + sigma_alpha * alpha_h_offset)
                p_hist = pm.Deterministic('p_hist', pm.math.invlogit(alpha_h))

                # --- Part 1: presence (structural zero vs present), SHARED by
                #     both windows so it is identified from pooled evidence. ---
                gamma_mu = pm.Normal('gamma_mu', mu=float(np.log(pres_frac / (1 - pres_frac))),
                                     sigma=1.5)
                gamma_sigma = pm.HalfNormal('gamma_sigma', sigma=1.5)
                gamma_offset = pm.Normal('gamma_offset', mu=0, sigma=1, shape=len(df))
                pi = pm.Deterministic('pi', pm.math.invlogit(gamma_mu + gamma_sigma * gamma_offset))

                # Presence-weighted CURRENT marginal recency rate -> drives SMR/SIR.
                theta = pm.Deterministic('theta', pi * p)

                # Current-window ZIBB as the observed distribution (drives PPC /
                # diagnostics).
                pm.CustomDist('y_obs', pi, p, kappa, n,
                              logp=_zibb_logp, random=_zibb_random, observed=y)
                # Historical-window ZIBB: shared presence pi, own intensity p_hist.
                # Added as a Potential masked to territories that actually have a
                # baseline, so it informs pi / kappa without a second PPC stream.
                _hist_mask = pt.as_tensor_variable(has_hist.astype(float))
                _n_h_safe = pt.as_tensor_variable(np.maximum(n_h, 1))
                pm.Potential('zibb_hist_ll',
                             (_zibb_logp(y_h, pi, p_hist, kappa, _n_h_safe) * _hist_mask).sum())

                sampling_config = ParallelSamplingConfig.get_sampling_config(
                    n_territories=len(df), fast_mode=bool(self.cfg.get('fast_sampling', False)),
                    cores_override=self.cfg.get('sampling', {}).get('cores'),
                )
                draws = sampling_config['draws']
                tune = sampling_config['tune']
                chains = sampling_config['chains']
                cores = sampling_config['cores']
                target_accept = max(sampling_config['target_accept'], 0.95)
                logger.info(f"Using optimized sampling: {chains} chains, {draws} draws, {cores} cores")
                progress_callback = SamplingProgressBar.create_progress_callback()
                logger.info("Sampling from posterior (two-part model; this may take a few minutes)...")

                trace, sampling_info = ParallelSamplingConfig.adaptive_sample(
                    model=model, initial_target_accept=target_accept,
                    draws=draws, tune=tune, chains=chains, cores=cores,
                    random_seed=self.cfg.get('random_seed', 42),
                    progressbar=False,
                    callback=progress_callback if progress_callback else None,
                )

                if sampling_info['adapted']:
                    logger.info(f"[OK] Adaptive sampling: {sampling_info['n_attempts']} attempts, "
                                f"final target_accept={sampling_info['final_target_accept']:.2f}, "
                                f"divergences={sampling_info['divergence_pct']:.1f}%")
                else:
                    logger.info(f"[OK] Sampling completed (divergences={sampling_info['divergence_pct']:.1f}%)")

                convergence_fatal = False
                if sampling_info['divergence_pct'] > 5.0:
                    logger.error("[WARN] CRITICAL: TWO-PART MODEL CONVERGENCE FAILED "
                                 f"(divergences={sampling_info['divergence_pct']:.1f}%) - RESULTS UNRELIABLE")
                    convergence_fatal = True

            saved_model = model
            logger.info("[OK] Two-part model converged")

            # --- posterior extraction: theta (presence-weighted rate) drives
            #     every downstream number, presence_prob is reported alongside.
            theta_samples = trace.posterior['theta'].values.reshape(-1, len(df))
            pi_mean = trace.posterior['pi'].mean(dim=['chain', 'draw']).values
            theta_mean = theta_samples.mean(axis=0)

            df['presence_prob'] = pi_mean
            df['predicted_prob'] = theta_mean
            df['predicted'] = theta_mean * n
            df['residual'] = y - df['predicted']
            df['prob_lower'] = np.percentile(theta_samples, 2.5, axis=0)
            df['prob_upper'] = np.percentile(theta_samples, 97.5, axis=0)

            rng = np.random.default_rng(self.cfg.get('random_seed', 42))
            df['count_lower'] = [float(np.percentile(rng.binomial(n[i], theta_samples[:, i]), 2.5))
                                 for i in range(len(df))]
            df['count_upper'] = [float(np.percentile(rng.binomial(n[i], theta_samples[:, i]), 97.5))
                                 for i in range(len(df))]
            df['exceedance_prob'] = (theta_samples > national_rate).mean(axis=0)

            _dt = (self.cfg or {}).get('detection', {}) if isinstance(self.cfg, dict) else {}
            p_samples_list = [theta_samples[:, i] for i in range(len(df))]
            _smr_sir = BaseHotspotAnalyzer._compute_smr_sir(
                p_samples_list, df, national_rate,
                smr_threshold=float(_dt.get('smr_threshold', 2.0)),
                sir_threshold=float(_dt.get('sir_threshold', 1.5)),
                leave_one_out=bool((self.cfg or {}).get('smr_leave_one_out', False)),
            )
            for _k in ('national_rate_curr', 'baseline_rate_eb', 'smr_mean', 'smr_median',
                       'smr_lower', 'smr_upper', 'sir_mean', 'sir_lower', 'sir_upper',
                       'exc_prob_smr', 'exc_prob_sir', 'exc_prob_smr_low', 'exc_prob_sir_low'):
                df[_k] = _smr_sir[_k]
            eb_K = float(_smr_sir.get('eb_concentration', 0.0) or 0.0)
            df['sir_informative'] = df['all_tested_hist'].fillna(0).astype(float) > eb_K
            logger.info(
                f"SMR/SIR (two-part): national_rate_curr={_smr_sir['national_rate_curr']:.4f}, "
                f"mean presence={float(pi_mean.mean()):.2f}, EB K={eb_K:.1f}"
            )

            df = self.calculate_z_scores(df, national_rate)
            df = self._finalize_classification(df, national_rate)

            result_cols = ['predicted', 'predicted_prob', 'prob_lower', 'prob_upper', 'residual',
                           'exceedance_prob', 'z_national', 'z_residual', 'combined_z', 'classification',
                           'national_baseline', 'deviation_pct',
                           'presence_prob',  # two-part: P(recency signal present)
                           'smr_mean', 'smr_median', 'smr_lower', 'smr_upper',
                           'sir_mean', 'sir_lower', 'sir_upper',
                           'exc_prob_smr', 'exc_prob_sir',
                           'exc_prob_smr_low', 'exc_prob_sir_low',
                           'classification_smr_sir', 'is_new_site',
                           'national_rate_curr', 'baseline_rate_eb',
                           'on_watchlist', 'watch_reason', 'watch_rank',
                           'burden_rank', 'rate_rank', 'burden_share_pct',
                           'burden_high', 'rate_high']
            for col in result_cols:
                if col in df.columns:
                    gdf_admin.loc[gdf_admin['all_tested_curr'] > 0, col] = \
                        gdf_admin.loc[gdf_admin['all_tested_curr'] > 0].index.map(df[col])
            if 'sir_informative' in df.columns:
                gdf_admin.loc[gdf_admin['all_tested_curr'] > 0, 'sir_informative'] = \
                    gdf_admin.loc[gdf_admin['all_tested_curr'] > 0].index.map(df['sir_informative'])

            logger.info("Generating posterior predictive samples...")
            with saved_model:
                ppc = pm.sample_posterior_predictive(
                    trace, progressbar=False, random_seed=self.cfg.get('random_seed', 42))

            diagnostics = self._calculate_diagnostics(
                trace, df, level_name, national_rate, saved_model, ppc, convergence_fatal)
            diagnostics['model'] = saved_model
            diagnostics['trace'] = trace
            diagnostics['y_obs'] = y
            diagnostics['ppc'] = ppc
            diagnostics['model_type'] = 'Two-Part Zero-Inflated Beta-Binomial'
            diagnostics['mean_presence_prob'] = float(pi_mean.mean())
            return gdf_admin, diagnostics

        except (ValueError, RuntimeError, KeyError) as e:
            logger.error(f"Two-part model failed: {e}")
            logger.error(traceback.format_exc())
            return gdf_admin, None

    def run_two_period_model(self, gdf_admin: gpd.GeoDataFrame, level_name: str,
                             national_rate: float, national_se: float,
                             parametrization: str = 'non_centered') -> Tuple[gpd.GeoDataFrame, dict]:
        """Joint two-period model -- the trend axis as a single hierarchical fit.

        Fits both windows at once with

            logit(p_it) = mu_t + a_i + delta_i * I(t = current)

        where ``mu_t`` is the national logit level in each period (baseline vs
        current national trend), ``a_i`` is a hierarchical territory intercept
        and ``delta_i`` is a hierarchical territory-specific current-period
        change. Each territory contributes TWO Beta-Binomial observations
        (history and current), so ``a_i`` and ``delta_i`` are jointly identified
        only because ``delta_i`` partially pools toward a common trend
        ``mu_delta`` -- that shrinkage is what keeps the per-territory slope
        well-behaved (the failure mode of the retired Hurdle model, which used a
        free per-territory slope on a single observation).

        This replaces the two-step SIR machinery (fit the current window, then
        take a ratio against a separately EB-shrunk history): the trend is read
        DIRECTLY from the posterior of ``delta_i``. The "vs national now" axis
        (SMR) still comes from the current-period rate ``p_curr_i``; the trend
        axis (SIR) is ``exp(delta_i)``, the multiplicative current-vs-history
        change (approximately the recency-proportion ratio at low prevalence).
        Zero-count territories are handled by the taxonomy presence gate as
        before. Enabled with the config flag ``two_period_model``.
        """
        prep = prepare_bayesian_inputs(
            self.cfg, gdf_admin, level_name, national_rate, national_se, parametrization,
        )
        if prep is None:
            return gdf_admin, None
        df = prep['df']
        y = prep['y']
        n = prep['n']
        parametrization = prep['parametrization']
        sigma_hyperprior = prep['sigma_hyperprior']

        y_h = np.nan_to_num(df['recent_count_hist'].to_numpy(), nan=0.0).astype(int)
        n_h = np.nan_to_num(df['all_tested_hist'].to_numpy(), nan=0.0).astype(int)
        has_hist = n_h > 0
        n_terr = len(df)

        logger.info(f"\n--- Joint Two-Period Model for {level_name} ---")
        logger.info(f"{int(has_hist.sum())}/{n_terr} territories have a baseline (history) window")

        _clip = float(np.clip(national_rate,
                              ANALYSIS_CONSTANTS['prior_mu_logit_clip_min']['value'],
                              ANALYSIS_CONSTANTS['prior_mu_logit_clip_max']['value']))
        prior_mu = float(np.log(_clip / (1.0 - _clip)))

        try:
            with pm.Model() as model:
                # Period national logit levels (baseline vs current).
                mu_hist = pm.Normal('mu_hist', mu=prior_mu, sigma=1.5)
                mu_curr = pm.Normal('mu_curr', mu=prior_mu, sigma=1.5)

                # Hierarchical territory intercept and current-period change.
                sigma_a = pm.HalfNormal('sigma_a', sigma=2)
                mu_delta = pm.Normal('mu_delta', mu=0.0, sigma=1.0)
                # Between-territory trend SD. Kept deliberately permissive: a tight
                # prior over-shrinks genuine risers toward the common trend and the
                # model then never clears the FDR-controlled exceedance cutoff (a
                # simulation-confirmed failure mode). 1.0 on the logit scale allows
                # a territory to move well away from the pool when the data support it.
                sigma_delta = pm.HalfNormal('sigma_delta', sigma=1.0)

                if parametrization == 'non_centered':
                    a_offset = pm.Normal('a_offset', mu=0, sigma=1, shape=n_terr)
                    a = pm.Deterministic('a', sigma_a * a_offset)
                    d_offset = pm.Normal('d_offset', mu=0, sigma=1, shape=n_terr)
                    delta = pm.Deterministic('delta', mu_delta + sigma_delta * d_offset)
                else:
                    a = pm.Normal('a', mu=0, sigma=sigma_a, shape=n_terr)
                    delta = pm.Normal('delta', mu=mu_delta, sigma=sigma_delta, shape=n_terr)

                p_hist = pm.Deterministic('p_hist', pm.math.invlogit(mu_hist + a))
                p_curr = pm.Deterministic('p_curr', pm.math.invlogit(mu_curr + a + delta))
                # Weakly-informative concentration: let the data pick kappa,
                # including near-Binomial (large kappa). The previous
                # Gamma(3, 0.2) (mean 15) forced overdispersion even on
                # Binomial-like data, weakening the likelihood so strong signals
                # were over-shrunk toward the national rate (a 45/250 hex read as
                # SMR ~1.3, i.e. Normal). See validation/overdispersion notes.
                kappa = pm.Gamma('kappa', alpha=2, beta=0.01)

                # Current window: observed Beta-Binomial (drives PPC / diagnostics).
                pm.BetaBinomial('y_obs', alpha=p_curr * kappa, beta=(1 - p_curr) * kappa,
                                n=n, observed=y)
                # History window: a masked Potential so it informs a_i / mu_hist /
                # kappa for territories that actually have a baseline, without a
                # second posterior-predictive stream.
                _hist_mask = pt.as_tensor_variable(has_hist.astype(float))
                _n_h_safe = pt.as_tensor_variable(np.maximum(n_h, 1))
                _bb_h = pm.BetaBinomial.dist(alpha=p_hist * kappa, beta=(1 - p_hist) * kappa,
                                             n=_n_h_safe)
                pm.Potential('y_hist_ll',
                             (pm.logp(_bb_h, pt.as_tensor_variable(y_h)) * _hist_mask).sum())

                sampling_config = ParallelSamplingConfig.get_sampling_config(
                    n_territories=n_terr, fast_mode=bool(self.cfg.get('fast_sampling', False)),
                    cores_override=self.cfg.get('sampling', {}).get('cores'),
                )
                draws = sampling_config['draws']
                tune = sampling_config['tune']
                chains = sampling_config['chains']
                cores = sampling_config['cores']
                target_accept = max(sampling_config['target_accept'], 0.95)
                logger.info(f"Using optimized sampling: {chains} chains, {draws} draws, {cores} cores")
                progress_callback = SamplingProgressBar.create_progress_callback()
                logger.info("Sampling from posterior (two-period model; this may take a few minutes)...")

                trace, sampling_info = ParallelSamplingConfig.adaptive_sample(
                    model=model, initial_target_accept=target_accept,
                    draws=draws, tune=tune, chains=chains, cores=cores,
                    random_seed=self.cfg.get('random_seed', 42),
                    progressbar=False,
                    callback=progress_callback if progress_callback else None,
                )

                if sampling_info['adapted']:
                    logger.info(f"[OK] Adaptive sampling: {sampling_info['n_attempts']} attempts, "
                                f"final target_accept={sampling_info['final_target_accept']:.2f}, "
                                f"divergences={sampling_info['divergence_pct']:.1f}%")
                else:
                    logger.info(f"[OK] Sampling completed (divergences={sampling_info['divergence_pct']:.1f}%)")

                convergence_fatal = False
                if sampling_info['divergence_pct'] > 5.0:
                    logger.error("[WARN] CRITICAL: TWO-PERIOD MODEL CONVERGENCE FAILED "
                                 f"(divergences={sampling_info['divergence_pct']:.1f}%) - RESULTS UNRELIABLE")
                    convergence_fatal = True

            saved_model = model
            logger.info("[OK] Two-period model converged")

            p_curr_samples = trace.posterior['p_curr'].values.reshape(-1, n_terr)
            delta_samples = trace.posterior['delta'].values.reshape(-1, n_terr)
            p_curr_mean = p_curr_samples.mean(axis=0)

            df['predicted_prob'] = p_curr_mean
            df['predicted'] = p_curr_mean * n
            df['residual'] = y - df['predicted']
            df['prob_lower'] = np.percentile(p_curr_samples, 2.5, axis=0)
            df['prob_upper'] = np.percentile(p_curr_samples, 97.5, axis=0)
            rng = np.random.default_rng(self.cfg.get('random_seed', 42))
            df['count_lower'] = [float(np.percentile(rng.binomial(n[i], p_curr_samples[:, i]), 2.5))
                                 for i in range(n_terr)]
            df['count_upper'] = [float(np.percentile(rng.binomial(n[i], p_curr_samples[:, i]), 97.5))
                                 for i in range(n_terr)]
            df['exceedance_prob'] = (p_curr_samples > national_rate).mean(axis=0)

            # SMR (vs national now) from the current-period rate.
            _dt = (self.cfg or {}).get('detection', {}) if isinstance(self.cfg, dict) else {}
            p_samples_list = [p_curr_samples[:, i] for i in range(n_terr)]
            _smr_sir = BaseHotspotAnalyzer._compute_smr_sir(
                p_samples_list, df, national_rate,
                smr_threshold=float(_dt.get('smr_threshold', 2.0)),
                sir_threshold=float(_dt.get('sir_threshold', 1.5)),
                leave_one_out=bool((self.cfg or {}).get('smr_leave_one_out', False)),
            )
            for _k in ('national_rate_curr', 'baseline_rate_eb', 'smr_mean', 'smr_median',
                       'smr_lower', 'smr_upper', 'exc_prob_smr', 'exc_prob_smr_low'):
                df[_k] = _smr_sir[_k]
            eb_K = float(_smr_sir.get('eb_concentration', 0.0) or 0.0)
            df['sir_informative'] = df['all_tested_hist'].fillna(0).astype(float) > eb_K

            # Trend axis read DIRECTLY from the joint model: exp(delta_i) is the
            # multiplicative current-vs-history change. Territories without a
            # baseline keep delta ~ mu_delta (pooled), and the taxonomy's
            # sir_informative gate withholds a trend claim there.
            sir_threshold = float(_dt.get('sir_threshold', 1.5))
            ratio = np.exp(delta_samples)
            df['sir_mean'] = ratio.mean(axis=0)
            df['sir_lower'] = np.percentile(ratio, 2.5, axis=0)
            df['sir_upper'] = np.percentile(ratio, 97.5, axis=0)
            df['exc_prob_sir'] = (delta_samples > np.log(sir_threshold)).mean(axis=0)
            df['exc_prob_sir_low'] = (delta_samples < -np.log(sir_threshold)).mean(axis=0)
            logger.info(
                f"Two-period: national_rate_curr={_smr_sir['national_rate_curr']:.4f}, "
                f"mean trend exp(delta)={float(np.exp(delta_samples.mean())):.2f}, EB K={eb_K:.1f}"
            )

            df = self.calculate_z_scores(df, national_rate)
            df = self._finalize_classification(df, national_rate)

            result_cols = ['predicted', 'predicted_prob', 'prob_lower', 'prob_upper', 'residual',
                           'exceedance_prob', 'z_national', 'z_residual', 'combined_z', 'classification',
                           'national_baseline', 'deviation_pct',
                           'smr_mean', 'smr_median', 'smr_lower', 'smr_upper',
                           'sir_mean', 'sir_lower', 'sir_upper',
                           'exc_prob_smr', 'exc_prob_sir',
                           'exc_prob_smr_low', 'exc_prob_sir_low',
                           'classification_smr_sir', 'is_new_site',
                           'national_rate_curr', 'baseline_rate_eb',
                           'on_watchlist', 'watch_reason', 'watch_rank',
                           'burden_rank', 'rate_rank', 'burden_share_pct',
                           'burden_high', 'rate_high']
            for col in result_cols:
                if col in df.columns:
                    gdf_admin.loc[gdf_admin['all_tested_curr'] > 0, col] = \
                        gdf_admin.loc[gdf_admin['all_tested_curr'] > 0].index.map(df[col])
            if 'sir_informative' in df.columns:
                gdf_admin.loc[gdf_admin['all_tested_curr'] > 0, 'sir_informative'] = \
                    gdf_admin.loc[gdf_admin['all_tested_curr'] > 0].index.map(df['sir_informative'])

            logger.info("Generating posterior predictive samples...")
            with saved_model:
                ppc = pm.sample_posterior_predictive(
                    trace, progressbar=False, random_seed=self.cfg.get('random_seed', 42))
            diagnostics = self._calculate_diagnostics(
                trace, df, level_name, national_rate, saved_model, ppc, convergence_fatal)
            diagnostics['model'] = saved_model
            diagnostics['trace'] = trace
            diagnostics['y_obs'] = y
            diagnostics['ppc'] = ppc
            diagnostics['model_type'] = 'Joint Two-Period Beta-Binomial'
            return gdf_admin, diagnostics

        except (ValueError, RuntimeError, KeyError) as e:
            logger.error(f"Two-period model failed: {e}")
            logger.error(traceback.format_exc())
            return gdf_admin, None
