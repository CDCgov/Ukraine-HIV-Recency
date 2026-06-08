"""
Standardized comparison ratios used by the SIR/SMR hotspot taxonomy.

``eb_baseline_rate`` shrinks per-territory historical recency proportions
toward the national baseline using a closed-form Empirical Bayes (Beta)
prior. It stabilises rates on small samples (where ``recent_hist /
tested_hist`` would otherwise be wildly noisy) and lifts a zero baseline
gently off the floor so SIR does not explode at the slightest
current-period signal.

``compute_smr_sir`` builds the two standardized comparison ratios from
the posterior probability samples and the data aggregates:

    SMR_i = p_i / national_rate_curr
    SIR_i = p_i / (baseline_rate_i_EB * national_rate_curr / national_rate_baseline)

SMR flags territories whose current recency proportion is high relative
to where the country is right now; SIR flags territories whose proportion
has risen relative to their own history after removing the national
trend. Together they separate "rose recently" (early signal) from
"endemically high" (sustained burden).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def eb_baseline_rate(recent_hist: Iterable[float],
                     tested_hist: Iterable[float],
                     national_rate_baseline: float) -> Tuple[np.ndarray, float]:
    """
    Closed-form Empirical Bayes shrinkage of historical recency proportions.

    Method-of-moments fits a Beta(alpha, beta) prior whose mean equals the
    national baseline proportion. Its concentration K = alpha + beta is
    set from the empirical between-territory variability, so the prior is
    strong when territories look alike (noisy data) and weak when they
    differ (informative data). Each territory's posterior is

        rate_eb_i = (recent_hist_i + alpha) / (tested_hist_i + alpha + beta)

    Args:
        recent_hist: per-territory count of recent infections in baseline.
        tested_hist: per-territory count of tests in baseline.
        national_rate_baseline: national recency proportion over the
            baseline period (the prior mean).

    Returns:
        ``(rate_eb, K)`` -- shrunken rates and the fitted concentration K.
    """
    recent = np.asarray(recent_hist, dtype=float)
    tested = np.asarray(tested_hist, dtype=float)
    p_nat = float(national_rate_baseline)

    # Bounds on prior strength: at least a few effective observations so a
    # zero-baseline territory never sits exactly at 0, and a cap so that
    # genuinely large samples are not over-shrunk.
    K_min, K_max, K_default = 5.0, 500.0, 20.0

    K = K_default
    valid = tested > 0
    if int(valid.sum()) >= 3 and p_nat > 0:
        p_hat_i = recent[valid] / tested[valid]
        s2 = float(np.var(p_hat_i, ddof=1)) if int(valid.sum()) > 1 else 0.0
        mean_bin_var = float(np.mean(p_nat * (1.0 - p_nat) / tested[valid]))
        extra = s2 - mean_bin_var
        if extra > 1e-8 and (p_nat * (1.0 - p_nat)) > extra:
            K = (p_nat * (1.0 - p_nat) - extra) / extra
            K = float(np.clip(K, K_min, K_max))

    alpha_prior = K * p_nat
    beta_prior = K * (1.0 - p_nat)
    rate_eb = (recent + alpha_prior) / (tested + alpha_prior + beta_prior)
    return rate_eb, K


def compute_smr_sir(p_samples,
                    df: pd.DataFrame,
                    national_rate_baseline: float,
                    smr_threshold: float = 2.0,
                    sir_threshold: float = 1.5,
                    smr_low_threshold: float = 0.5,
                    sir_low_threshold: float = 1.0 / 1.5,
                    national_rate_curr_floor: float = 1e-3,
                    ) -> Dict[str, Any]:
    """
    Build SMR / SIR posterior summaries and exceedance probabilities.

    ``national_rate_curr`` is computed inside this helper from the
    in-window totals (``recent_count_curr / all_tested_curr``) and
    clamped to ``national_rate_curr_floor`` when the country-wide
    proportion is so small that the ratios become numerically unstable.

    ``baseline_rate_i`` is supplied by :func:`eb_baseline_rate` so a new
    site without history does not produce an infinite SIR.

    Args:
        p_samples: per-territory posterior probability samples, either
            a list of 1-D arrays (one per territory) or a 2-D ndarray of
            shape ``(n_samples, n_territories)``.
        df: per-territory frame holding ``recent_count_curr``,
            ``all_tested_curr``, ``recent_count_hist``, ``all_tested_hist``.
        national_rate_baseline: national recency proportion over the
            baseline period (denominator of the SIR trend factor).
        smr_threshold / sir_threshold: epidemiological cut-offs for the
            upper tail of each metric.
        smr_low_threshold / sir_low_threshold: cut-offs for the lower
            tail (used to detect decline categories).
        national_rate_curr_floor: minimum value used for the current
            national rate; below it the ratios are too unstable.

    Returns:
        dict with the current-period national rate, the fitted EB
        concentration ``K``, the per-territory point summaries (mean and
        95% credible interval) of SMR and SIR, and the per-territory
        posterior exceedance probabilities ``P(SMR > smr_threshold)``,
        ``P(SIR > sir_threshold)``, ``P(SMR < smr_low_threshold)`` and
        ``P(SIR < sir_low_threshold)``.
    """
    recent_curr_total = float(df['recent_count_curr'].sum())
    tested_curr_total = float(df['all_tested_curr'].sum())
    raw_national_curr = (recent_curr_total / tested_curr_total
                         if tested_curr_total > 0 else 0.0)
    national_rate_curr = max(raw_national_curr, national_rate_curr_floor)
    if raw_national_curr < national_rate_curr_floor:
        logger.warning(
            f"Current national recency proportion {raw_national_curr:.5f} "
            f"below floor {national_rate_curr_floor}; clamped -- "
            "SMR/SIR may be unstable, interpret with caution"
        )

    recent_hist = df['recent_count_hist'].values
    tested_hist = df['all_tested_hist'].values
    baseline_rate_eb, eb_K = eb_baseline_rate(
        recent_hist, tested_hist, national_rate_baseline)

    # Trend factor: the country-wide move between baseline and the
    # analysis window. SIR's expected proportion uses each territory's own
    # shrunken history scaled by this factor, so a national-level shift
    # does not masquerade as a local signal.
    trend_factor = (national_rate_curr / national_rate_baseline
                    if national_rate_baseline > 0 else 1.0)
    sir_denom = np.maximum(baseline_rate_eb * trend_factor,
                           national_rate_curr_floor)

    n_t = len(df)
    smr_mean = np.empty(n_t); smr_lo = np.empty(n_t); smr_hi = np.empty(n_t)
    smr_med = np.empty(n_t)
    sir_mean = np.empty(n_t); sir_lo = np.empty(n_t); sir_hi = np.empty(n_t)
    exc_smr = np.empty(n_t); exc_sir = np.empty(n_t)
    exc_smr_low = np.empty(n_t); exc_sir_low = np.empty(n_t)

    is_list = isinstance(p_samples, list)
    for i in range(n_t):
        p_i = np.asarray(p_samples[i] if is_list else p_samples[:, i])
        smr_i = p_i / national_rate_curr
        sir_i = p_i / sir_denom[i]
        smr_mean[i] = float(np.mean(smr_i))
        # Median is the more honest central estimate on the right-skewed SMR
        # posterior of sparse-count territories (the mean is tail-inflated).
        smr_med[i] = float(np.median(smr_i))
        smr_lo[i] = float(np.percentile(smr_i, 2.5))
        smr_hi[i] = float(np.percentile(smr_i, 97.5))
        sir_mean[i] = float(np.mean(sir_i))
        sir_lo[i] = float(np.percentile(sir_i, 2.5))
        sir_hi[i] = float(np.percentile(sir_i, 97.5))
        exc_smr[i] = float(np.mean(smr_i > smr_threshold))
        exc_sir[i] = float(np.mean(sir_i > sir_threshold))
        exc_smr_low[i] = float(np.mean(smr_i < smr_low_threshold))
        exc_sir_low[i] = float(np.mean(sir_i < sir_low_threshold))

    return {
        'national_rate_curr': national_rate_curr,
        'baseline_rate_eb': baseline_rate_eb,
        'eb_concentration': eb_K,
        'smr_mean': smr_mean, 'smr_median': smr_med,
        'smr_lower': smr_lo, 'smr_upper': smr_hi,
        'sir_mean': sir_mean, 'sir_lower': sir_lo, 'sir_upper': sir_hi,
        'exc_prob_smr': exc_smr, 'exc_prob_sir': exc_sir,
        'exc_prob_smr_low': exc_smr_low, 'exc_prob_sir_low': exc_sir_low,
        'smr_threshold': smr_threshold,
        'sir_threshold': sir_threshold,
        'smr_low_threshold': smr_low_threshold,
        'sir_low_threshold': sir_low_threshold,
    }
