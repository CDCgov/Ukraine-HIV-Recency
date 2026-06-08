"""
Automatic data-specification analysis and Bayesian-model recommendation.

The :class:`AutoSpecificationSystem` runs the Cameron--Trivedi dispersion
test (a moment-based replacement for the LOO-IC comparison that previous
versions used) and feeds the result, along with a small set of summary
statistics (zero share, sample-size distribution, outlier share), into a
composite recommendation score. The score is conservative: every model
recommendation it makes today is *Bayesian Hierarchical*; what varies is
the confidence level, which surfaces the warnings the analyst should
attend to before they trust the run.

Thresholds live in :mod:`pipeline.constants` rather than as local magic
numbers because they are scientific defaults (with literature anchors)
that should not need a code change to tune.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

from pipeline.constants import (
    BAYESIAN_SCORE_HIGH_CONFIDENCE,
    BAYESIAN_SCORE_MEDIUM_CONFIDENCE,
    IQR_MULTIPLIER_EXTREME,
    OUTLIER_THRESHOLD_PCT,
    SMALL_SAMPLE_PCT_HIGH,
    SMALL_SAMPLE_PCT_MODERATE,
    SMALL_SAMPLE_SIZE,
    VERY_SMALL_SAMPLE_SIZE,
    ZERO_INFLATION_THRESHOLD_HIGH,
    ZERO_INFLATION_THRESHOLD_MODERATE,
)
from pipeline.validators import validate_dataframe

logger = logging.getLogger(__name__)


class AutoSpecificationSystem:
    """Cameron--Trivedi dispersion test plus composite Bayesian-recommendation score."""

    @staticmethod
    def _fit_beta_binomial_test(df: pd.DataFrame, y_col: str, n_col: str) -> Optional[float]:
        """Cameron--Trivedi dispersion ratio for Binomial overdispersion.

        Computes ``phi = sum((y - n*p_hat)**2) / sum(n*p_hat*(1-p_hat))``
        rescaled by ``k/(k-1)``. Under the Binomial null ``phi ~= 1``;
        values above 1.5 indicate moderate, above 2.0 strong overdispersion.
        Returns ``None`` when the test cannot be computed (fewer than 5
        active sites or boundary ``p_hat``).
        """
        try:
            df_active = df[(df[n_col] > 0) & (df[y_col] >= 0)].copy()

            if len(df_active) < 5:
                logger.warning("Too few active sites (<5) for overdispersion test")
                return None

            y = df_active[y_col].values.astype(float)
            n = df_active[n_col].values.astype(float)
            k = len(y)

            p_hat = y.sum() / n.sum()

            if p_hat <= 0 or p_hat >= 1:
                logger.warning(f"Pooled estimate p̂={p_hat:.4f} at boundary — cannot test overdispersion")
                return None

            expected = n * p_hat
            var_expected = n * p_hat * (1 - p_hat)

            residuals_sq = (y - expected) ** 2
            phi = residuals_sq.sum() / var_expected.sum() * k / (k - 1) if (k > 1) else 1.0

            if phi > 2.0:
                interpretation = "STRONG overdispersion"
            elif phi > 1.5:
                interpretation = "MODERATE overdispersion"
            elif phi > 1.2:
                interpretation = "MILD overdispersion"
            else:
                interpretation = "No significant overdispersion"

            logger.info(f"Cameron-Trivedi test: φ={phi:.3f} ({interpretation})")
            logger.info(f"  Territories: {k}, pooled p̂={p_hat:.4f}")
            logger.info(f"  Σ(y-expected)²={residuals_sq.sum():.1f}, Σ(var_expected)={var_expected.sum():.1f}")

            return float(phi)

        except Exception as e:
            logger.warning(f"Overdispersion test failed: {e}")
            return None

    @staticmethod
    def test_beta_binomial_overdispersion(df: pd.DataFrame, y_col: str, n_col: str):
        """Alias for :meth:`_fit_beta_binomial_test` (kept for back-compat)."""
        return AutoSpecificationSystem._fit_beta_binomial_test(df, y_col, n_col)

    @staticmethod
    def recommend_specification(df: pd.DataFrame, y_col: str, n_col: str) -> Dict[str, Any]:
        """Compute summary statistics and a Bayesian-recommendation score.

        Returns a dict with ``data_analysis`` (zero share, dispersion, etc.),
        a list of ``warnings`` to surface to the user, and the recommended
        model + confidence level.
        """
        validate_dataframe(df, "df", required_columns=[y_col, n_col])

        if not y_col or not n_col:
            raise ValueError("y_col and n_col must be non-empty strings")

        analysis: Dict[str, Any] = {
            'data_analysis': {},
            'warnings': [],
            'recommended_model': 'Bayesian Hierarchical Model',
            'confidence': 'MEDIUM',
            'recommended_distribution': 'Negative Binomial',
        }

        n_territories = len(df)
        total_events = df[y_col].sum()
        total_n = df[n_col].sum()

        df_work = df.copy()
        df_work['proportion'] = df_work[y_col] / df_work[n_col]

        n_zeros = (df_work[y_col] == 0).sum()
        pct_zeros = n_zeros / n_territories * 100

        phi = AutoSpecificationSystem._fit_beta_binomial_test(df_work, y_col, n_col)

        if phi is not None:
            overdispersion_detected = phi > 1.5
            analysis['data_analysis']['dispersion_ratio'] = phi
            analysis['data_analysis']['overdispersion_test'] = 'Cameron-Trivedi (dispersion ratio)'
        else:
            mean_prop = df_work['proportion'].mean()
            var_prop = df_work['proportion'].var()
            mean_n = df_work[n_col].mean()
            expected_var = mean_prop * (1 - mean_prop) / mean_n if mean_n > 0 else 0
            overdispersion_detected = var_prop > 2 * expected_var if expected_var > 0 else False
            analysis['data_analysis']['overdispersion_test'] = 'Variance heuristic (fallback)'

        small_n = (df_work[n_col] < SMALL_SAMPLE_SIZE).sum()
        pct_small_n = small_n / n_territories * 100

        Q1 = df_work['proportion'].quantile(0.25)
        Q3 = df_work['proportion'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df_work['proportion'] < Q1 - IQR_MULTIPLIER_EXTREME * IQR) |
                    (df_work['proportion'] > Q3 + IQR_MULTIPLIER_EXTREME * IQR)).sum()
        pct_outliers = outliers / n_territories * 100

        analysis['data_analysis'].update({
            'n_territories': n_territories,
            'total_events': int(total_events),
            'total_n': int(total_n),
            'overall_rate': total_events / total_n if total_n > 0 else 0,
            'n_zeros': int(n_zeros),
            'pct_zeros': pct_zeros,
            'overdispersion_detected': overdispersion_detected,
            'small_n': int(small_n),
            'pct_small_n': pct_small_n,
            'outliers': int(outliers),
            'pct_outliers': pct_outliers,
        })

        bayesian_score = 0

        if pct_zeros > ZERO_INFLATION_THRESHOLD_HIGH:
            bayesian_score += 2
            analysis['warnings'].append(f"[WARN] {pct_zeros:.1f}% territories have zero events - Bayesian recommended")
        elif pct_zeros > ZERO_INFLATION_THRESHOLD_MODERATE:
            bayesian_score += 1
            analysis['warnings'].append(f"[WARN] {pct_zeros:.1f}% territories have zero events")

        if overdispersion_detected:
            bayesian_score += 1
            if phi is not None:
                analysis['warnings'].append(f"[WARN] Overdispersion detected (φ={phi:.3f}) - Bayesian recommended")
            else:
                analysis['warnings'].append(f"[WARN] Overdispersion detected - Bayesian recommended")

        if pct_small_n > SMALL_SAMPLE_PCT_HIGH:
            bayesian_score += 2
            analysis['warnings'].append(f"[WARN] {pct_small_n:.1f}% territories have small sample sizes (<{SMALL_SAMPLE_SIZE})")
        elif pct_small_n > SMALL_SAMPLE_PCT_MODERATE:
            bayesian_score += 1

        if pct_outliers > OUTLIER_THRESHOLD_PCT:
            bayesian_score += 1
            analysis['warnings'].append(f"[WARN] {pct_outliers:.1f}% territories are extreme outliers")

        if n_territories < VERY_SMALL_SAMPLE_SIZE:
            bayesian_score += 1
            analysis['warnings'].append(f"[WARN] Only {n_territories} territories - limited data")

        if bayesian_score >= BAYESIAN_SCORE_HIGH_CONFIDENCE:
            analysis['recommended_model'] = 'Bayesian Hierarchical Model'
            analysis['confidence'] = 'HIGH'
        elif bayesian_score >= BAYESIAN_SCORE_MEDIUM_CONFIDENCE:
            analysis['recommended_model'] = 'Bayesian Hierarchical Model'
            analysis['confidence'] = 'MEDIUM'
        else:
            analysis['recommended_model'] = 'Bayesian Hierarchical Model'
            analysis['confidence'] = 'MEDIUM' if bayesian_score == 0 else 'LOW'

        return analysis
