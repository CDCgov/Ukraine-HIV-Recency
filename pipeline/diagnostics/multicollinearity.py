"""
VIF / pairwise-correlation multicollinearity check for covariates.

With a single covariate the check is a no-op (multicollinearity is not
defined on one column); the wired call site in
:meth:`BayesianCovariatesAnalyzer.run_model` still invokes it so that
adding any second covariate immediately picks up the diagnostic without
a follow-up edit.

VIF interpretation (Belsley, Kuh & Welsch 1980; O'Brien 2007):
``< 5`` no problem, ``5--10`` moderate, ``> 10`` severe -- coefficient
identifiability breaks down past 10. Pairwise correlations above
|r| > 0.7 are reported in the diagnostics dict for completeness.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)


def check_multicollinearity(df: pd.DataFrame, covariate_cols: List[str]) -> dict:
    """Check VIF and pairwise correlations among covariates.

    Called from :meth:`run_model` before the PyMC model is built. With a
    single covariate (the current default ``proportion_high_risk``) the
    function returns ``vif_ok=True`` trivially -- multicollinearity is
    not defined on one column -- but the call site is wired up so that
    adding ``testing_intensity`` or any future covariate immediately
    gains the VIF check and the same correlation-pair warning, without
    needing a follow-up edit to ``run_model``.

    VIF interpretation (Belsley, Kuh & Welsch 1980; O'Brien 2007):
    ``< 5`` no problem, ``5--10`` moderate, ``> 10`` severe and the
    coefficient identifiability breaks down.
    """

    if len(covariate_cols) < 2:
        logger.info("Only one covariate - no multicollinearity possible")
        return {
            'vif_ok': True,
            'max_vif': 1.0,
            'n_covariates': 1,
            'multicollinearity_risk': 'None'
        }

    try:

        # Prepare covariate matrix
        X = df[covariate_cols].values

        # Calculate VIF for each covariate
        vif_data = {}
        for i, col in enumerate(covariate_cols):
            vif = variance_inflation_factor(X, i)
            vif_data[col] = float(vif)
            logger.info(f"  VIF for {col}: {vif:.2f}")

        max_vif = max(vif_data.values())

        # VIF interpretation:
        # < 5: No multicollinearity
        # 5-10: Moderate multicollinearity
        # > 10: Severe multicollinearity
        if max_vif > 10:
            logger.error(f"[WARN] SEVERE multicollinearity detected (max VIF={max_vif:.2f})")
            risk_level = 'Severe'
        elif max_vif > 5:
            logger.warning(f"[WARN] Moderate multicollinearity detected (max VIF={max_vif:.2f})")
            risk_level = 'Moderate'
        else:
            logger.info(f"[OK] No problematic multicollinearity (max VIF={max_vif:.2f})")
            risk_level = 'None'

        # Calculate correlation matrix
        corr_matrix = df[covariate_cols].corr()
        logger.info(f"Correlation matrix:\n{corr_matrix}")

        # Find high correlations (|r| > 0.7)
        high_corr_pairs = []
        for i in range(len(covariate_cols)):
            for j in range(i+1, len(covariate_cols)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    high_corr_pairs.append({
                        'var1': covariate_cols[i],
                        'var2': covariate_cols[j],
                        'correlation': float(corr)
                    })

        if high_corr_pairs:
            logger.warning(f"High correlations detected: {high_corr_pairs}")

        return {
            'vif_data': vif_data,
            'max_vif': float(max_vif),
            'vif_ok': max_vif < 5,
            'multicollinearity_risk': risk_level,
            'n_covariates': len(covariate_cols),
            'high_correlations': high_corr_pairs,
            'corr_matrix': corr_matrix.to_dict()
        }

    except (ValueError, np.linalg.LinAlgError) as e:
        logger.warning(f"Could not calculate VIF: {e}")
        return {
            'vif_ok': True,
            'max_vif': None,
            'multicollinearity_risk': 'Unknown',
            'error': str(e)
        }

