"""
Standardised Z-scores against the national baseline.

The :func:`calculate_z_scores` routine writes three columns onto a
per-territory frame:

* ``z_national`` -- standardised difference between the territory's
  current recency proportion and the national baseline rate, with a
  pooled standard error so the values are comparable across models;
* ``z_residual`` -- standardised residual from whatever predictor the
  model used (``df['residual']`` is computed upstream);
* ``combined_z`` -- the mean of the two, kept for back-compat with the
  legacy single-axis classification.

Structural zeros (``n=0``) produce ``NaN`` rather than 0 -- otherwise
the pooled SE goes to ``sqrt(inf)`` and ``z = 0 / inf = 0`` would mis-
classify a "no data" territory as "no difference".
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_z_scores(df: pd.DataFrame, national_rate: float) -> pd.DataFrame:
    """Add ``z_national`` / ``z_residual`` / ``combined_z`` columns to ``df``."""
    n = df['all_tested_curr'].values
    total_tests = df['all_tested_curr'].sum()

    pooled_se = np.where(
        n > 0,
        np.sqrt(national_rate * (1 - national_rate) * (1 / n + 1 / total_tests)),
        np.nan,
    )
    df['z_national'] = np.where(
        n > 0,
        (df['recent_proportion_curr'] - national_rate) / pooled_se,
        np.nan,
    )

    residual_variance = df['residual'].var()
    if residual_variance > 0:
        residual_se = np.where(
            n > 0,
            np.sqrt(residual_variance * (1 / n + 1 / total_tests)),
            np.nan,
        )
        df['z_residual'] = np.where(n > 0, df['residual'] / residual_se, np.nan)
    else:
        df['z_residual'] = 0

    df['combined_z'] = (df['z_national'] + df['z_residual']) / 2

    return df
