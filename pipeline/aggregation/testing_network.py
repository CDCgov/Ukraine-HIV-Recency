"""
Testing-network characterisation: testing intensity and network stability.

* :func:`calculate_testing_intensity` -- "test-months", i.e. intensity
  averaged over *active* months (months with at least one test). Robust to a
  site opening / closing partway through the window; the two-period detector
  normalises the recency level for it.
* :func:`classify_network_stability` -- z-score-based stability flag comparing
  this territory's intensity change to the typical change across all
  territories at the same level.
"""

from __future__ import annotations

from typing import Any, Dict

import geopandas as gpd
import numpy as np
import pandas as pd


def calculate_testing_intensity(gdf_cases: gpd.GeoDataFrame,
                                start_date: pd.Timestamp,
                                end_date: pd.Timestamp) -> Dict[str, Any]:
    """Weighted testing-intensity (testo-months) for a period.

    Returns ``weighted_intensity`` (tests per *active* month, robust to
    sites opening / closing mid-period), ``n_active_months``,
    ``n_calendar_months`` and the monthly counts list.
    """
    period_cases = gdf_cases[
        (gdf_cases['test_date'] >= start_date) &
        (gdf_cases['test_date'] <= end_date)
    ]

    if len(period_cases) == 0:
        return {
            'weighted_intensity': 0,
            'n_active_months': 0,
            'n_calendar_months': 0,
            'monthly_tests': [],
        }

    period_cases = period_cases.copy()
    period_cases['year_month'] = period_cases['test_date'].dt.to_period('M')
    monthly_counts = period_cases.groupby('year_month').size()

    n_active_months = len(monthly_counts)
    total_tests = monthly_counts.sum()
    weighted_intensity = total_tests / n_active_months if n_active_months > 0 else 0

    n_calendar_months = ((end_date.year - start_date.year) * 12 +
                         (end_date.month - start_date.month) + 1)

    return {
        'weighted_intensity': weighted_intensity,
        'n_active_months': n_active_months,
        'n_calendar_months': n_calendar_months,
        'monthly_tests': monthly_counts.tolist(),
    }


def classify_network_stability(intensity_curr: float, intensity_hist: float,
                               all_intensities_curr: np.ndarray,
                               all_intensities_hist: np.ndarray) -> Dict[str, Any]:
    """Stability flag from a z-score on relative intensity change.

    The z-score compares this territory's relative change to the
    distribution of relative changes across all territories at the same
    level. With < 3 territories available the function falls back to a
    fixed-threshold rule (|Δ| < 0.25 stable, < 0.50 moderate, else major).
    """
    if intensity_hist == 0:
        return {
            'stability': 'NO_BASELINE',
            'z_score': None,
            'absolute_change': intensity_curr,
            'relative_change': None,
            'description': 'No baseline testing intensity data',
        }

    absolute_change = intensity_curr - intensity_hist
    relative_change = absolute_change / intensity_hist

    all_changes = []
    for curr, hist in zip(all_intensities_curr, all_intensities_hist):
        if hist > 0:
            change = (curr - hist) / hist
            all_changes.append(change)

    if len(all_changes) < 3:
        if abs(relative_change) < 0.25:
            stability = 'STABLE'
            desc = 'Testing network stable'
        elif abs(relative_change) < 0.50:
            stability = 'MODERATE_CHANGE'
            desc = 'Testing network changed moderately'
        else:
            stability = 'MAJOR_CHANGE'
            desc = 'Testing network changed significantly'

        return {
            'stability': stability,
            'z_score': None,
            'absolute_change': absolute_change,
            'relative_change': relative_change,
            'description': desc,
        }

    mean_change = np.mean(all_changes)
    std_change = np.std(all_changes)

    if std_change == 0:
        z_score = 0
    else:
        z_score = (relative_change - mean_change) / std_change

    abs_z = abs(z_score)

    if abs_z < 1.0:
        stability = 'STABLE'
        desc = 'Testing network stable (change within typical variation)'
    elif abs_z < 2.0:
        stability = 'MODERATE_CHANGE'
        desc = 'Testing network changed moderately (change unusual but not extreme)'
    else:
        stability = 'MAJOR_CHANGE'
        desc = 'Testing network changed significantly (change beyond typical variation)'

    return {
        'stability': stability,
        'z_score': z_score,
        'absolute_change': absolute_change,
        'relative_change': relative_change,
        'description': desc,
        'mean_change_level': mean_change,
        'std_change_level': std_change,
    }

