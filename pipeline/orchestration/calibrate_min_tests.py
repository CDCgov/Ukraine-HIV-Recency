"""
Data-driven ``min_tests`` threshold.

Group territories by test-count bins, compute the coefficient of
variation of the recent-positive rate per bin, and pick the smallest
bin where CV drops below 1.0 -- the elbow where rate estimates start
to stabilise. The result is clamped to ``[10, 30]`` (literature
default 20) and the per-bin table is written to
``MinTests_Calibration.xlsx`` for the audit trail.

A bin is only considered "stable" when it holds at least 5
territories and both mean and std of the rate are strictly positive,
to avoid mistaking sparse/degenerate bins for stable ones.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calibrate_min_tests(gdf: pd.DataFrame, output_dir: Path) -> int:
    """Return the data-driven ``min_tests`` threshold and write the calibration report."""
    logger.info("\n" + "=" * 60)
    logger.info("MIN_TESTS CALIBRATION")
    logger.info("=" * 60)

    df_active = gdf[gdf['all_tested_curr'] > 0].copy()
    df_active['rate'] = df_active['recent_count_curr'] / df_active['all_tested_curr']

    # Group by test count bins and compute variance
    bin_edges = [0, 3, 5, 10, 15, 20, 25, 30, 50, 100, 999]
    bin_labels = ['0-3', '3-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-50', '50-100', '100+']

    df_active['test_bin'] = pd.cut(df_active['all_tested_curr'], bins=bin_edges, labels=bin_labels, right=True)

    bin_stats = df_active.groupby('test_bin', observed=True).agg(
        n_territories=('rate', 'count'),
        mean_rate=('rate', 'mean'),
        var_rate=('rate', 'var'),
        std_rate=('rate', 'std'),
        mean_tests=('all_tested_curr', 'mean')
    ).reset_index()

    # CV (coefficient of variation) as stability metric
    # Skip bins where:
    #   - mean_rate == 0 (no recent infections — not "stable", just no data)
    #   - std_rate == 0 with mean_rate > 0 (all territories identical rate — degenerate, not stable)
    bin_stats['cv'] = np.where(
        (bin_stats['mean_rate'] > 0) & (bin_stats['std_rate'] > 0),
        bin_stats['std_rate'] / bin_stats['mean_rate'],
        np.nan  # exclude from stability analysis
    )

    logger.info(f"\n  RATE VARIANCE BY TEST COUNT:")
    logger.info(f"  {'Bin':<10} {'N':<6} {'Mean Rate':<12} {'Var Rate':<12} {'CV':<8}")
    for _, row in bin_stats.iterrows():
        if row['n_territories'] >= 2:
            logger.info(f"  {row['test_bin']:<10} {row['n_territories']:<6} "
                        f"{row['mean_rate']:<12.4f} {row['var_rate']:<12.6f} {row['cv']:<8.2f}")

    # Find elbow: where CV drops below 1.0 (rate becomes stable)
    # Require minimum 5 territories per bin — low CV with 2-3 territories is
    # meaningless (not "stable", just insufficient data)
    stable_bins = bin_stats[(bin_stats['cv'] < 1.0) & (bin_stats['n_territories'] >= 5)]
    if len(stable_bins) > 0:
        optimal_min = int(stable_bins.iloc[0]['mean_tests'])
    else:
        # Fallback: use bin with lowest CV that has >= 5 territories
        valid_bins = bin_stats[bin_stats['n_territories'] >= 5]
        if len(valid_bins) > 0:
            optimal_min = int(valid_bins.loc[valid_bins['cv'].idxmin(), 'mean_tests'])
        else:
            optimal_min = 20  # Literature default

    # Clamp to reasonable range (10 minimum — below that, data is too sparse)
    optimal_min = max(10, min(optimal_min, 30))

    logger.info(f"\n  OPTIMAL min_tests: {optimal_min}")
    logger.info(f"  (Default was 20)")

    # Save
    cal_file = output_dir / f'MinTests_Calibration.xlsx'
    bin_stats.to_excel(cal_file, index=False)
    logger.info(f"  Calibration report saved: {cal_file}")

    return optimal_min
