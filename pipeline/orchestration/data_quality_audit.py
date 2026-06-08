"""
Per-level "Data Quality Assessment" audit-trail block.

For each level, ``run_for_mode`` opens a Data Quality Assessment
stage on the audit trail before any modelling begins. Two
substages -- structural-zeros analysis and sample-size assessment
-- record their thresholded decisions on the trail so the eventual
DecisionAuditTrail report explains why a particular prior
strength or hurdle preference was chosen. The numbers (n_active,
pct_structural_zeros, mean tests) are also computed downstream
for the wizard, so this block is fire-and-forget: it mutates the
audit trail and returns nothing.

Constants here mirror the wizard's own thresholds (70% zeros,
50/100 mean tests) so the trail's text matches the actual
decision boundary.
"""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd

logger = logging.getLogger(__name__)


def assess_data_quality(audit_trail: Any, gdf: gpd.GeoDataFrame) -> None:
    """Open the Data Quality Assessment stage and record its two substage decisions."""
    audit_trail.start_stage(
        "Data Quality Assessment",
        "Analyze data characteristics, structural zeros, sample sizes, and data adequacy"
    )

    # Record basic data statistics
    n_territories = len(gdf)
    n_active_sites = (gdf['all_tested_curr'] > 0).sum()
    n_structural_zeros = n_territories - n_active_sites
    pct_structural_zeros = (n_structural_zeros / n_territories * 100) if n_territories > 0 else 0

    audit_trail.set_metadata(
        n_territories=n_territories,
        n_active_sites=n_active_sites
    )

    # Substage: Structural Zeros Analysis
    audit_trail.start_substage(
        "Structural Zeros Analysis",
        "Check proportion of territories without testing sites"
    )

    audit_trail.add_decision(
        test_name="Structural Zeros Percentage",
        test_type="threshold",
        result=f"{pct_structural_zeros:.1f}% territories have zero tests ({n_structural_zeros}/{n_territories})",
        decision="Consider Hurdle model" if pct_structural_zeros > 70 else "Standard model adequate",
        reason=f"Threshold: 70%. Current: {pct_structural_zeros:.1f}%. High proportion of structural zeros requires special handling.",
        impact="Hurdle model explicitly models excess zeros" if pct_structural_zeros > 70 else "Standard models can handle this level of zeros",
        details={'n_structural_zeros': n_structural_zeros, 'n_territories': n_territories, 'threshold': 70.0},
        substage_name="Structural Zeros Analysis"
    )

    # Substage: Sample Size Assessment
    audit_trail.start_substage(
        "Sample Size Assessment",
        "Evaluate adequacy of sample sizes for statistical inference"
    )

    median_tests = gdf['all_tested_curr'].median()
    mean_tests = gdf['all_tested_curr'].mean()
    total_tests = gdf['all_tested_curr'].sum()

    audit_trail.add_decision(
        test_name="Sample Size Check",
        test_type="diagnostic",
        result=f"Total tests={total_tests:.0f}, Median={median_tests:.0f}, Mean={mean_tests:.1f}, Active sites={n_active_sites}",
        decision="Use informative priors" if mean_tests < 50 else "Use weak priors" if mean_tests < 100 else "Use very weak priors",
        reason=f"Low sample sizes require regularization to prevent overfitting. Mean tests per territory: {mean_tests:.1f}",
        impact="Stronger priors provide regularization for small samples" if mean_tests < 50 else "Weak priors allow data to dominate",
        details={'median_tests': float(median_tests), 'mean_tests': float(mean_tests), 'total_tests': int(total_tests)},
        substage_name="Sample Size Assessment"
    )
