"""
Per-run textual summary printed at the end of each model fit.

The :func:`print_summary` routine emits a 5- to 10-line block to the run
log with the headline numbers from a single model run: the model name,
level, convergence flag, ``R-hat`` max and the classification histogram.
It is intentionally tiny -- the reporting Excel and dashboards carry
the detail; this is what an operator scanning the log sees first.
"""

from __future__ import annotations

import logging
from typing import Optional

import geopandas as gpd

logger = logging.getLogger(__name__)


def print_summary(gdf_admin: gpd.GeoDataFrame, level_name: str,
                  model_name: str, converged: bool, period_str: str,
                  rhat_max: Optional[float] = None) -> str:
    """Render and log the per-run summary block; return the text."""
    active = gdf_admin[gdf_admin['all_tested_curr'] > 0]

    summary = []
    summary.append("=" * 60)
    summary.append(f"{model_name} SUMMARY - {level_name}")
    summary.append("=" * 60)
    summary.append(f"Period: {period_str}")
    summary.append(f"Converged: {converged}")
    summary.append(f"R-hat max: {rhat_max:.4f}" if rhat_max else "R-hat max: N/A")
    summary.append(f"Territories analyzed: {len(active)}")

    if 'classification' in gdf_admin.columns:
        summary.append("\nClassification distribution (SMR/SIR taxonomy):")
        for cat in ["Established hotspot", "Emerging hotspot", "Stable high-burden",
                    "Declining from high-burden", "Emerging decrease",
                    "Significant decrease", "Normal", "No Data"]:
            count = (gdf_admin['classification'] == cat).sum()
            if count > 0:
                summary.append(f"  {cat}: {count}")

    text = '\n'.join(summary)
    logger.info(text)

    return text
