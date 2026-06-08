"""
Append the per-model summary block and update the running
"selected model" trio (``model_used`` / ``final_diag`` /
``final_gdf``).

Same shape for the standard Bayesian and the Bayesian Covariates
paths: build a short text summary, log it, append to
``all_summaries``, and -- when convergence and quality both pass
-- promote this model to the running selection. When they fail,
emit a warning and keep whatever was selected before.

The warning includes a "Keeping previous model" line only when
``report_kept_previous=True``; the standard Bayesian path passes
``False`` because it is always the first model to be tried, so
"previous" would be meaningless.

Returns the updated ``(model_used, final_diag, final_gdf)`` --
the caller rebinds the three so the next model stage sees the
right state.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd

logger = logging.getLogger(__name__)


def finalize_model_choice(diag: Optional[Dict[str, Any]],
                          gdf: Optional[gpd.GeoDataFrame],
                          level_name: str,
                          model_label: str,
                          model_used: str,
                          final_diag: Optional[Dict[str, Any]],
                          final_gdf: Optional[gpd.GeoDataFrame],
                          all_summaries: List[str],
                          report_kept_previous: bool = False) -> Tuple[str, Optional[Dict[str, Any]], Optional[gpd.GeoDataFrame]]:
    """Log summary, promote ``model_label`` to selected if it passes quality gates."""
    if not diag:
        return model_used, final_diag, final_gdf

    summary = f"\n{'='*60}\n{model_label} Summary - {level_name}\n{'='*60}\n"
    summary += f"Converged: {diag.get('convergence_ok', 'Unknown')}\n"
    summary += f"Territories: {diag.get('n_territories', 'N/A')}\n"
    logger.info(summary)
    all_summaries.append(summary)

    convergence_ok = diag.get('convergence_ok', 'Unknown')
    overall_quality = diag.get('overall_quality', 'UNKNOWN')

    if convergence_ok == 'No' or overall_quality == 'POOR':
        logger.warning("="*80)
        logger.warning(f"[WARN] MODEL SELECTION WARNING: {model_label} has poor quality")
        logger.warning(f"   Convergence: {convergence_ok}, Quality: {overall_quality}")
        if report_kept_previous:
            logger.warning(f"   Keeping previous model: {model_used}")
        logger.warning("="*80)
        return model_used, final_diag, final_gdf

    logger.info(f"[OK] Model selection: {model_label} (convergence: {convergence_ok}, quality: {overall_quality})")
    return model_label, diag, gdf
