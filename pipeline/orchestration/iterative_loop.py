"""
Run the per-window Bayesian fit loop for iterative mode.

For each sliding window, delegate to the orchestrator's
``_run_bayesian_for_window``, then keep hotspots that pass
``_is_hotspot``, stamp iteration metadata, and accumulate. Per-window
failures (exceptions raised inside the per-window fit) are logged
and skipped -- the loop continues on the next window so a single bad
fit doesn't abort the whole iterative run.

Memory hygiene at the end of each iteration drops the large per-window
result and runs ``gc.collect()``. Cross-window state leakage (the bug a
previous version worked around by clearing PyTensor's private compiled-
function cache) is prevented at the source: every window builds a fresh
``pm.Model()`` and the sampler is seeded with fresh initial points, so no
graph or initial-point state carries over. The undocumented cache hack was
removed (audit Mo4) to avoid a silent break on a PyTensor upgrade.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Callable, Dict, List, Optional

import geopandas as gpd

logger = logging.getLogger(__name__)


def run_iterative_loop(orchestrator: Any,
                       windows: List[Dict[str, Any]],
                       is_hotspot_fn: Callable[[gpd.GeoDataFrame], Any]) -> List[gpd.GeoDataFrame]:
    """Run the per-window fit loop and return the list of per-window hotspot frames."""
    all_hotspots: List[gpd.GeoDataFrame] = []

    for i, window in enumerate(windows):
        logger.info("\n" + "-" * 60)
        logger.info(f"ITERATION {window['iteration']}/{len(windows)}")
        logger.info(f"Analysis: {window['analysis_start'].date()} to {window['analysis_end'].date()}")
        logger.info(f"Baseline: {window['baseline_start'].date()} to {window['baseline_end'].date()}")
        logger.info("-" * 60)

        gdf_result: Optional[gpd.GeoDataFrame] = None
        try:
            # Run Bayesian model for this window
            gdf_result = orchestrator._run_bayesian_for_window(window)

            if gdf_result is not None and len(gdf_result) > 0:
                # Bug fix: guard against missing 'classification' column when model fails
                if 'classification' not in gdf_result.columns:
                    logger.warning(f"[WARN] Iteration {window['iteration']}: no classification column — model likely failed. Skipping.")
                    continue

                # Debug: check if exceedance_prob exists
                if 'exceedance_prob' not in gdf_result.columns:
                    logger.warning(f"[WARN] exceedance_prob missing in gdf_result. Available columns: {gdf_result.columns.tolist()}")

                # Filter only obvious increase
                hotspots = gdf_result[is_hotspot_fn(gdf_result)].copy()

                if len(hotspots) > 0:
                    # Add iteration metadata
                    hotspots['iteration'] = window['iteration']
                    hotspots['analysis_period'] = f"{window['analysis_start'].strftime('%Y-%m')} to {window['analysis_end'].strftime('%Y-%m')}"
                    hotspots['analysis_start_date'] = window['analysis_start']
                    hotspots['analysis_end_date'] = window['analysis_end']

                    # Debug: check reliability scores in hotspots
                    if 'reliability_score' in hotspots.columns:
                        unique_scores = hotspots['reliability_score'].unique()
                        logger.info(f"Hotspots reliability scores: {unique_scores}")

                    all_hotspots.append(hotspots)
                    logger.info(f"[OK] Found {len(hotspots)} hotspot(s) in this iteration")
                else:
                    logger.info("No hotspots found in this iteration")
            else:
                logger.warning("No results from this iteration")

        except Exception as e:
            logger.error(f"Failed iteration {window['iteration']}: {e}")
            continue
        finally:
            # Free memory after each iteration. Each window builds a fresh
            # ``pm.Model()`` and the sampler is given fresh initial points
            # (see ``ParallelSamplingConfig.adaptive_sample``), so cross-window
            # shape/state leakage is prevented at the source. Here we only need
            # to drop the large per-window result and let the collector run.
            #
            # The previous implementation also cleared PyTensor's
            # compiled-function cache through the private
            # ``FunctionMaker._cache``; that undocumented internal could break
            # silently on a library upgrade, so the dependency is removed
            # (audit Mo4) in favour of the source-level safeguards above.
            gdf_result = None
            gc.collect()

    return all_hotspots
