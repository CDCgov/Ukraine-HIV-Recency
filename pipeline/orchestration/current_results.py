"""
Per-run hotspot snapshot for the historical-comparison file.

The :func:`collect_current_results` routine walks the orchestrator's
``results`` dict (one entry per admin / hex level) and produces a
flat snapshot of:

* the period string,
* a timestamp,
* the pipeline version,
* the total hotspot count,
* the hotspot count per level (Obvious Increase + Slight Increase),
* a list of (territory, level, combined_z, recent_proportion) tuples
  for every Obvious Increase row.

The next run's :class:`~pipeline.history.HistoricalComparison` reads
that snapshot to diff against the new results.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict


def collect_current_results(results: Dict[str, Any], period_str: str,
                            pipeline_version: str,
                            is_hotspot_fn: Callable) -> Dict[str, Any]:
    """Snapshot the orchestrator's per-level results for the comparison file."""
    total_hotspots = 0
    hotspots_by_level: Dict[str, Any] = {}
    territories = []

    for level_name, level_data in results.items():
        if not isinstance(level_data, dict):
            continue

        gdf = level_data.get('gdf')
        if gdf is None or len(gdf) == 0:
            continue

        if 'classification' in gdf.columns:
            obvious_increase = len(gdf[is_hotspot_fn(gdf)])
            # The legacy "Slight Increase" soft tier was retired with the
            # single-axis classifier (audit M4); the SMR/SIR taxonomy has no
            # soft tier, so this is kept at 0 for JSON-schema stability.
            slight_increase = 0

            hotspots_by_level[level_name] = {
                'obvious_increase': obvious_increase,
                'slight_increase': slight_increase,
                'total': obvious_increase + slight_increase,
            }
            total_hotspots += obvious_increase + slight_increase

            if 'combined_z' in gdf.columns:
                hotspots = gdf[is_hotspot_fn(gdf)].copy()
                for _, row in hotspots.iterrows():
                    territory_name = row.get('ADM3_EN') or row.get('ADM2_EN') or row.get('ADM1_EN') or row.get('h3_id', 'Unknown')
                    territories.append({
                        'name': territory_name,
                        'level': level_name,
                        'combined_z': float(row.get('combined_z', 0)),
                        'recent_proportion': float(row.get('recent_proportion_curr', 0)),
                    })

    return {
        'period': period_str,
        'timestamp': datetime.now().isoformat(),
        'pipeline_version': pipeline_version,
        'total_hotspots': total_hotspots,
        'hotspots_by_level': hotspots_by_level,
        'territories': territories,
    }
