"""
Roll up per-level model results into the dashboard payload.

The :func:`collect_dashboard_data` routine walks the orchestrator's
``results`` dict (one entry per admin / hex level) and produces the
five-bucket payload that :class:`~pipeline.reporting.SummaryDashboard`
expects: hotspot counts per level, model-quality flags, the test-volume
summary, the convergence summary, and the top-ten hotspots ranked by
posterior exceedance probability.

The ``is_hotspot_fn`` argument is the analyzer's hotspot mask helper
(``pipeline.classification.is_hotspot``); passing it explicitly keeps
this module decoupled from the analyzer classes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict


def collect_dashboard_data(results: Dict[str, Any],
                           is_hotspot_fn: Callable) -> Dict[str, Any]:
    """Build the five-bucket dashboard payload from per-level results."""
    hotspots_by_level: Dict[str, Any] = {}
    model_quality: Dict[str, Any] = {}
    test_summary: Dict[str, Any] = {}
    convergence_summary: Dict[str, Any] = {}
    top_hotspots = []

    for level_name, level_data in results.items():
        if not isinstance(level_data, dict):
            continue

        gdf = level_data.get('gdf')
        if gdf is None or len(gdf) == 0:
            continue

        if 'classification' in gdf.columns:
            classifications = gdf['classification'].value_counts().to_dict()
            hotspots_by_level[level_name] = {
                'Established hotspot': classifications.get('Established hotspot', 0),
                'Emerging hotspot': classifications.get('Emerging hotspot', 0),
                'Stable high-burden': classifications.get('Stable high-burden', 0),
                'Declining from high-burden': classifications.get('Declining from high-burden', 0),
                'Emerging decrease': classifications.get('Emerging decrease', 0),
                'Significant decrease': classifications.get('Significant decrease', 0),
                'Normal': classifications.get('Normal', 0),
                'total': len(gdf),
            }

        model_used = level_data.get('model_used', 'Unknown')
        diagnostics = level_data.get('diagnostics', {})

        if diagnostics:
            if 'Bayesian' in model_used:
                model_quality[level_name] = {
                    'model': model_used,
                    'converged': diagnostics.get('convergence_ok', False),
                    'rhat_max': diagnostics.get('rhat_max', 0),
                    'quality': 'Good' if diagnostics.get('convergence_ok', False) else 'Poor',
                }

        if 'all_tested_curr' in gdf.columns and 'recent_count_curr' in gdf.columns:
            test_summary[level_name] = {
                'total_tests': int(gdf['all_tested_curr'].sum()),
                'total_recent': int(gdf['recent_count_curr'].sum()),
                'proportion': float(gdf['recent_count_curr'].sum() / gdf['all_tested_curr'].sum()) if gdf['all_tested_curr'].sum() > 0 else 0,
            }

        if 'classification' in gdf.columns and 'exceedance_prob' in gdf.columns:
            hotspots = gdf[is_hotspot_fn(gdf)].copy()
            if len(hotspots) > 0:
                hotspots = hotspots.nlargest(5, 'exceedance_prob')
                for _, row in hotspots.iterrows():
                    territory_name = row.get('ADM3_EN') or row.get('ADM2_EN') or row.get('ADM1_EN') or row.get('h3_id', 'Unknown')
                    top_hotspots.append({
                        'territory': territory_name,
                        'level': level_name,
                        'exceedance_prob': float(row.get('exceedance_prob', 0.0)),
                        'combined_z': float(row.get('combined_z', 0.0)),
                        'recent_proportion': float(row.get('recent_proportion_curr', 0)),
                    })

    top_hotspots = sorted(top_hotspots, key=lambda x: x['exceedance_prob'], reverse=True)[:10]

    return {
        'hotspots_by_level': hotspots_by_level,
        'model_quality': model_quality,
        'test_summary': test_summary,
        'convergence_summary': convergence_summary,
        'top_hotspots': top_hotspots,
    }
