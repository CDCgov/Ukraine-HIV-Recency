"""
Per-level data-specification analysis.

Before the Bayesian fit, ``run_for_mode`` runs an auto-specification
pass over the active (non-structural-zero) territories to recommend
a model family and surface obvious red flags (excess zeros,
overdispersion, outliers). Either the heuristic
``AutoSpecificationSystem`` or the LOO-IC model-selection routine
runs, depending on the per-level ``use_loo_ic`` flag. The textual
report is written to ``Specification_Analysis_<level>_<period>.txt``
and a "Data Specification Analysis" decision is recorded on the
audit trail.

The orchestrator is passed in so this routine can reach
``get_output_path`` (per-level/hex directory policy) and
``_run_loo_ic_model_selection`` without importing the orchestrator
class directly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd

from pipeline.spec import AutoSpecificationSystem, ModelSpecificationAnalyzer

logger = logging.getLogger(__name__)


def analyze_specification(orchestrator: Any,
                          gdf: gpd.GeoDataFrame,
                          level_name: str,
                          period_str: str,
                          national_rate: float,
                          national_se: float,
                          level_use_loo_ic: bool,
                          audit_trail: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    """Run specification analysis, write report, record audit decision; return (spec, recommended_model)."""
    logger.info(f"\n--- Data Specification Analysis ---")
    df_analysis = gdf[gdf['all_tested_curr'] > 0].copy()

    spec_analysis: Optional[Dict[str, Any]] = None
    recommended_model = 'Bayesian Hierarchical Model'  # Default

    try:
        # Use LOO-IC based selection if enabled
        if level_use_loo_ic:
            logger.info("Using LOO-IC for model selection")
            spec_analysis = orchestrator._run_loo_ic_model_selection(
                df_analysis, gdf, level_name, national_rate, national_se
            )
        else:
            # Use heuristic-based selection (default)
            spec_analysis = AutoSpecificationSystem.recommend_specification(
                df_analysis,
                y_col='recent_count_curr',
                n_col='all_tested_curr'
            )

        spec_report = ModelSpecificationAnalyzer.generate_specification_report(
            spec_analysis['data_analysis']
        )

        # Save specification report
        is_hex = level_name.startswith("Hex_Res")
        spec_output_file = orchestrator.get_output_path(
            "bayesian", level_name,
            f"Specification_Analysis_{level_name}_{period_str}.txt",
            is_hex=is_hex,
        )
        with open(spec_output_file, 'w', encoding='utf-8') as f:
            f.write(spec_report)
        logger.info(f"[OK] Specification report saved: {spec_output_file}")

        # Log warnings
        for warning in spec_analysis.get('warnings', []):
            logger.warning(warning)

        # Get recommendation
        recommended_model = spec_analysis.get('recommended_model', 'Bayesian Hierarchical Model')
        logger.info(f"Recommended model: {recommended_model}")

        # Record specification analysis decision
        data_analysis = spec_analysis.get('data_analysis', {})
        audit_trail.add_decision(
            test_name="Data Specification Analysis",
            test_type="diagnostic",
            result=f"Recommended: {recommended_model}. " +
                   f"Zeros: {data_analysis.get('pct_zero', 0):.1f}%, " +
                   f"Low counts: {data_analysis.get('pct_low', 0):.1f}%, " +
                   f"Outliers: {data_analysis.get('n_outliers', 0)}",
            decision=f"Recommend {recommended_model} model",
            reason=f"Based on data characteristics: " +
                   f"zero inflation={data_analysis.get('pct_zero', 0):.1f}%, " +
                   f"overdispersion={'Yes' if data_analysis.get('overdispersion', False) else 'No'}, " +
                   f"outliers={data_analysis.get('n_outliers', 0)}",
            impact="Model choice affects bias-variance tradeoff and computational cost",
            details={
                'recommended_model': recommended_model,
                'pct_zero': data_analysis.get('pct_zero', 0),
                'pct_low': data_analysis.get('pct_low', 0),
                'n_outliers': data_analysis.get('n_outliers', 0),
                'overdispersion': data_analysis.get('overdispersion', False)
            }
        )

        # Bayesian is always the model
        if recommended_model == 'Bayesian Hierarchical Model':
            logger.info("Using Bayesian Hierarchical Model")

    except (ValueError, KeyError, IOError) as e:
        logger.error(f"Failed to analyze data specification: {e}")
        spec_analysis = None

    return spec_analysis, recommended_model
