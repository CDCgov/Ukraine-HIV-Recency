"""
STEP 3 of ``run_for_mode``: pool the per-model interpretations,
compute a reliability scorecard for the *selected* model, and --
when at least two models have an interpretation -- emit the
side-by-side Model Comparison report.

The reliability scorecard is built from
:meth:`ReliabilityScoreCalculator.calculate_territory_scores` and
the ``reliability_weights`` block in ``config``; the overall
rating is a population-weighted vote over the per-territory
HIGH / MODERATE / LOW labels (>=50% HIGH -> HIGH; otherwise >=50%
HIGH+MODERATE -> MODERATE; else LOW).

This function only writes a file and logs; it returns ``None``
and does not mutate caller state.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import geopandas as gpd

from pipeline.diagnostics import DiagnosticInterpreter, ReliabilityScoreCalculator

logger = logging.getLogger(__name__)


def _score_interp(score: float) -> str:
    if score >= 80:
        return "Adequate data coverage"
    if score >= 60:
        return "Moderate data coverage"
    return "Limited data coverage"


def _build_reliability_info(final_diag: Dict[str, Any],
                            final_gdf: gpd.GeoDataFrame,
                            config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        df_for_reliability = final_gdf[final_gdf['all_tested_curr'] > 0].copy()
        df_with_scores = ReliabilityScoreCalculator.calculate_territory_scores(
            df_for_reliability, final_diag,
        )

        scores = df_with_scores['reliability_score'].values
        high_count = (df_with_scores['reliability_category'] == 'HIGH').sum()
        mod_count = (df_with_scores['reliability_category'] == 'MODERATE').sum()
        low_count = (df_with_scores['reliability_category'] == 'LOW').sum()

        if high_count / len(df_with_scores) >= 0.5:
            overall_rating = "HIGH"
        elif (high_count + mod_count) / len(df_with_scores) >= 0.5:
            overall_rating = "MODERATE"
        else:
            overall_rating = "LOW"

        if overall_rating == "HIGH":
            flag = "[OK]"
            recommendation = "Results are reliable for decision-making"
        elif overall_rating == "MODERATE":
            flag = "[WARN]"
            recommendation = "Results are acceptable but interpret with caution"
        else:
            flag = "[FAIL]"
            recommendation = "Results have high uncertainty - use with caution"

        weights = config.get('reliability_weights', {})
        w_data = weights.get('data_adequacy', 40) / 100.0
        w_sample = weights.get('sample_size', 30) / 100.0
        w_model = weights.get('model_quality', 30) / 100.0

        data_mean = df_with_scores['data_adequacy_score'].mean() if 'data_adequacy_score' in df_with_scores.columns else 0
        sample_mean = df_with_scores['sample_size_score'].mean() if 'sample_size_score' in df_with_scores.columns else 0
        model_mean = df_with_scores['model_quality_score'].mean() if 'model_quality_score' in df_with_scores.columns else 0

        reliability_info = {
            'overall_score': round(scores.mean(), 1),
            'rating': overall_rating,
            'flag': flag,
            'recommendation': recommendation,
            'min_score': round(scores.min(), 1),
            'max_score': round(scores.max(), 1),
            'distribution': {
                'HIGH': int(high_count),
                'MODERATE': int(mod_count),
                'LOW': int(low_count),
            },
            'components': {
                'data_adequacy': {
                    'score': round(data_mean, 1),
                    'weight': int(w_data * 100),
                    'interpretation': _score_interp(data_mean),
                },
                'sample_size': {
                    'score': round(sample_mean, 1),
                    'weight': int(w_sample * 100),
                    'interpretation': _score_interp(sample_mean),
                },
                'model_quality': {
                    'score': round(model_mean, 1),
                    'weight': int(w_model * 100),
                    'interpretation': _score_interp(model_mean),
                },
            },
        }
        logger.info(f"Reliability scores: mean={reliability_info['overall_score']:.1f}, "
                    f"range=[{reliability_info['min_score']:.1f}, {reliability_info['max_score']:.1f}], "
                    f"rating={overall_rating}")
        return reliability_info
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        logger.warning(f"Could not calculate reliability score: {e}")
        return None


def generate_comparison_report(orchestrator: Any,
                               level_name: str,
                               period_str: str,
                               bayesian_interpretation: Optional[List[str]],
                               bayesian_cov_interpretation: Optional[List[str]],
                               final_diag: Optional[Dict[str, Any]],
                               final_gdf: Optional[gpd.GeoDataFrame]) -> None:
    """Write the Model_Comparison_<level>_<period>.txt file (when >=2 models present)."""
    logger.info(f"\n--- Model Comparison Analysis ---")
    try:
        interpretations: Dict[str, List[str]] = {}
        if bayesian_interpretation is not None:
            interpretations['Bayesian'] = bayesian_interpretation
        if bayesian_cov_interpretation is not None:
            interpretations['Bayesian Covariates'] = bayesian_cov_interpretation

        reliability_info: Optional[Dict[str, Any]] = None
        if final_diag and final_gdf is not None:
            reliability_info = _build_reliability_info(final_diag, final_gdf, orchestrator.config)

        if len(interpretations) >= 2:
            comparison_report = DiagnosticInterpreter.generate_recommendations_report(
                interpretations.get('Bayesian'),
                interpretations.get('Bayesian Covariates'),
                reliability_info=reliability_info,
            )

            is_hex = level_name.startswith("Hex_Res")
            comparison_file = orchestrator.get_output_path(
                "summary", "Comparison",
                f'Model_Comparison_{level_name}_{period_str}.txt',
                is_hex=is_hex,
            )
            with open(comparison_file, 'w', encoding='utf-8') as f:
                f.write(comparison_report)
            logger.info(f"[OK] Model comparison report saved: {comparison_file}")
    except (IOError, KeyError, AttributeError) as e:
        logger.error(f"Failed to collect comparison data: {e}")
