"""Territory-level and national aggregation helpers."""

from pipeline.aggregation.national_baseline import calculate_national_baseline
from pipeline.aggregation.periods import get_periods
from pipeline.aggregation.geo_utils import ensure_crs_match
from pipeline.aggregation.outbreak_defaults import soft_fallback_result
from pipeline.aggregation.territory import (
    aggregate_stats,
    aggregate_covariates,
    aggregate_stats_stratified,
    aggregate_stats_hard_stratified,
    detect_outbreak_and_artifact,
)
from pipeline.aggregation.testing_network import (
    analyze_site_profile,
    calculate_testing_intensity,
    classify_network_stability,
    analyze_network_change,
    generate_network_explanation,
)

__all__ = [
    "calculate_national_baseline",
    "get_periods",
    "ensure_crs_match",
    "soft_fallback_result",
    "aggregate_stats",
    "aggregate_covariates",
    "aggregate_stats_stratified",
    "aggregate_stats_hard_stratified",
    "detect_outbreak_and_artifact",
    "analyze_site_profile",
    "calculate_testing_intensity",
    "classify_network_stability",
    "analyze_network_change",
    "generate_network_explanation",
]
