"""Territory-level and national aggregation helpers."""

from pipeline.aggregation.national_baseline import calculate_national_baseline
from pipeline.aggregation.periods import get_periods
from pipeline.aggregation.geo_utils import ensure_crs_match
from pipeline.aggregation.territory import aggregate_stats
from pipeline.aggregation.testing_network import (
    calculate_testing_intensity,
    classify_network_stability,
)

__all__ = [
    "calculate_national_baseline",
    "get_periods",
    "ensure_crs_match",
    "aggregate_stats",
    "calculate_testing_intensity",
    "classify_network_stability",
]
