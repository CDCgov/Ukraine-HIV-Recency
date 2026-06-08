"""
Aggregate covariates, build stratified frames, and run the
Bayesian Covariates fit.

Mirrors :func:`run_bayesian_dispatch` but for the covariates
analyser. The function:

* Skips the fit entirely in "standard Bayesian only" mode (the
  inverse of ``force_bayes_cov_only`` in the standard dispatch).
* Aggregates per-territory covariates and the SOFT / HARD
  stratification frames, storing them on the covariates analyser
  so downstream interpret / plot stages can read them back.
* Runs ``bayesian_cov.run_model`` with the configured
  parametrisation; appends the diagnostics dict to the analyser's
  history.

Returns ``(gdf_bayes_cov, diag_bayes_cov)`` -- ``diag_bayes_cov``
is ``None`` when the fit was skipped or the stratified frame came
back empty.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd

logger = logging.getLogger(__name__)


def run_bayesian_covariates_dispatch(bayesian: Any,
                                     bayesian_cov: Any,
                                     gdf: gpd.GeoDataFrame,
                                     gdf_cases: gpd.GeoDataFrame,
                                     level: str,
                                     level_name: str,
                                     start: Any,
                                     end: Any,
                                     b_start: Any,
                                     b_end: Any,
                                     national_rate: float,
                                     national_se: float,
                                     force_bayesian_only: bool,
                                     config: Dict[str, Any]) -> Tuple[gpd.GeoDataFrame, Optional[Dict[str, Any]]]:
    """Run the Bayesian Covariates fit; return ``(gdf_bayes_cov, diag_bayes_cov)``."""
    if force_bayesian_only:
        logger.info("⏭Skipping Bayesian Covariates (Bayesian only mode)")
        return gdf.copy(), None

    logger.info(f"\n--- Bayesian with Covariates Analysis (STRATIFIED) ---")

    gdf_cov = bayesian.load_geodata(level)
    gdf_cov = bayesian.aggregate_stats(gdf_cov, gdf_cases, start, end, b_start, b_end)
    gdf_cov = bayesian_cov.aggregate_covariates(gdf_cov, gdf_cases, start, end)

    df_stratified = bayesian_cov.aggregate_stats_stratified(gdf_cov, gdf_cases, start, end, b_start, b_end)
    bayesian_cov.df_stratified = df_stratified

    df_hard_stratified = bayesian_cov.aggregate_stats_hard_stratified(gdf_cov, gdf_cases, start, end, b_start, b_end)
    bayesian_cov.df_hard_stratified = df_hard_stratified

    if len(df_stratified) == 0:
        logger.error("No stratified data available - skipping Bayesian Covariates")
        return gdf_cov, None

    parametrization = config.get('bayesian_parametrization', 'centered')
    gdf_bayes_cov, diag_bayes_cov = bayesian_cov.run_model(
        gdf_cov, level_name, national_rate, national_se,
        parametrization=parametrization,
    )

    if diag_bayes_cov:
        bayesian_cov.diagnostics.append(diag_bayes_cov)

    return gdf_bayes_cov, diag_bayes_cov
