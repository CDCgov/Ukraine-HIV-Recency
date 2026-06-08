"""
Pick which Bayesian model to fit, fit it, and return the result.

Three branches:

* "Bayesian Covariates only" manual override -- skip the standard
  Bayesian fit entirely; the caller will run the covariates model.
* Hurdle enabled, ``site_present`` available, structural-zeros share
  at or above the threshold -- fit the Truncated Binomial (Hurdle)
  on the active sites.
* Otherwise -- fit the standard hierarchical model with the
  configured parametrisation (centered / non_centered).

Returns ``(gdf_bayes, diag_bayes)``; ``diag_bayes`` is ``None`` when
the standard branch was skipped (covariates-only) or the fit raised.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd

logger = logging.getLogger(__name__)


def run_bayesian_dispatch(bayesian: Any,
                          gdf: gpd.GeoDataFrame,
                          level_name: str,
                          national_rate: float,
                          national_se: float,
                          level_use_hurdle: bool,
                          level_hurdle_threshold: float,
                          force_bayes_cov_only: bool,
                          config: Dict[str, Any]) -> Tuple[gpd.GeoDataFrame, Optional[Dict[str, Any]]]:
    """Dispatch to Hurdle or standard Bayesian; return ``(gdf_bayes, diag_bayes)``."""
    logger.info(f"\n--- Bayesian Analysis ---")

    if force_bayes_cov_only:
        logger.info("⏭Skipping standard Bayesian (Bayesian Covariates only mode)")
        return gdf.copy(), None

    if level_use_hurdle and 'site_present' in gdf.columns:
        n_total = len(gdf)
        n_structural_zeros = (~gdf['site_present']).sum()
        pct_structural = (n_structural_zeros / n_total) * 100

        if pct_structural >= level_hurdle_threshold:
            logger.info(f"Using Truncated Binomial (active sites): {pct_structural:.1f}% structural zeros (threshold: {level_hurdle_threshold}%)")
            return bayesian.run_hurdle_model(gdf, level_name, national_rate)

        logger.info(f"Using standard Bayesian model: {pct_structural:.1f}% structural zeros (below threshold)")
    else:
        if not level_use_hurdle:
            logger.info(f"Using standard Bayesian model (Truncated Binomial disabled)")
        elif 'site_present' not in gdf.columns:
            logger.warning(f"site_present column missing - cannot use Hurdle model, using standard Bayesian")

    parametrization = config.get('bayesian_parametrization', 'centered')
    return bayesian.run_model(
        gdf, level_name, national_rate, national_se,
        parametrization=parametrization,
    )
