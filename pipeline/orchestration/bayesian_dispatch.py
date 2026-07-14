"""
Fit the Bayesian model and return the result.

Two branches:

* Joint two-period model (``two_period_model`` config) -- the production
  detector; reads the level from the current-period rate and the trend from a
  hierarchical per-territory change.
* Otherwise -- the standard single-window hierarchical model with the
  configured parametrisation (centered / non_centered).

Returns ``(gdf_bayes, diag_bayes)``; ``diag_bayes`` is ``None`` when the fit
raised.
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
                          config: Dict[str, Any]) -> Tuple[gpd.GeoDataFrame, Optional[Dict[str, Any]]]:
    """Dispatch to the two-period or standard Bayesian model; return ``(gdf_bayes, diag_bayes)``."""
    logger.info(f"\n--- Bayesian Analysis ---")

    # Joint two-period model -- the production detector. Reads the trend from a
    # hierarchical per-territory current-period change delta_i and the level from
    # the current-period rate. When enabled it supersedes the single-window model.
    if config.get('two_period_model', False):
        parametrization = config.get('bayesian_parametrization', 'non_centered')
        logger.info("Using Joint Two-Period Beta-Binomial model")
        return bayesian.run_two_period_model(
            gdf, level_name, national_rate, national_se,
            parametrization=parametrization,
        )

    logger.info("Using standard Bayesian hierarchical model")
    parametrization = config.get('bayesian_parametrization', 'centered')
    return bayesian.run_model(
        gdf, level_name, national_rate, national_se,
        parametrization=parametrization,
    )
