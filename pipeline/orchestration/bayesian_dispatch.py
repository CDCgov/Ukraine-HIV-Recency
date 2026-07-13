"""
Pick which Bayesian model to fit, fit it, and return the result.

Three branches:

* "Bayesian Covariates only" manual override -- skip the standard
  Bayesian fit entirely; the caller will run the covariates model.
* Joint two-period model (``two_period_model`` config) -- the production
  detector; reads the trend from a hierarchical per-territory change.
* Otherwise -- fit the standard hierarchical model with the
  configured parametrisation (centered / non_centered).

The two-part zero-inflated and Truncated-Binomial ("Hurdle") branches have
been retired; zero-count units are handled by the denominator filter plus
the two-recent-event presence gate.

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
                          force_bayes_cov_only: bool,
                          config: Dict[str, Any]) -> Tuple[gpd.GeoDataFrame, Optional[Dict[str, Any]]]:
    """Dispatch to the two-period or standard Bayesian model; return ``(gdf_bayes, diag_bayes)``."""
    logger.info(f"\n--- Bayesian Analysis ---")

    if force_bayes_cov_only:
        logger.info("⏭Skipping standard Bayesian (Bayesian Covariates only mode)")
        return gdf.copy(), None

    # Joint two-period model -- the production detector. Reads the trend directly
    # from a hierarchical per-territory current-period change delta_i and the
    # level from the current-period rate. When enabled it supersedes the standard
    # single-window model. (The two-part zero-inflated and Truncated-Binomial
    # "Hurdle" branches are retired; zero-count units are handled by the
    # denominator filter plus the two-recent-event presence gate.)
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
