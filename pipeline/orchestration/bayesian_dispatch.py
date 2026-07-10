"""
Pick which Bayesian model to fit, fit it, and return the result.

Three branches:

* "Bayesian Covariates only" manual override -- skip the standard
  Bayesian fit entirely; the caller will run the covariates model.
* Two-part model enabled (or a retired Hurdle request) -- fit the
  two-part zero-inflated Beta-Binomial model, whose presence sub-model
  handles zero-count territories directly.
* Otherwise -- fit the standard hierarchical model with the
  configured parametrisation (centered / non_centered).

The legacy Truncated Binomial ("Hurdle") model has been retired in
favour of the two-part model; a ``level_use_hurdle`` request is routed
to the two-part model.

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

    # Two-part (zero-inflated Beta-Binomial) model -- Fu 2023 style. Its presence
    # sub-model handles zero-count territories directly (a territory with many
    # tests and zero recent events is treated as structurally absent, not shrunk
    # to the national rate), which is exactly the failure mode the single-part
    # SIR ratio inflated into false 'Emerging hotspot's.
    #
    # The legacy Truncated Binomial ("Hurdle") branch is retired: it fit a
    # weakly-identified per-territory random slope (one Binomial observation per
    # territory) under a less robust sampling configuration. The two-part model
    # is its replacement, so a lingering Hurdle request is honoured by running
    # the two-part model instead.
    if config.get('two_part_model', False) or level_use_hurdle:
        if level_use_hurdle and not config.get('two_part_model', False):
            logger.warning("The Truncated Binomial (Hurdle) model is retired; "
                           "running the two-part zero-inflated Beta-Binomial model instead.")
        parametrization = config.get('bayesian_parametrization', 'non_centered')
        logger.info("Using Two-Part zero-inflated Beta-Binomial model (Fu 2023 style)")
        return bayesian.run_two_part_model(
            gdf, level_name, national_rate, national_se,
            parametrization=parametrization,
        )

    logger.info("Using standard Bayesian hierarchical model")
    parametrization = config.get('bayesian_parametrization', 'centered')
    return bayesian.run_model(
        gdf, level_name, national_rate, national_se,
        parametrization=parametrization,
    )
