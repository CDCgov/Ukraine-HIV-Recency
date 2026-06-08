"""
National baseline rate from cases filtered to the baseline window.

The baseline rate is the country-wide recency proportion in the period
*before* the analysis window. It is used as the reference point against
which per-territory SMR / SIR ratios are calculated. Anchoring on the
*baseline* period rather than the current one is essential -- otherwise
an outbreak in a populous city
inflates the rate it is later compared against and masks itself.

Two corrections live here:

* **FRR (false-recent rate) correction** -- when ``cfg['bayesian']['frr']``
  is set, the expected count of false-recent results (``total_tests * frr``)
  is subtracted from the observed recent count before the rate is computed.
* **ICC / design-effect correction on the standard error** -- pooled SRS
  underestimates uncertainty when tests cluster within facilities. The
  routine computes a one-way random-effects ICC across ``site_id`` groups
  (≥ 3 facilities required) and inflates the SE by ``sqrt(1 + (n_avg-1) *
  ICC)``. Without ``site_id`` or with fewer facilities, the SRS SE is used.

Returns the ``(rate, se)`` tuple. The caller is responsible for storing
them on the analyzer instance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_national_baseline(cfg: Dict[str, Any], gdf_cases: gpd.GeoDataFrame,
                                b_start: pd.Timestamp, b_end: pd.Timestamp) -> Tuple[float, float]:
    """Return ``(rate, se)`` for the baseline window with ICC-corrected SE."""
    cases_baseline = gdf_cases[(gdf_cases['test_date'] >= b_start) & (gdf_cases['test_date'] <= b_end)]

    total_tests = len(cases_baseline)
    total_recent = (cases_baseline['type'] == 'recent').sum()

    frr = cfg.get('bayesian', {}).get('frr')
    if frr is not None and frr > 0:
        false_recent = int(round(total_tests * frr))
        total_recent = max(0, total_recent - false_recent)
        logger.info(f"FRR correction on national baseline: removed {false_recent} false recent")

    if total_tests > 0:
        rate = total_recent / total_tests

        se_srs = np.sqrt(rate * (1 - rate) / total_tests)
        icc = None
        deff = 1.0

        if 'site_id' in cases_baseline.columns:
            site_groups = cases_baseline.groupby('site_id')
            k = len(site_groups)

            if k >= 3:
                site_stats = site_groups.agg(
                    n_tests=('type', 'size'),
                    n_recent=('type', lambda x: (x == 'recent').sum()),
                )
                site_stats['rate'] = site_stats['n_recent'] / site_stats['n_tests']

                n_per_site = site_stats['n_tests'].values
                rates = site_stats['rate'].values
                n_total = n_per_site.sum()

                mean_rate = rate

                msb = np.sum(n_per_site * (rates - mean_rate) ** 2) / (k - 1)
                msw = mean_rate * (1 - mean_rate)

                n_avg = n_per_site.mean()
                icc = (msb - msw) / (msb + (n_avg - 1) * msw) if (msb + (n_avg - 1) * msw) > 0 else 0
                icc = np.clip(icc, 0, 1)

                deff = 1 + (n_avg - 1) * icc

                se = se_srs * np.sqrt(deff)

                logger.info(f"ICC correction:")
                logger.info(f"  Facilities (k): {k}")
                logger.info(f"  Avg tests/facility: {n_avg:.1f}")
                logger.info(f"  ICC: {icc:.4f}")
                logger.info(f"  DEFF: {deff:.2f}")
                logger.info(f"  SE (SRS): {se_srs:.4f} → SE (corrected): {se:.4f}")
            else:
                se = se_srs
                logger.info(f"Only {k} facilities — too few for ICC, using SRS SE")
        else:
            se = se_srs
            logger.info(f"No site_id column — using SRS SE (no design effect correction)")
    else:
        rate = 0
        se = 0.01

    logger.info(f"National baseline: {rate:.4f} (SE: {se:.4f}) from BASELINE period {b_start.date()} to {b_end.date()}")
    logger.info(f"Based on {total_tests} tests, {total_recent} recent infections")

    return rate, se
