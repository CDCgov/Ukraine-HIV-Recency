"""
Geographic helpers shared across aggregation routines.

:func:`ensure_crs_match` is the one piece needed often enough that it
deserved its own helper: spatial joins silently produce wrong results
when the two frames disagree on CRS (or one of them has no CRS set), so
this routine harmonises them before the join and logs each correction.
"""

from __future__ import annotations

import logging
from typing import Tuple

import geopandas as gpd

logger = logging.getLogger(__name__)


def ensure_crs_match(gdf_left: gpd.GeoDataFrame, gdf_right: gpd.GeoDataFrame,
                     operation: str = "spatial join"
                     ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Harmonise CRS between two frames.

    Missing CRS is filled with EPSG:4326; mismatched CRS reprojects the
    right frame into the left frame's CRS. Each correction is logged at
    WARNING level so the caller can spot configuration drift.
    """
    if gdf_left.crs is None:
        logger.warning(f"Left GeoDataFrame has no CRS in {operation} — setting to EPSG:4326")
        gdf_left = gdf_left.set_crs('EPSG:4326', allow_override=True)
    if gdf_right.crs is None:
        logger.warning(f"Right GeoDataFrame has no CRS in {operation} — setting to EPSG:4326")
        gdf_right = gdf_right.set_crs('EPSG:4326', allow_override=True)
    if gdf_left.crs != gdf_right.crs:
        logger.warning(f"CRS mismatch in {operation}: left={gdf_left.crs}, right={gdf_right.crs}")
        logger.warning(f"  → Reprojecting right to match left ({gdf_left.crs})")
        gdf_right = gdf_right.to_crs(gdf_left.crs)
    return gdf_left, gdf_right
