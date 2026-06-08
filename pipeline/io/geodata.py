"""
Geo-data loader for administrative units and H3 hexagons.

Reads the GeoJSON for the requested level (admin string ``Community`` /
``District`` / ``Oblast`` or H3 resolution integer), projects it onto
``target_crs`` and optionally writes a side-by-side ``.parquet`` cache
for subsequent runs. Sets two convenience columns on the result:
``level_name`` (a string label) and, for admin units, ``name_col``
(the territory ID column name -- looked up from
``cfg['administrative_units']``).

The caller (analyzer wrapper) owns the in-memory dict cache plus the
file-mtime invalidation check; this routine is pure read-and-project.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Union

import geopandas as gpd

logger = logging.getLogger(__name__)


def load_geodata_from_disk(cfg: Dict[str, Any], path: Union[str, Path],
                           level: Union[str, int],
                           use_parquet: bool = False) -> gpd.GeoDataFrame:
    """Read a level's GeoJSON / Parquet, project, attach level metadata."""
    parquet_path = Path(path).with_suffix('.parquet')

    should_reload = False
    if use_parquet and parquet_path.exists():
        source_mtime = os.path.getmtime(path)
        parquet_mtime = os.path.getmtime(parquet_path)
        if source_mtime > parquet_mtime:
            logger.info(f"GeoJSON file modified after Parquet cache - reloading from source")
            should_reload = True

    if use_parquet and parquet_path.exists() and not should_reload:
        logger.info(f"Loading geodata from Parquet: {parquet_path}")
        gdf = gpd.read_parquet(parquet_path)
    else:
        logger.info(f"Loading geodata from GeoJSON: {path}")
        gdf = gpd.read_file(path).to_crs(cfg['target_crs'])
        if use_parquet:
            logger.info(f"Saving geodata to Parquet: {parquet_path}")
            gdf.to_parquet(parquet_path)

    if isinstance(level, int):
        gdf['level_name'] = f'Hex_Res{level}'
    else:
        gdf['level_name'] = level
        admin_paths = cfg['administrative_units']
        if level == 'Community':
            gdf['name_col'] = admin_paths['otg_col']
        elif level == 'District':
            gdf['name_col'] = admin_paths['rayon_col']
        else:
            gdf['name_col'] = admin_paths['oblast_col']

    return gdf
