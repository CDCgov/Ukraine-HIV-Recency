"""
HIV-case loader (Excel + optional Parquet cache).

The :func:`load_cases_from_disk` routine reads the ``hiv_cases`` sheet
from the analysis Excel file, parses the test-date column, and projects
the rows onto ``target_crs`` as a GeoDataFrame. When ``use_parquet`` is
enabled, the result is cached as a side-by-side ``.parquet`` file and
read from there on subsequent runs, with a source-mtime check that
invalidates the cache if the Excel has been edited.

Returns the loaded GeoDataFrame. The caller (analyzer wrapper) owns
the in-memory cache plus the timestamp-validation logic.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd

from pipeline.exceptions import DataValidationError

logger = logging.getLogger(__name__)


def load_cases_from_disk(excel_path: Union[str, Path], target_crs: str,
                         use_parquet: bool = False) -> gpd.GeoDataFrame:
    """Read ``hiv_cases`` from Excel / Parquet, project to ``target_crs``."""
    parquet_path = Path(excel_path).with_suffix('.parquet')

    should_reload = False
    if use_parquet and parquet_path.exists():
        excel_mtime = os.path.getmtime(excel_path)
        parquet_mtime = os.path.getmtime(parquet_path)
        if excel_mtime > parquet_mtime:
            logger.info(f"Excel file modified after Parquet cache - reloading from source")
            should_reload = True

    if use_parquet and parquet_path.exists() and not should_reload:
        logger.info(f"Loading cases from Parquet: {parquet_path}")
        df_cases = pd.read_parquet(parquet_path)
        df_cases['test_date'] = pd.to_datetime(df_cases['test_date'])
    else:
        logger.info(f"Loading cases from Excel: {excel_path}")
        try:
            df_cases = pd.read_excel(excel_path, sheet_name='hiv_cases')
        except UnicodeDecodeError:
            logger.warning("UTF-8 decoding failed, trying CP1251 (Windows Cyrillic)")
            try:
                df_cases = pd.read_excel(excel_path, sheet_name='hiv_cases', encoding='cp1251')
                logger.info("Successfully loaded with CP1251 encoding")
            except (ValueError, KeyError, IOError) as e:
                logger.error(f"Failed to load Excel with both UTF-8 and CP1251: {e}")
                raise DataValidationError(f"Cannot read Excel file - encoding issue: {e}")
        except (ValueError, KeyError, IOError) as e:
            logger.error(f"Failed to load Excel file: {e}")
            raise DataValidationError(f"Cannot read Excel file: {e}")

        df_cases['test_date'] = pd.to_datetime(df_cases['test_date'])

        if use_parquet:
            logger.info(f"Saving to Parquet for faster future loads: {parquet_path}")
            df_cases.to_parquet(parquet_path, index=False)

    if 'type' in df_cases.columns:
        type_counts = df_cases['type'].value_counts()
        logger.info(f"Test result types found: {dict(type_counts)}")

        expected_types = ['recent', 'long-term', 'negative']
        found_types = set(df_cases['type'].dropna().unique())
        unexpected = found_types - set(expected_types)
        if unexpected:
            logger.warning(f"Unexpected test types found: {unexpected}")
            logger.warning("Expected types: 'recent', 'long-term', 'negative'")
    else:
        logger.error("Column 'type' not found in data!")

    gdf_cases = gpd.GeoDataFrame(
        df_cases,
        geometry=gpd.points_from_xy(df_cases['longitude'], df_cases['latitude']),
        crs='EPSG:4326',
    ).to_crs(target_crs)

    return gdf_cases
