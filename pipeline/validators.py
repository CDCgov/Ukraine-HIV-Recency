"""
Input validation for the HIV hotspot detection pipeline.

These checks run at the system boundary -- immediately after data is loaded --
so that malformed input is rejected with a clear message instead of surfacing
later as an opaque failure inside the sampler (a wrong dtype or a stray NaN in
a count column silently corrupts the posterior).
"""
from typing import List, Optional

import geopandas as gpd
import pandas as pd


def validate_dataframe(df: pd.DataFrame, name: str,
                       required_columns: Optional[List[str]] = None,
                       numeric_columns: Optional[List[str]] = None,
                       non_negative_columns: Optional[List[str]] = None) -> None:
    """
    Validate the structure, types and values of an input DataFrame.

    Args:
        df: DataFrame to validate.
        name: Human-readable name used in error messages.
        required_columns: Columns that must be present and free of NaN.
        numeric_columns: Columns that must hold a numeric dtype.
        non_negative_columns: Columns whose values must all be >= 0
            (counts of tests and infections can never be negative).

    Raises:
        ValueError: If any check fails.
    """
    if df is None:
        raise ValueError(f"{name} cannot be None")
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame, got {type(df)}")
    if len(df) == 0:
        raise ValueError(f"{name} is empty (0 rows)")

    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")

    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"{name} column '{col}' must be numeric, got {df[col].dtype}")

    if non_negative_columns:
        for col in non_negative_columns:
            if col in df.columns and (df[col] < 0).any():
                raise ValueError(f"{name} column '{col}' contains negative values")

    # Required columns must be complete: a NaN here would propagate into the
    # likelihood and quietly corrupt the posterior.
    if required_columns:
        for col in required_columns:
            if col in df.columns and df[col].isna().any():
                raise ValueError(f"{name} column '{col}' contains NaN values")


def validate_geodataframe(gdf, name: str, required_columns: Optional[List[str]] = None) -> None:
    """
    Validate the structure of an input GeoDataFrame.

    Ensures a usable geometry column exists before any spatial operation
    (joins, point-in-polygon aggregation) is attempted.

    Args:
        gdf: GeoDataFrame to validate.
        name: Human-readable name used in error messages.
        required_columns: Non-geometry columns that must be present.

    Raises:
        ValueError: If any check fails.
    """
    if gdf is None:
        raise ValueError(f"{name} cannot be None")
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise ValueError(f"{name} must be a GeoDataFrame, got {type(gdf)}")
    if len(gdf) == 0:
        raise ValueError(f"{name} is empty (0 rows)")
    if 'geometry' not in gdf.columns:
        raise ValueError(f"{name} must have a 'geometry' column")
    if gdf.geometry.isna().all():
        raise ValueError(f"{name} has no valid geometries")

    if required_columns:
        missing = [c for c in required_columns if c not in gdf.columns]
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")
