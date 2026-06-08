"""
Testing-sites loader.

Reads the ``testing_sites`` sheet from the analysis Excel file and
returns a DataFrame with each site's coordinates and its activation /
deactivation dates. The deactivation column is sparse: most sites are
permanently active, but ones that closed (war-related, relocation,
re-organisation) carry a date that is later compared to the analysis
window to decide whether the site contributed during that period.

Returns ``None`` (with a WARN) when the sheet is missing or unreadable,
so downstream code can fall back to "no testing-effort information" mode
rather than crashing.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_testing_sites(excel_path: str) -> Optional[pd.DataFrame]:
    """Load the ``testing_sites`` sheet; return ``None`` on missing / unreadable."""
    try:
        df_sites = pd.read_excel(excel_path, sheet_name='testing_sites')

        df_sites['activation_date'] = pd.to_datetime(df_sites['activation_date'])
        df_sites['deactivation_date'] = pd.to_datetime(df_sites['deactivation_date'])

        logger.info(f"[OK] Loaded {len(df_sites)} testing sites")

        permanent_active = df_sites['deactivation_date'].isna().sum()
        has_deactivation = df_sites['deactivation_date'].notna().sum()
        logger.info(f"  Sites without deactivation date: {permanent_active}, "
                    f"Sites with deactivation date: {has_deactivation}")
        if has_deactivation > 0:
            logger.info(f"  Note: Actual active sites depend on analysis period (filtered by activation/deactivation dates)")

        return df_sites

    except ValueError as e:
        if 'testing_sites' in str(e):
            logger.warning("[WARN] Sheet 'testing_sites' not found in Excel file")
            logger.warning("   Testing effort analysis will be disabled")
            return None
        else:
            raise
    except (IOError, FileNotFoundError, KeyError) as e:
        logger.error(f"Failed to load testing_sites: {e}")
        logger.warning("Continuing without testing effort analysis")
        return None
