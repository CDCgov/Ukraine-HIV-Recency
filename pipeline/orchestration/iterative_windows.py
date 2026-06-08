"""
Build the sliding analysis windows used by iterative-mode runs.

The iterative analyser walks analysis periods backward in 1-month steps,
each with its own preceding, non-overlapping baseline. The analysis-window
length (``iterative_analysis_months``, 3 or 6) and the baseline length
(``iterative_baseline_months``, default 12) come from the config. This
helper takes the case Excel, applies the optional ``iterative_date_range``
filter, and emits the windows in reverse-chronological order. The loop in
:meth:`PipelineOrchestrator.run_iterative_analysis` only has to iterate
over the result.

The "last complete month" anchor matters: a max_date of 2026-03-20
generates an analysis window ending 2026-03-31, not on the 20th.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h3
import pandas as pd
from dateutil.relativedelta import relativedelta

from pipeline.aggregation.periods import BASELINE_DATA_FLOOR, baseline_months_for

logger = logging.getLogger(__name__)


def load_and_filter_cases(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Load cases, attach H3 res-4 IDs, apply ``iterative_date_range`` filter."""
    excel_path = Path(config['excel_path'])
    logger.info(f"Loading cases from: {excel_path}")

    df_cases = pd.read_excel(excel_path, sheet_name='hiv_cases')
    df_cases['test_date'] = pd.to_datetime(df_cases['test_date'])

    # Generate h3_id for each case based on coordinates (resolution 4 for iterative mode)
    if 'latitude' in df_cases.columns and 'longitude' in df_cases.columns:
        logger.info("Generating H3 IDs (resolution 4) for cases...")
        df_cases['h3_id'] = df_cases.apply(
            lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], 4)
            if pd.notna(row['latitude']) and pd.notna(row['longitude']) else None,
            axis=1
        )
        h3_count = df_cases['h3_id'].notna().sum()
        logger.info(f"Generated H3 IDs for {h3_count}/{len(df_cases)} cases")
    else:
        logger.warning("latitude/longitude columns not found - case dates will not be populated")

    # Filter by user-selected date range if specified
    if 'iterative_date_range' in config:
        filter_start = pd.to_datetime(config['iterative_date_range']['start'])
        filter_end = pd.to_datetime(config['iterative_date_range']['end'])

        original_count = len(df_cases)
        df_cases = df_cases[(df_cases['test_date'] >= filter_start) & (df_cases['test_date'] <= filter_end)]
        filtered_count = len(df_cases)

        logger.info(f"Filtered dataset by date range: {filter_start.date()} to {filter_end.date()}")
        logger.info(f"Cases: {original_count} → {filtered_count} ({filtered_count/original_count*100:.1f}%)")

        max_date = filter_end
        min_date = filter_start
    else:
        max_date = df_cases['test_date'].max()
        min_date = df_cases['test_date'].min()

    logger.info(f"Analysis date range: {min_date.date()} to {max_date.date()}")
    return df_cases, min_date, max_date


def build_iteration_windows(min_date: pd.Timestamp, max_date: pd.Timestamp,
                            analysis_months: int = 3,
                            baseline_months: int = None) -> List[Dict[str, Any]]:
    """Return sliding analysis windows, each with a preceding baseline.

    ``analysis_months`` is the analysis-window length; ``baseline_months`` is
    derived from it (:func:`baseline_months_for`) unless given. Windows step
    back 1 month at a time. Two constraints:

    * the analysis window stays inside the requested range (``analysis_start
      >= min_date``);
    * the non-overlapping baseline never starts before
      :data:`BASELINE_DATA_FLOOR` -- the baseline may reach earlier than the
      analysis range (it draws on the full history), but not before the floor.

    Once stepping back would push the baseline below the floor, no earlier
    window can qualify, so generation stops.
    """
    if baseline_months is None:
        baseline_months = baseline_months_for(analysis_months)

    windows: List[Dict[str, Any]] = []

    analysis_end = (max_date.replace(day=1) + relativedelta(months=1)) - timedelta(days=1)
    analysis_start = (analysis_end.replace(day=1) - relativedelta(months=analysis_months - 1))

    iteration = 1
    while analysis_start >= min_date:
        # Baseline: exactly ``baseline_months`` immediately before the analysis.
        baseline_end = analysis_start - timedelta(days=1)
        baseline_start = analysis_start - relativedelta(months=baseline_months)

        if baseline_start < BASELINE_DATA_FLOOR:
            break  # earlier windows would reach even further below the floor

        windows.append({
            'iteration': iteration,
            'analysis_start': analysis_start,
            'analysis_end': analysis_end,
            'baseline_start': baseline_start,
            'baseline_end': baseline_end
        })

        # Step back exactly 1 month.
        analysis_end = (analysis_end.replace(day=1) - relativedelta(months=1))
        analysis_end = (analysis_end.replace(day=1) + relativedelta(months=1)) - timedelta(days=1)
        analysis_start = (analysis_end.replace(day=1) - relativedelta(months=analysis_months - 1))
        iteration += 1

    return windows


def prepare_iterative_windows(config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Load+filter cases and build iteration windows.

    Raises ``ValueError`` (asking the user to choose a different period) when
    the chosen analysis-window length and the 2023 baseline floor leave fewer
    than two iterations inside the requested range.
    """
    df_cases, min_date, max_date = load_and_filter_cases(config)
    analysis_months = int(config.get('iterative_analysis_months', 3))
    baseline_months = baseline_months_for(analysis_months)
    logger.info(f"Iterative windows: {analysis_months}-month analysis, "
                f"{baseline_months}-month baseline (floor {BASELINE_DATA_FLOOR.date()})")

    windows = build_iteration_windows(min_date, max_date, analysis_months=analysis_months)

    if len(windows) < 2:
        earliest_start = BASELINE_DATA_FLOOR + relativedelta(months=baseline_months)
        # End needed so that >=2 monthly-stepped windows fit above the floor.
        need_end = earliest_start + relativedelta(months=analysis_months)
        raise ValueError(
            f"A {analysis_months}-month analysis window needs a {baseline_months}-month "
            f"baseline, which cannot start before {BASELINE_DATA_FLOOR.date()}. In the "
            f"requested range {min_date.date()}..{max_date.date()} that yields only "
            f"{len(windows)} iteration(s); at least 2 are required. The earliest the "
            f"analysis period can start is {earliest_start.date()}, so choose an analysis "
            f"range ending on/after ~{need_end.date()} (or a shorter analysis window)."
        )

    logger.info(f"Generated {len(windows)} sliding windows "
                f"(analysis {windows[-1]['analysis_start'].date()}..{windows[0]['analysis_end'].date()})")
    return df_cases, windows
