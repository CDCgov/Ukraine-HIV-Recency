"""
Analysis / baseline period setup.

The pipeline runs over a current "analysis" window and a preceding,
non-overlapping "baseline" window. The analysis window comes from
``cfg['analysis_period']``; the baseline length is **derived from the
analysis-window length** (see :func:`baseline_months_for`) rather than fixed
at 12 months, so a longer analysis window gets a longer, more informative
reference. Baselines never reach earlier than :data:`BASELINE_DATA_FLOOR`.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict, Tuple

import pandas as pd
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)


# Hard floor on baseline data. The testing programme before 2023 had a very
# different site network and risk-group case-mix, so baselines never reach
# earlier than this date; a run whose baseline would start before it is
# rejected with a request to choose a later analysis period (audit).
BASELINE_DATA_FLOOR = pd.Timestamp('2023-01-01')


def baseline_months_for(analysis_months: int) -> int:
    """Baseline length (months) for an analysis window of ``analysis_months``.

    1-6 months -> 12, 7-9 -> 18, 10-12 -> 24. A longer analysis window gets a
    proportionally longer baseline so the historical reference stays at least
    as informative as the analysis window itself, capped at 24 months.
    """
    if analysis_months <= 6:
        return 12
    if analysis_months <= 9:
        return 18
    return 24


def months_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Inclusive whole-month length of the [start, end] period."""
    return (end.year - start.year) * 12 + (end.month - start.month) + 1


def get_periods(cfg: Dict[str, Any], extend_period: bool = False
                ) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return ``(start, end, b_start, b_end)`` for the analysis run.

    The baseline length is derived from the analysis-window length via
    :func:`baseline_months_for`. Raises ``ValueError`` if the baseline would
    start before :data:`BASELINE_DATA_FLOOR` (the caller / wizard should have
    already asked the user to pick a later analysis period).

    ``extend_period=True`` keeps the legacy doubled-window behaviour for the
    sparse-data retry path and is not subject to the derived-baseline rule.
    """
    start = pd.to_datetime(cfg['analysis_period']['start'])
    end = pd.to_datetime(cfg['analysis_period']['end'])

    if extend_period:
        original_duration = (end - start).days
        start = end - pd.Timedelta(days=original_duration * 2)
        logger.info(f"[WARN] EXTENDING PERIOD: {original_duration} → {original_duration * 2} days")

        b_end = start - pd.Timedelta(days=1)
        b_start = b_end - pd.Timedelta(days=original_duration * 2) + pd.Timedelta(days=1)
    else:
        analysis_months = months_between(start, end)
        baseline_months = baseline_months_for(analysis_months)
        b_end = start - timedelta(days=1)
        b_start = start - relativedelta(months=baseline_months)

        if b_start < BASELINE_DATA_FLOOR:
            earliest = BASELINE_DATA_FLOOR + relativedelta(months=baseline_months)
            raise ValueError(
                f"Baseline would start {b_start.date()}, before the "
                f"{BASELINE_DATA_FLOOR.date()} floor: a {analysis_months}-month analysis "
                f"window needs a {baseline_months}-month baseline. Choose an analysis "
                f"period starting on or after {earliest.date()}."
            )
        logger.info(f"Baseline length: {baseline_months} months "
                    f"(for a {analysis_months}-month analysis window)")

    logger.info(f"Analysis: {start.date()} to {end.date()}")
    logger.info(f"Baseline: {b_start.date()} to {b_end.date()}")

    return start, end, b_start, b_end
