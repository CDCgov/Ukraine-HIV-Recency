"""
Standard-mode post-pipeline steps: dashboard + historical comparison.

After ``run_for_mode`` and the model-comparison / recommendations
writers, the pipeline renders a single-page summary dashboard and
runs a side-by-side comparison against the previous period's
``Results_<period>.json``. Both steps are best-effort: they catch
exceptions so a dashboard or comparison failure does not lose the
analysis output that has already been written.

These two routines take the orchestrator so they can reuse its
``get_output_path`` and the already-defined ``_collect_*`` helpers.
This keeps the directory layout policy (sessioned timestamps,
``summary`` subfolder) where it lives, on the orchestrator.
"""

from __future__ import annotations

import json
import logging
import traceback
from typing import Any

logger = logging.getLogger(__name__)


def create_summary_dashboard(orchestrator: Any, dashboard_cls: Any) -> None:
    """Render the summary dashboard PNG. Best-effort: logs and swallows IO/data errors."""
    try:
        logger.info("\n" + "=" * 60)
        logger.info("CREATING SUMMARY DASHBOARD")
        logger.info("=" * 60)

        dashboard_results = orchestrator._collect_dashboard_data()

        if dashboard_results:
            dashboard_file = orchestrator.get_output_path(
                "summary", "Dashboard",
                f'Dashboard_{orchestrator.period_str}.png',
                is_hex=False,
            )
            dashboard_cls.create_dashboard(dashboard_results, dashboard_file)
            logger.info(f"[OK] Dashboard created: {dashboard_file}")
        else:
            logger.warning("[WARN] No results available for dashboard")

    except (ValueError, KeyError, AttributeError, IOError) as e:
        logger.error(f"Failed to create dashboard: {e}")
        logger.error(traceback.format_exc())


def run_historical_comparison(orchestrator: Any,
                              comparison_cls: Any,
                              pipeline_version: str) -> None:
    """Compare current results vs prior ``Results_<period>.json``; write Comparison + Results JSON."""
    try:
        logger.info("\n" + "=" * 60)
        logger.info("HISTORICAL COMPARISON")
        logger.info("=" * 60)

        previous_results = comparison_cls.load_previous_results(
            output_dir=orchestrator.output_dir,
            period=orchestrator.period_str,
        )

        if previous_results:
            current_results = orchestrator._collect_current_results()
            comparison = comparison_cls.compare_periods(
                current=current_results,
                previous=previous_results,
                pipeline_version=pipeline_version,
            )
            comparison_cls.print_comparison(comparison)
            comp_file = orchestrator.get_output_path(
                "summary", "Comparison",
                f'Comparison_{orchestrator.period_str}.json',
                is_hex=False,
            )
            with open(comp_file, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
            logger.info(f"[OK] Historical comparison saved: {comp_file}")
        else:
            logger.info("No previous results found for comparison")

        current_results = orchestrator._collect_current_results()
        results_file = orchestrator.get_output_path(
            "summary", "Results",
            f'Results_{orchestrator.period_str}.json',
            is_hex=False,
        )
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(current_results, f, indent=2, ensure_ascii=False)
        logger.info(f"[OK] Current results saved for future comparison: {results_file}")

    except (IOError, KeyError, ValueError) as e:
        logger.error(f"Failed to perform historical comparison: {e}")
