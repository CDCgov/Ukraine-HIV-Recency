"""
Render Decision Audit Trail reports for each processed level.

Each level accumulates a ``DecisionAuditTrail`` during the per-level
loop in ``run_for_mode``. At the end of the mode this routine walks
the orchestrator's ``audit_trails`` dict, closes any still-open
stage, stamps the per-level final summary (model used, hotspot
count) onto the metadata, and emits Markdown / HTML / JSON
companions to the same output directory the rest of the level's
artefacts went into.

Best-effort per level: a failure in one level's report generation
logs the traceback but does not abort the others. The orchestrator
is passed in (rather than just dicts) because ``get_output_path``
encapsulates the per-level/hex directory policy.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

logger = logging.getLogger(__name__)


def generate_audit_trail_reports(orchestrator: Any, period_str: str) -> None:
    """Generate Markdown/HTML/JSON audit-trail reports for every level on the orchestrator."""
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING DECISION AUDIT TRAIL REPORTS")
    logger.info("=" * 80)

    for level_name_key, audit_trail in orchestrator.audit_trails.items():
        try:
            # End any open stage
            if audit_trail.current_stage:
                audit_trail.end_stage()

            # Set final metadata
            if level_name_key in orchestrator.results:
                result_data = orchestrator.results[level_name_key]
                audit_trail.set_metadata(
                    model_used=result_data.get('model_used'),
                    final_hotspots=int(result_data['gdf']['is_hotspot'].sum()) if 'gdf' in result_data and 'is_hotspot' in result_data['gdf'].columns else None
                )

            # Generate reports
            is_hex = level_name_key.startswith("Hex_Res")

            markdown_path = orchestrator.get_output_path(
                "summary", level_name_key,
                f"DecisionAuditTrail_{level_name_key}_{period_str}.md",
                is_hex=is_hex
            )
            html_path = orchestrator.get_output_path(
                "summary", level_name_key,
                f"DecisionAuditTrail_{level_name_key}_{period_str}.html",
                is_hex=is_hex
            )
            json_path = orchestrator.get_output_path(
                "summary", level_name_key,
                f"DecisionAuditTrail_{level_name_key}_{period_str}.json",
                is_hex=is_hex
            )

            audit_trail.generate_markdown(str(markdown_path))
            audit_trail.generate_html(str(html_path))
            audit_trail.to_json(str(json_path))

            logger.info(f"[OK] Decision Audit Trail generated for {level_name_key}")
            logger.info(f"   Markdown: {markdown_path}")
            logger.info(f"   HTML: {html_path}")
            logger.info(f"   JSON: {json_path}")

        except Exception as e:
            logger.error(f"Failed to generate audit trail for {level_name_key}: {e}")
            logger.error(traceback.format_exc())

    logger.info("=" * 80)
