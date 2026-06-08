"""
Compare current pipeline results with a previously persisted snapshot.

The :class:`HistoricalComparison` is intentionally tiny: it loads a JSON
results file written by an earlier run, diffs the hotspot set (new /
resolved / persistent territories) and emits a structured comparison
dict plus a short log report. Nothing here owns analytical logic --
all classification has already happened by the time results land here.

The pipeline version is taken as an argument rather than imported, so
this module stays self-contained.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class HistoricalComparison:
    """Snapshot comparison between two pipeline runs."""

    @staticmethod
    def load_previous_results(output_dir: Path, period: str) -> Optional[Dict[str, Any]]:
        """Load ``results_<period>.json`` from a previous run, or ``None``."""
        try:
            results_file = output_dir / f'results_{period}.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    return json.load(f)
            return None
        except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load previous results: {e}")
            return None

    @staticmethod
    def compare_periods(current: Dict[str, Any], previous: Dict[str, Any],
                        pipeline_version: str = "unknown") -> Dict[str, Any]:
        """Diff ``current`` against ``previous``.

        Returns a dict with the hotspot count delta, the partition of
        territories into ``new`` / ``resolved`` / ``persistent`` sets, and
        a short list of human-readable change strings.
        """
        comparison: Dict[str, Any] = {
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': pipeline_version,
            'changes': [],
        }

        current_hotspots = current.get('n_hotspots', 0)
        previous_hotspots = previous.get('n_hotspots', 0)
        change = current_hotspots - previous_hotspots
        pct_change = (change / previous_hotspots * 100) if previous_hotspots > 0 else 0

        comparison['hotspot_change'] = {
            'current': current_hotspots,
            'previous': previous_hotspots,
            'absolute_change': change,
            'percent_change': pct_change,
        }

        if change > 0:
            comparison['changes'].append(f"[WARN]  Hotspots INCREASED by {change} ({pct_change:+.1f}%)")
        elif change < 0:
            comparison['changes'].append(f"[OK] Hotspots DECREASED by {abs(change)} ({pct_change:+.1f}%)")
        else:
            comparison['changes'].append("→ Number of hotspots UNCHANGED")

        current_territories = set(current.get('hotspot_territories', []))
        previous_territories = set(previous.get('hotspot_territories', []))

        new_hotspots = current_territories - previous_territories
        resolved_hotspots = previous_territories - current_territories
        persistent_hotspots = current_territories & previous_territories

        comparison['territory_changes'] = {
            'new': list(new_hotspots),
            'resolved': list(resolved_hotspots),
            'persistent': list(persistent_hotspots),
        }

        if new_hotspots:
            comparison['changes'].append(f"NEW hotspots: {', '.join(list(new_hotspots)[:5])}")

        if resolved_hotspots:
            comparison['changes'].append(f"[OK] RESOLVED hotspots: {', '.join(list(resolved_hotspots)[:5])}")

        if persistent_hotspots:
            comparison['changes'].append(f"[WARN]  PERSISTENT hotspots: {len(persistent_hotspots)} territories")

        return comparison

    @staticmethod
    def print_comparison(comparison: Dict[str, Any]) -> None:
        """Pretty-print the comparison dict to the run log."""
        logger.info("\n" + "=" * 80)
        logger.info("HISTORICAL COMPARISON")
        logger.info("=" * 80)

        hc = comparison.get('hotspot_change', {})
        logger.info(f"\nHotspots: {hc.get('previous', 0)} → {hc.get('current', 0)} "
                    f"({hc.get('percent_change', 0):+.1f}%)")

        if comparison.get('changes'):
            logger.info("\nKey Changes:")
            for change in comparison['changes']:
                logger.info(f"  {change}")

        tc = comparison.get('territory_changes', {})
        logger.info(f"\nTerritory Status:")
        logger.info(f"  New hotspots: {len(tc.get('new', []))}")
        logger.info(f"  Resolved: {len(tc.get('resolved', []))}")
        logger.info(f"  Persistent: {len(tc.get('persistent', []))}")

        logger.info("=" * 80 + "\n")
