"""
Emit the iterative-mode skipped-windows CSV (convergence-skip artefact).

When the rolling-window iterative analyser collects windows whose
posterior was flagged ``convergence_fatal``, this writer drops a CSV
alongside the aggregated hotspots report. The file is always written
-- if the list is empty, just the header row is emitted -- so a reader
auditing iteration coverage gets explicit evidence of "no failures"
rather than a missing file that could mean either "none happened" or
"the writer crashed".
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def write_skipped_windows_table(skipped_windows: List[Dict[str, Any]],
                                iterative_dir: Path, level_name: str = None) -> None:
    """Write ``SkippedWindows_convergence_fatal[_<level>].csv`` into ``iterative_dir``.

    ``level_name`` is appended to the filename so per-level iterative runs do
    not overwrite each other's skip tables.
    """
    try:
        suffix = f'_{level_name}' if level_name else ''
        skipped_path = iterative_dir / f'SkippedWindows_convergence_fatal{suffix}.csv'
        if skipped_windows:
            pd.DataFrame(skipped_windows).to_csv(skipped_path, index=False)
            logger.warning(
                f"{len(skipped_windows)} window(s) flagged convergence_fatal — "
                f"see {skipped_path}"
            )
        else:
            skipped_path.write_text(
                "iteration,analysis_start,analysis_end,pct_divergences,rhat_max,ess_min,reason\n",
                encoding='utf-8',
            )
            logger.info(f"No windows skipped for convergence_fatal — empty table at {skipped_path}")
    except (IOError, OSError) as e:
        logger.error(f"Could not write skipped-windows table: {e}")
