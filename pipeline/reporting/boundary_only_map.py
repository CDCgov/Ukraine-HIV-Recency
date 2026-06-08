"""
Degenerate boundary-only map for the UNRELIABLE fallback.

When the posterior fails convergence (``convergence_fatal`` set), the
territory-level classification cannot be trusted; drawing the normal
choropleth would actively mislead the reader. This routine substitutes
a single PNG showing only oblast outlines and a red WARN banner with
the divergence diagnostics, so a reader scanning the output folder sees
the run is unreliable without having to open any other artefact.

Pulled out of :class:`BaseHotspotAnalyzer` so the rendering logic does
not need the analyzer's caches at call time -- the caller pre-loads the
``gdf_oblast`` boundary frame and resolves the output path; the routine
itself is pure draw-to-file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def render_boundary_only_map(cfg: Dict[str, Any], gdf_oblast: gpd.GeoDataFrame,
                             add_oblast_labels_fn: Callable[..., None],
                             output_path: Path, level_name: str,
                             start: pd.Timestamp, end: pd.Timestamp,
                             model_name: str, lang: str,
                             diagnostics: Dict[str, Any]) -> Optional[Path]:
    """Draw the oblast outline + WARN banner; save to ``output_path``."""
    fig, ax = plt.subplots(figsize=(14, 14))
    try:
        gdf_oblast.boundary.plot(ax=ax, color='#444444', linewidth=1.0, alpha=0.9)
        add_oblast_labels_fn(cfg, ax, gdf_oblast, lang=lang)
        ax.set_axis_off()

        pct_div = diagnostics.get('pct_divergences', 'N/A')
        rhat_max = diagnostics.get('rhat_max', 'N/A')
        ess_min = diagnostics.get('ess_alpha_min', diagnostics.get('min_ess_bulk', 'N/A'))
        banner = (
            f"UNRELIABLE -- convergence_fatal\n"
            f"{model_name} model failed to converge for {level_name}\n"
            f"Divergences: {pct_div}   R-hat max: {rhat_max}   ESS min: {ess_min}\n"
            f"Analysis period: {start.date() if start is not None else 'N/A'} to "
            f"{end.date() if end is not None else 'N/A'}\n"
            f"Per-territory results were intentionally not drawn."
        )
        ax.text(0.5, 0.5, banner, transform=ax.transAxes,
                fontsize=14, fontweight='bold', color='#aa0000',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#ffe5e5',
                          edgecolor='#aa0000', linewidth=2))

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.warning(f"UNRELIABLE map saved: {output_path}")
        return output_path
    finally:
        plt.close(fig)
