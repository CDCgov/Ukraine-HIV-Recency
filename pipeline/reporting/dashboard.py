"""
Summary dashboard rendered as a single PNG with five panels.

The :class:`SummaryDashboard` is used at the end of a pipeline run to
produce a one-glance overview: hotspot counts per administrative level,
per-model quality flags, the overall testing summary, the convergence
status of each model and a table of the top ten hotspots. Pure
matplotlib; no PyMC dependency.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SummaryDashboard:
    """One-page run summary as a PNG."""

    @staticmethod
    def create_dashboard(results: Dict[str, Any], output_file: Path) -> None:
        """Render the five-panel summary dashboard for the run.

        Expects a results dict with ``hotspots_by_level``, ``model_quality``,
        ``test_summary``, ``convergence_summary`` and ``top_hotspots`` keys
        (each optional; missing panels are simply skipped). Writes the PNG
        to ``output_file``.
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        levels = list(results.get('hotspots_by_level', {}).keys())
        counts = [sum(v.values()) if isinstance(v, dict) else v
                  for v in results.get('hotspots_by_level', {}).values()]
        ax1.barh(levels, counts, color='steelblue')
        ax1.set_xlabel('Number of Hotspots')
        ax1.set_title('Hotspots by Administrative Level')

        ax2 = fig.add_subplot(gs[0, 1])
        quality_data = results.get('model_quality', {})
        if quality_data:
            models = list(quality_data.keys())
            qualities = [1 if q == 'GOOD' else 0.5 if q == 'ACCEPTABLE' else 0
                         for q in quality_data.values()]
            colors = ['green' if q == 1 else 'orange' if q == 0.5 else 'red'
                      for q in qualities]
            ax2.barh(models, qualities, color=colors)
            ax2.set_xlim([0, 1])
            ax2.set_xlabel('Quality Score')
            ax2.set_title('Model Quality by Level')

        ax3 = fig.add_subplot(gs[0, 2])
        test_data = results.get('test_summary', {})
        if test_data:
            categories = ['Total Tests', 'Recent Infections', 'Long-term', 'Negative']
            values = [
                test_data.get('total_tests', 0),
                test_data.get('recent', 0),
                test_data.get('long_term', 0),
                test_data.get('negative', 0),
            ]
            ax3.bar(categories, values, color=['steelblue', 'red', 'gray', 'green'])
            ax3.set_ylabel('Count')
            ax3.set_title('Test Results Summary')
            ax3.tick_params(axis='x', rotation=45)

        ax4 = fig.add_subplot(gs[1, :])
        convergence_data = results.get('convergence_summary', {})
        if convergence_data:
            levels = list(convergence_data.keys())
            converged = [1 if convergence_data[l].get('converged') else 0 for l in levels]
            colors = ['green' if c else 'red' for c in converged]
            ax4.barh(levels, converged, color=colors)
            ax4.set_xlim([0, 1.2])
            ax4.set_xlabel('Converged (1) / Failed (0)')
            ax4.set_title('Model Convergence Status')

        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        top_hotspots = results.get('top_hotspots', [])[:10]
        if top_hotspots:
            table_data = []
            for hs in top_hotspots:
                table_data.append([
                    hs.get('territory', 'N/A'),
                    f"{hs.get('combined_z', 0):.2f}",
                    f"{hs.get('rate', 0) * 100:.1f}%",
                    str(hs.get('n_tests', 0)),
                ])

            table = ax5.table(
                cellText=table_data,
                colLabels=['Territory', 'Z-score', 'Rate', 'Tests'],
                cellLoc='left',
                loc='center',
                bbox=[0, 0, 1, 1],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            ax5.set_title('Top 10 Hotspots', pad=20, fontsize=12, fontweight='bold')

        plt.suptitle('HIV Hotspot Detection - Summary Dashboard',
                     fontsize=16, fontweight='bold', y=0.98)

        try:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"[OK] Dashboard saved: {output_file}")
        finally:
            plt.close(fig)
