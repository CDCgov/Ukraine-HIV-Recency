"""
Reliability choropleth map.

Colours each territory by the ``reliability_category`` produced by
:class:`~pipeline.diagnostics.reliability.ReliabilityScoreCalculator`
(HIGH / MODERATE / LOW). Used alongside the main anomaly map so a
reader can see at a glance which signals are decision-grade and which
need to be discounted.

The caller pre-loads the oblast boundary frame and resolves the output
path; this routine is pure draw-to-file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from pipeline.reporting.oblast_labels import add_oblast_labels

logger = logging.getLogger(__name__)


def render_reliability_map(cfg, gdf_oblast_opt, output_path, gdf_admin: gpd.GeoDataFrame, level_name: str,
                            start: pd.Timestamp, end: pd.Timestamp,
                            model_name: str, lang: str) -> Path:
    """Render a single reliability map in the specified language."""
    active = gdf_admin[gdf_admin['all_tested_curr'] > 0].copy()
    if active.empty:
        logger.warning(f"No data for {level_name} reliability map")
        return None

    if 'reliability_category' not in active.columns:
        logger.warning(f"reliability_category column not found - skipping reliability map")
        return None

    ua = cfg.get('map_strings_ua', {}) if lang == 'ua' else {}

    fig, ax = plt.subplots(figsize=(14, 14))

    reliability_cmap = {
        'HIGH': '#2ecc71',
        'MODERATE': '#f39c12',
        'LOW': '#e74c3c',
        'Unknown': '#95a5a6'
    }

    active['color'] = active['reliability_category'].map(reliability_cmap).fillna('#95a5a6')
    active.plot(ax=ax, color=active['color'], edgecolor='black', linewidth=0.3, alpha=0.8)

    # Add Ukraine oblast boundaries and labels
    if gdf_oblast_opt is not None:
        try:
            gdf_oblast_opt.boundary.plot(ax=ax, color='#444444', linewidth=1.0, alpha=0.7)
            add_oblast_labels(cfg, ax, gdf_oblast_opt, lang=lang)
        except (FileNotFoundError, KeyError, ValueError) as e:
            logger.warning(f"Could not add oblast boundaries: {e}")

    # Add context basemap (optional)
    try:
        ctx.add_basemap(ax, crs=active.crs.to_string(), source=ctx.providers.CartoDB.Positron, alpha=0.3)
    except (RuntimeError, IOError, ConnectionError) as e:
        logger.debug(f"Basemap not available: {e}")

    # Legend
    if lang == 'ua':
        legend_elements = [
            Patch(facecolor='#2ecc71', label=ua.get('High - Reliable for decisions', 'HIGH - Надійно для рішень')),
            Patch(facecolor='#f39c12', label=ua.get('Moderate - Use with caution', 'СЕРЕДНЯ — використовуйте обережно')),
            Patch(facecolor='#e74c3c', label=ua.get('Low - High uncertainty', 'НИЗЬКА — висока невизначеність'))
        ]
        legend_title = ua.get('Reliability Rating', 'Рівень надійності')
    else:
        legend_elements = [
            Patch(facecolor='#2ecc71', label='HIGH - Reliable for decisions'),
            Patch(facecolor='#f39c12', label='MODERATE - Use with caution'),
            Patch(facecolor='#e74c3c', label='LOW - High uncertainty')
        ]
        legend_title = "Reliability Rating"
    ax.legend(handles=legend_elements, title=legend_title, loc='lower right', fontsize=10)

    # Title
    if lang == 'ua':
        title = f"Карта надійності результатів — {level_name}\n"
        title += f"{ua.get('Model', 'Модель')}: {model_name}\n"
        title += f"{ua.get('Analysis period', 'Період аналізу')}: {start.date()} — {end.date()}\n"
        title += f"{ua.get('Based on', 'На основі')}: 100·exp(-CV) постеріору SMR/p; жорсткий gate: R-hat<1.01, ESS≥400, без fatal-дивергенцій"
    else:
        title = f"Results Reliability Map — {level_name}\n"
        title += f"Model: {model_name}\n"
        title += f"Analysis period: {start.date()} to {end.date()}\n"
        title += f"Based on: 100·exp(-CV) of posterior SMR/p; hard gate: R-hat<1.01, ESS>=400, no fatal divergences"
    plt.title(title, fontsize=14, fontweight='bold')
    ax.set_axis_off()

    # Statistics text box
    if lang == 'ua':
        stats_text = f"{ua.get('Territories by reliability', 'Території за надійністю')}:\n"
        ua_ratings = {'HIGH': 'Висока', 'MODERATE': 'Середня', 'LOW': 'Низька'}
        for rating in ['HIGH', 'MODERATE', 'LOW']:
            count = (active['reliability_category'] == rating).sum()
            pct = count / len(active) * 100 if len(active) > 0 else 0
            stats_text += f"{ua_ratings[rating]}: {count} ({pct:.0f}%)\n"
    else:
        stats_text = f"Territories by reliability:\n"
        for rating in ['HIGH', 'MODERATE', 'LOW']:
            count = (active['reliability_category'] == rating).sum()
            pct = count / len(active) * 100 if len(active) > 0 else 0
            stats_text += f"{rating}: {count} ({pct:.0f}%)\n"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Reliability map saved ({lang}): {output_path}")
    finally:
        plt.close(fig)

    return output_path

