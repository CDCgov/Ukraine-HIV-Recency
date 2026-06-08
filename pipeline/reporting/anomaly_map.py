"""
Choropleth anomaly map (the main "what's a hotspot" picture).

Renders the per-territory taxonomy on a Ukraine basemap. Uses the
SIR/SMR taxonomy (``classification_smr_sir``) when the model has
produced it and falls back to the legacy single-axis classification
otherwise. New sites (no historical baseline) get a small open-circle
marker at their centroid, and LOW-reliability territories get a hatched
overlay so the reader knows where to discount the colour.

Pulled out of :class:`BaseHotspotAnalyzer` with a callback-style
signature: the caller pre-loads ``gdf_oblast`` and supplies the resolved
``output_path``; this routine is pure draw-to-file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pipeline.reporting.oblast_labels import add_oblast_labels

logger = logging.getLogger(__name__)


def render_anomaly_map(cfg, national_baseline_rate, gdf_oblast_opt, output_path, gdf_admin: gpd.GeoDataFrame, level_name: str,
                start: pd.Timestamp, end: pd.Timestamp,
                b_start: pd.Timestamp, b_end: pd.Timestamp,
                model_name: str, lang: str) -> Path:
    """Render a single anomaly map in the specified language."""
    active = gdf_admin[gdf_admin['all_tested_curr'] > 0].copy()
    if active.empty:
        logger.warning(f"No data for {level_name} map")
        return None

    ua = cfg.get('map_strings_ua', {}) if lang == 'ua' else {}

    fig, ax = plt.subplots(figsize=(14, 14))
    cmap = cfg['color_map']
    # Prefer the SIR/SMR taxonomy when the model has produced it; fall
    # back to the legacy single-axis classification for older outputs.
    class_col = ('classification_smr_sir'
                 if 'classification_smr_sir' in active.columns
                 else 'classification')
    active['color'] = active[class_col].map(cmap).fillna('#ffffff')

    # Plot territories with their taxonomy colour.
    active.plot(ax=ax, color=active['color'], edgecolor='black',
                linewidth=0.3, alpha=0.8)

    # Mark new sites (no historical baseline, so SIR is undefined and the
    # taxonomy call rests on SMR alone) with a small open circle at the
    # hexagon centroid. This is a data-caveat marker, not a colour
    # category, so it survives any palette and remains readable in
    # greyscale prints and presentations.
    if 'is_new_site' in active.columns:
        new_sites = active[active['is_new_site'] == True]
        if not new_sites.empty:
            centroids = new_sites.geometry.centroid
            ax.scatter(centroids.x, centroids.y, s=40,
                       facecolors='none', edgecolors='black',
                       linewidths=1.2, zorder=5)

    # Add hatching for LOW reliability territories
    if 'reliability_category' in active.columns:
        low_reliability = active[active['reliability_category'] == 'LOW']
        if not low_reliability.empty:
            low_reliability.plot(ax=ax, color='none', edgecolor='black',
                                linewidth=0.3, hatch='///', alpha=0.6)

    # Add Ukraine oblast boundaries and labels
    if gdf_oblast_opt is not None:
        try:
            gdf_oblast_opt.boundary.plot(ax=ax, color='#444444', linewidth=1.0, alpha=0.7)
            add_oblast_labels(cfg, ax, gdf_oblast_opt, lang=lang)
        except (FileNotFoundError, KeyError, ValueError) as e:
            logger.warning(f"Could not add oblast boundaries: {e}")

    # Add context basemap (optional, may fail without internet)
    try:
        ctx.add_basemap(ax, crs=active.crs.to_string(), source=ctx.providers.CartoDB.Positron, alpha=0.3)
    except (RuntimeError, IOError, ConnectionError) as e:
        logger.debug(f"Basemap not available: {e}")

    # Legend: take labels from the same column that drove the colours,
    # so the legend reflects the taxonomy actually shown on the map.
    present_labels = set(active[class_col].unique())
    has_new_site = ('is_new_site' in active.columns
                    and bool(active['is_new_site'].any()))
    has_low_reliability = ('reliability_category' in active.columns
                           and (active['reliability_category'] == 'LOW').any())

    if lang == 'ua':
        legend_elements = [Patch(facecolor=c, label=ua.get(l, l))
                           for l, c in cmap.items() if l in present_labels]
        if has_new_site:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='none', markeredgecolor='black',
                       markersize=8, linestyle='None',
                       label=ua.get('New site (no baseline)',
                                    'Новий сайт (тренд недоступний)'))
            )
        if has_low_reliability:
            legend_elements.append(Patch(
                facecolor='white', edgecolor='black', hatch='///',
                label=ua.get('Low Reliability (weak data)',
                             'Low Reliability (weak data)')))
        legend_title = ua.get('Classification', 'Класифікація')
    else:
        legend_elements = [Patch(facecolor=c, label=l)
                           for l, c in cmap.items() if l in present_labels]
        if has_new_site:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='none', markeredgecolor='black',
                       markersize=8, linestyle='None',
                       label='New site (no baseline)')
            )
        if has_low_reliability:
            legend_elements.append(Patch(
                facecolor='white', edgecolor='black', hatch='///',
                label='Low Reliability (weak data)'))
        legend_title = "Classification"

    ax.legend(handles=legend_elements, title=legend_title,
              loc='lower right', fontsize=10)

    # Title
    if lang == 'ua':
        title = f"Виявлення гарячих точок ВІЛ — {level_name}\n"
        title += f"{ua.get('Model', 'Модель')}: {model_name}\n"
        title += f"{ua.get('Analysis period', 'Період аналізу')}: {start.date()} — {end.date()}"
        if b_start and b_end:
            title += f"\n{ua.get('Baseline period', 'Період базової лінії')}: {b_start.date()} — {b_end.date()}"
        if national_baseline_rate:
            title += f"\n{ua.get('National baseline', 'Національна базова лінія')}: {national_baseline_rate:.4f}"
    else:
        title = f"HIV Hotspot Detection — {level_name}\n"
        title += f"Model: {model_name}\n"
        title += f"Analysis period: {start.date()} to {end.date()}"
        if b_start and b_end:
            title += f"\nBaseline period: {b_start.date()} to {b_end.date()}"
        if national_baseline_rate:
            title += f"\nNational baseline: {national_baseline_rate:.4f}"
    plt.title(title, fontsize=14, fontweight='bold')
    ax.set_axis_off()

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Map saved ({lang}): {output_path}")
    finally:
        plt.close(fig)

    return output_path

