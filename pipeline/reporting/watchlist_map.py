"""
Watch-list triage map.

Colours each territory by ``watch_reason`` -- why the combined burden + rate
watch-list flagged it (burden / rate / both) -- and labels the listed
territories with their ``watch_rank`` (1 = highest priority). It is the visual
companion to the watch-list columns in the Excel report
(see :func:`pipeline.classification.add_watchlist`): the anomaly map shows the
rigorous SMR/SIR classification, this map shows the triage priority.

The caller pre-loads the oblast boundary frame and resolves the output path;
this routine is pure draw-to-file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from pipeline.reporting.oblast_labels import add_oblast_labels

logger = logging.getLogger(__name__)


# Reason -> colour. Empty string (active but not listed) is muted grey.
_REASON_CMAP = {
    'both':   '#d62728',  # red — high on both burden and rate
    'burden': '#fb8500',  # orange — many recent cases, near-average rate
    'rate':   '#7b2cbf',  # purple — relatively elevated rate
    '':       '#d3d3d3',  # not on the watch-list
}

_REASON_LABEL_EN = {
    'both':   'Both — high burden & elevated rate',
    'burden': 'High burden (recent caseload)',
    'rate':   'Relatively elevated rate',
}
_REASON_LABEL_UA = {
    'both':   'Обидва — тягар і підвищена частка',
    'burden': 'Високий тягар (число нещодавніх)',
    'rate':   'Відносно підвищена частка',
}


def render_watchlist_map(cfg, gdf_oblast_opt, output_path, gdf_admin: gpd.GeoDataFrame,
                         level_name: str, start: pd.Timestamp, end: pd.Timestamp,
                         model_name: str, lang: str) -> Path:
    """Render a single watch-list map in the specified language."""
    if 'watch_reason' not in gdf_admin.columns or 'watch_rank' not in gdf_admin.columns:
        logger.warning("watch_reason/watch_rank columns not found - skipping watch-list map")
        return None

    active = gdf_admin[gdf_admin['all_tested_curr'] > 0].copy()
    if active.empty:
        logger.warning(f"No data for {level_name} watch-list map")
        return None

    reason = active['watch_reason'].fillna('').astype(str)
    active['color'] = reason.map(_REASON_CMAP).fillna('#d3d3d3')

    fig, ax = plt.subplots(figsize=(14, 14))
    active.plot(ax=ax, color=active['color'], edgecolor='black', linewidth=0.3, alpha=0.85)

    if gdf_oblast_opt is not None:
        try:
            gdf_oblast_opt.boundary.plot(ax=ax, color='#444444', linewidth=1.0, alpha=0.7)
            add_oblast_labels(cfg, ax, gdf_oblast_opt, lang=lang)
        except (FileNotFoundError, KeyError, ValueError) as e:
            logger.warning(f"Could not add oblast boundaries: {e}")

    try:
        ctx.add_basemap(ax, crs=active.crs.to_string(),
                        source=ctx.providers.CartoDB.Positron, alpha=0.3)
    except (RuntimeError, IOError, ConnectionError) as e:
        logger.debug(f"Basemap not available: {e}")

    # Label listed territories with their watch_rank at the polygon centroid.
    listed = active[reason.isin(['both', 'burden', 'rate']).values & active['watch_rank'].notna()]
    for _, row in listed.iterrows():
        try:
            c = row.geometry.centroid
            ax.annotate(str(int(row['watch_rank'])), xy=(c.x, c.y),
                        ha='center', va='center', fontsize=9, fontweight='bold',
                        color='white',
                        bbox=dict(boxstyle='circle,pad=0.15', facecolor='#111111', alpha=0.65,
                                  edgecolor='none'))
        except (ValueError, AttributeError):
            continue

    labels = _REASON_LABEL_UA if lang == 'ua' else _REASON_LABEL_EN
    legend_elements = [Patch(facecolor=_REASON_CMAP[k], label=labels[k])
                       for k in ('both', 'burden', 'rate')]
    legend_title = 'Список спостереження (причина)' if lang == 'ua' else 'Watch-list (reason)'
    ax.legend(handles=legend_elements, title=legend_title, loc='lower right', fontsize=10)

    if lang == 'ua':
        title = (f"Карта списку спостереження — {level_name}\n"
                 f"Модель: {model_name}\n"
                 f"Період аналізу: {start.date()} — {end.date()}\n"
                 f"Тріаж (тягар + частка); число = пріоритет (watch_rank). НЕ статистична значущість")
    else:
        title = (f"Watch-list (triage) Map — {level_name}\n"
                 f"Model: {model_name}\n"
                 f"Analysis period: {start.date()} to {end.date()}\n"
                 f"Triage (burden + rate); number = priority (watch_rank). NOT statistical significance")
    plt.title(title, fontsize=14, fontweight='bold')
    ax.set_axis_off()

    n_listed = int(active['on_watchlist'].fillna(False).sum()) if 'on_watchlist' in active.columns \
        else int(reason.isin(['both', 'burden', 'rate']).sum())
    if lang == 'ua':
        stats_text = f"У списку: {n_listed} із {len(active)}\n"
        for k, ua_lbl in (('both', 'обидва'), ('burden', 'тягар'), ('rate', 'частка')):
            stats_text += f"{ua_lbl}: {int((reason == k).sum())}\n"
    else:
        stats_text = f"On watch-list: {n_listed} of {len(active)}\n"
        for k in ('both', 'burden', 'rate'):
            stats_text += f"{k}: {int((reason == k).sum())}\n"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Watch-list map saved ({lang}): {output_path}")
    finally:
        plt.close(fig)

    return output_path
