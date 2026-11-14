"""Shared helpers for the browser notebooks (JupyterLite or a normal kernel).

Everything here reads the precomputed files in ``data/``, which were produced
by running the unchanged pipeline on the public, fully synthetic dataset
(see ``jupyterlite/build_content.py`` in the repository).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent / 'data'

# Label colours as in the pipeline's config.json (color_map).
LABEL_COLORS = {
    'Established hotspot': '#d62728',
    'Emerging hotspot': '#fb8500',
    'Stable high-burden': '#ffd60a',
    'Elevated vs national (trend uncertain)': '#c77dff',
    'Declining from high-burden': '#48cae4',
    'Emerging decrease': '#a0e0a8',
    'Below national': '#06d6a0',
    'Normal': '#d3d3d3',
}
HOTSPOT_LABELS = ['Established hotspot', 'Emerging hotspot', 'Stable high-burden',
                  'Elevated vs national (trend uncertain)']

PYODIDE_PACKAGES = ['pandas', 'geopandas', 'shapely', 'pyproj', 'matplotlib', 'scipy']


def in_browser() -> bool:
    """True inside JupyterLite (Pyodide)."""
    return sys.platform == 'emscripten'


def build_info() -> dict:
    return json.loads((DATA / 'build_info.json').read_text(encoding='utf-8'))


def load_results():
    """H3 res-3 cells with the pipeline's labels and counts (GeoDataFrame)."""
    import geopandas as gpd
    return gpd.read_file(DATA / 'results_h3.geojson')


def load_oblasts():
    import geopandas as gpd
    return gpd.read_file(DATA / 'oblasts.geojson')


def load_sites():
    """Synthetic testing sites as a GeoDataFrame (random points within the true oblast)."""
    import geopandas as gpd
    import pandas as pd
    s = pd.read_csv(DATA / 'synthetic_sites.csv', parse_dates=['activation_date', 'deactivation_date'])
    return gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s.longitude, s.latitude), crs='EPSG:4326')


def load_cases():
    import pandas as pd
    return pd.read_csv(DATA / 'synthetic_cases.csv', parse_dates=['test_date'])


def plot_labels(cells, ax=None, oblasts=None, sites=None, title=None, legend=True,
                label_col='label', reliability_col='reliability'):
    """Map cells coloured by label, LOW reliability hatched (the pipeline's convention).

    Cells without active sites in the window are drawn as empty outlines.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6.5))
    crs = 'EPSG:3035'
    cells = cells.to_crs(crs)
    if oblasts is not None:
        oblasts.to_crs(crs).boundary.plot(ax=ax, color='#9a9a9a', linewidth=0.4, zorder=1)
    empty = cells[cells[label_col].isna()]
    if not empty.empty:
        empty.plot(ax=ax, facecolor='none', edgecolor='#cfcfcf', linewidth=0.3, zorder=2)
    filled = cells[cells[label_col].notna()]
    filled.plot(ax=ax, color=filled[label_col].map(LABEL_COLORS).fillna('#d3d3d3'),
                edgecolor='#555555', linewidth=0.5, alpha=0.9, zorder=3)
    low = filled[filled[reliability_col].eq('LOW')] if reliability_col in filled else filled.iloc[0:0]
    if not low.empty:
        plt.rcParams['hatch.linewidth'] = 0.5
        low.plot(ax=ax, facecolor='none', edgecolor='#333333', linewidth=0, hatch='////', zorder=4)
    if sites is not None:
        sites.to_crs(crs).plot(ax=ax, color='black', markersize=3, zorder=5)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    if legend:
        present = [k for k in LABEL_COLORS if k in set(filled[label_col])]
        handles = [Patch(facecolor=LABEL_COLORS[k], edgecolor='#555555', label=k) for k in present]
        if not low.empty:
            handles.append(Patch(facecolor='white', edgecolor='#333333', hatch='////',
                                 label='Low reliability (weak data)'))
        handles.append(Patch(facecolor='white', edgecolor='#cfcfcf', label='No active sites'))
        ax.legend(handles=handles, loc='lower left', fontsize=8, frameon=False)
    return ax
