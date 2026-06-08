"""
Oblast (region) name labels on Ukraine choropleth maps.

The :func:`add_oblast_labels` routine annotates each oblast polygon with
its name (English or Ukrainian, picked from ``ADM1_EN`` / ``ADM1_UA``
with sensible fallbacks). Kyiv city gets bold styling to stand out as
the capital; every label sits inside a translucent white box so it
remains legible over any basemap.

Pulled from the analyzer's ``self.cfg['oblast_names_*']`` lookup tables
when present; otherwise the labels use the source column verbatim.
"""

from __future__ import annotations

from typing import Any, Dict

import geopandas as gpd


def add_oblast_labels(cfg, ax, gdf_oblast: gpd.GeoDataFrame, lang: str = 'en') -> None:
    """
    Add oblast (region) name labels to the map.

    Uses ADM1_EN (English) as primary label, falls back to ADM1_UA.
    Kyiv city gets bold styling to stand out. Labels use a white
    semi-transparent background box for readability over any basemap.

    Args:
        lang: 'en' for English, 'ua' for Ukrainian
    """
    # Determine name column (English or Ukrainian from GeoData)
    name_col = None
    if lang == 'ua':
        for col in ('ADM1_UA', 'ADM1_EN', 'NAME_1', 'name'):
            if col in gdf_oblast.columns:
                name_col = col
                break
    else:
        for col in ('ADM1_EN', 'ADM1_UA', 'NAME_1', 'name'):
            if col in gdf_oblast.columns:
                name_col = col
                break
    if name_col is None:
        return

    # Ukrainian short name mapping
    ua_short = cfg.get('oblast_names_ua', {})

    # Reproject to WGS84 for centroid calculation if needed, then back.
    # to_crs raises pyproj CRSError or ValueError when the source CRS
    # is missing or unprojectable; both are recoverable -- fall back to
    # the unprojected frame and continue with WGS84 representative
    # points (less accurate but still usable for a label location).
    try:
        gdf_proj = gdf_oblast.to_crs(epsg=32636)   # UTM zone 36N (Ukraine)
    except (ValueError, AttributeError) as e:
        logger.debug(f"UTM reprojection skipped ({e}); using source CRS for labels")
        gdf_proj = gdf_oblast.copy()

    # Get representative point (inside polygon) for each oblast
    gdf_proj['_label_geom'] = gdf_proj.geometry.representative_point()

    # Reproject label points back to map CRS
    try:
        label_gdf = gpd.GeoDataFrame(
            gdf_proj[[name_col, '_label_geom']].copy(),
            geometry='_label_geom',
            crs=gdf_proj.crs
        ).to_crs(gdf_oblast.crs)
    except (ValueError, AttributeError) as e:
        logger.debug(f"Label back-projection skipped ({e}); keeping UTM coordinates")
        label_gdf = gpd.GeoDataFrame(
            gdf_proj[[name_col, '_label_geom']].copy(),
            geometry='_label_geom',
            crs=gdf_oblast.crs
        )

    # Short name mapping: trim long Ukrainian oblast names to fit nicely
    short_names = {
        'Dnipropetrovska': 'Dnipro',
        'Dnipropetrovsk': 'Dnipro',
        'Ivano-Frankivska': 'Iv.-Frank.',
        'Ivano-Frankivsk': 'Iv.-Frank.',
        'Khmelnytska': 'Khmeln.',
        'Khmelnytsk': 'Khmeln.',
        'Kropyvnytska': 'Kropyvn.',
        'Kirovohradska': 'Kirovohr.',
        'Kirovohrad': 'Kirovohr.',
        'Transcarpathian': 'Zakarpattia',
        'Zakarpatska': 'Zakarpattia',
        'Zaporizka': 'Zaporizhzhia',
        'Zaporizhzhia': 'Zaporizhzhia',
    }

    kyiv_keywords = {'kyiv', 'київ', 'kiev'}

    for _, row in label_gdf.iterrows():
        raw_name = str(row[name_col])
        label = short_names.get(raw_name, raw_name)

        # Strip " Oblast" / " oblast" suffix for cleaner display
        for suffix in (' Oblast', ' oblast', ' Region', ' region',
                       ' область', ' Область'):
            if label.endswith(suffix):
                label = label[:-len(suffix)]
                break

        # Ukrainian translation
        if lang == 'ua':
            label = ua_short.get(label, label)

        is_kyiv = any(kw in label.lower() for kw in kyiv_keywords)

        pt = row['_label_geom']
        x, y = pt.x, pt.y

        ax.annotate(
            label,
            xy=(x, y),
            fontsize=6.5 if not is_kyiv else 7.5,
            fontweight='bold' if is_kyiv else 'normal',
            color='#1a1a2e' if not is_kyiv else '#8B0000',
            ha='center',
            va='center',
            zorder=5,
            bbox=dict(
                boxstyle='round,pad=0.15',
                facecolor='white',
                edgecolor='none',
                alpha=0.65,
            ),
        )

