"""
Parent-territory enrichment for admin units and H3 hexagons.

Both Ukrainian-name lookups and parent-territory (district / oblast)
joins need access to the full hierarchy of admin boundaries. These
helpers take the analyzer's caching ``load_geodata_fn`` callback so the
already-cached GeoDataFrames are reused; if the analyzer has not seen a
layer yet, the callback transparently reads it from disk and caches it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import geopandas as gpd
import pandas as pd


def _unit_at_centroid(point_gdf: gpd.GeoDataFrame, gdf_units: gpd.GeoDataFrame,
                      sort_col: str):
    """Admin-unit row containing the hex centroid, or ``None`` if none does.

    A centroid on a shared boundary can match more than one polygon; sorting
    by ``sort_col`` makes the pick content-deterministic.
    """
    match = gpd.sjoin(point_gdf, gdf_units, how='left', predicate='within')
    if sort_col not in match.columns:
        return None
    match = match.sort_values(sort_col, na_position='last')
    if len(match) == 0 or pd.isna(match.iloc[0][sort_col]):
        return None
    return match.iloc[0]


def _community_for_border_hex(hex_geom, gdf_community: gpd.GeoDataFrame,
                              site_points: gpd.GeoDataFrame):
    """Community for a hex whose centroid lies outside Ukraine.

    A border hex (centroid in the sea or across the state border) is named
    after the community holding most of its testing sites -- e.g. the Odesa
    and Kharkiv city hexes. A hex with no site inside falls back to the
    community with the largest overlap.
    """
    if site_points is not None and len(site_points):
        inside = site_points[site_points.within(hex_geom)]
        if len(inside):
            # Count by polygon index, not by name: community names repeat
            # across oblasts (there is a Chernihivska hromada in Zaporizka).
            hits = gpd.sjoin(inside, gdf_community, how='inner', predicate='within')
            if len(hits):
                top = hits['index_right'].value_counts().sort_index().idxmax()
                return gdf_community.loc[top]
    overlap = gdf_community.geometry.intersection(hex_geom).area
    if len(overlap) == 0 or overlap.max() <= 0:
        return None
    return gdf_community.loc[overlap.idxmax()]


def _load_site_points(cfg, crs):
    """Testing-site locations as points in ``crs`` (``None`` if unavailable)."""
    try:
        sites = pd.read_excel(cfg['excel_path'], sheet_name='testing_sites')
    except (ValueError, IOError, KeyError):
        return None
    pts = gpd.GeoDataFrame(sites, geometry=gpd.points_from_xy(sites['longitude'], sites['latitude']),
                           crs='EPSG:4326')
    return pts.to_crs(crs)


def add_admin_territory_info(cfg, load_geodata_fn, gdf: gpd.GeoDataFrame, level_name: str) -> gpd.GeoDataFrame:
    """Add Ukrainian names and parent territories for admin units."""
    # Add Ukrainian name for current level
    if 'Community' in level_name and 'ADM3_UA' in gdf.columns:
        gdf['community_ua'] = gdf['ADM3_UA']
    elif 'District' in level_name and 'ADM2_UA' in gdf.columns:
        gdf['district_ua'] = gdf['ADM2_UA']
    elif 'Oblast' in level_name and 'ADM1_UA' in gdf.columns:
        gdf['oblast_ua'] = gdf['ADM1_UA']

    # Load parent territories
    if 'Community' in level_name:
        # Add district and oblast
        gdf_district = load_geodata_fn('District')
        gdf_oblast = load_geodata_fn('Oblast')

        gdf['district_en'] = ''
        gdf['district_ua'] = ''
        gdf['oblast_en'] = ''
        gdf['oblast_ua'] = ''

        for idx, row in gdf.iterrows():
            # Use centroid for spatial join
            centroid = row.geometry.centroid
            point_gdf = gpd.GeoDataFrame([{'geometry': centroid}], geometry='geometry', crs=gdf.crs)

            # Find district
            district_match = gpd.sjoin(point_gdf, gdf_district, how='left', predicate='within')
            if len(district_match) > 0 and not district_match.iloc[0].isna().all():
                rayon_col = cfg['administrative_units']['rayon_col']
                if rayon_col in district_match.columns:
                    gdf.at[idx, 'district_en'] = district_match.iloc[0][rayon_col]
                if 'ADM2_UA' in district_match.columns:
                    gdf.at[idx, 'district_ua'] = district_match.iloc[0]['ADM2_UA']

            # Find oblast
            oblast_match = gpd.sjoin(point_gdf, gdf_oblast, how='left', predicate='within')
            if len(oblast_match) > 0 and not oblast_match.iloc[0].isna().all():
                oblast_col = cfg['administrative_units']['oblast_col']
                if oblast_col in oblast_match.columns:
                    gdf.at[idx, 'oblast_en'] = oblast_match.iloc[0][oblast_col]
                if 'ADM1_UA' in oblast_match.columns:
                    gdf.at[idx, 'oblast_ua'] = oblast_match.iloc[0]['ADM1_UA']

    elif 'District' in level_name:
        # Add oblast
        gdf_oblast = load_geodata_fn('Oblast')

        gdf['oblast_en'] = ''
        gdf['oblast_ua'] = ''

        for idx, row in gdf.iterrows():
            # Use centroid for spatial join
            centroid = row.geometry.centroid
            point_gdf = gpd.GeoDataFrame([{'geometry': centroid}], geometry='geometry', crs=gdf.crs)

            oblast_match = gpd.sjoin(point_gdf, gdf_oblast, how='left', predicate='within')
            if len(oblast_match) > 0 and not oblast_match.iloc[0].isna().all():
                oblast_col = cfg['administrative_units']['oblast_col']
                if oblast_col in oblast_match.columns:
                    gdf.at[idx, 'oblast_en'] = oblast_match.iloc[0][oblast_col]
                if 'ADM1_UA' in oblast_match.columns:
                    gdf.at[idx, 'oblast_ua'] = oblast_match.iloc[0]['ADM1_UA']

    return gdf



def add_hex_territory_info(cfg, load_geodata_fn, gdf: gpd.GeoDataFrame, level_name: str) -> gpd.GeoDataFrame:
    """Add territory names for hexagons based on centroid location."""
    # Load all admin levels
    gdf_community = load_geodata_fn('Community')
    gdf_district = load_geodata_fn('District')
    gdf_oblast = load_geodata_fn('Oblast')

    # Get column names
    otg_col = cfg['administrative_units']['otg_col']
    rayon_col = cfg['administrative_units']['rayon_col']
    oblast_col = cfg['administrative_units']['oblast_col']

    # Initialize columns
    gdf['h3_id'] = gdf[cfg['h3_hexagons']['h3_id_col']] if cfg['h3_hexagons']['h3_id_col'] in gdf.columns else gdf.index
    gdf['hex_name_en'] = ''
    gdf['hex_name_ua'] = ''
    gdf['community_en'] = ''
    gdf['community_ua'] = ''
    gdf['district_en'] = ''
    gdf['district_ua'] = ''
    gdf['oblast_en'] = ''
    gdf['oblast_ua'] = ''

    site_points = None  # loaded only if a border hex needs it

    for idx, row in gdf.iterrows():
        # Get centroid
        centroid = row.geometry.centroid
        point_gdf = gpd.GeoDataFrame([{'geometry': centroid}], geometry='geometry', crs=gdf.crs)

        comm = _unit_at_centroid(point_gdf, gdf_community, otg_col)
        district = _unit_at_centroid(point_gdf, gdf_district, rayon_col)
        oblast = _unit_at_centroid(point_gdf, gdf_oblast, oblast_col)

        if comm is None:
            # Border hex: name it from its sites, and take district / oblast
            # from that same community so the three names stay consistent.
            if site_points is None:
                site_points = _load_site_points(cfg, gdf_community.crs)
            comm = _community_for_border_hex(row.geometry, gdf_community, site_points)
            if comm is not None:
                district = comm if rayon_col in comm.index else district
                oblast = comm if oblast_col in comm.index else oblast

        if comm is not None:
            if otg_col in comm.index:
                gdf.at[idx, 'community_en'] = comm[otg_col]
                gdf.at[idx, 'hex_name_en'] = f"Hex in {comm[otg_col]}"
            if 'ADM3_UA' in comm.index:
                gdf.at[idx, 'community_ua'] = comm['ADM3_UA']
                gdf.at[idx, 'hex_name_ua'] = f"Hexagon in {comm['ADM3_UA']}"

        if district is not None:
            if rayon_col in district.index:
                gdf.at[idx, 'district_en'] = district[rayon_col]
            if 'ADM2_UA' in district.index:
                gdf.at[idx, 'district_ua'] = district['ADM2_UA']

        if oblast is not None:
            if oblast_col in oblast.index:
                gdf.at[idx, 'oblast_en'] = oblast[oblast_col]
            if 'ADM1_UA' in oblast.index:
                gdf.at[idx, 'oblast_ua'] = oblast['ADM1_UA']

    return gdf

