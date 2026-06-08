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

    for idx, row in gdf.iterrows():
        # Get centroid
        centroid = row.geometry.centroid
        point_gdf = gpd.GeoDataFrame([{'geometry': centroid}], geometry='geometry', crs=gdf.crs)

        # Find community
        community_match = gpd.sjoin(point_gdf, gdf_community, how='left', predicate='within')
        if len(community_match) > 0 and not community_match.iloc[0].isna().all():
            if otg_col in community_match.columns:
                comm_en = community_match.iloc[0][otg_col]
                gdf.at[idx, 'community_en'] = comm_en
                gdf.at[idx, 'hex_name_en'] = f"Hex in {comm_en}"
            if 'ADM3_UA' in community_match.columns:
                comm_ua = community_match.iloc[0]['ADM3_UA']
                gdf.at[idx, 'community_ua'] = comm_ua
                gdf.at[idx, 'hex_name_ua'] = f"Hexagon in {comm_ua}"

        # Find district
        district_match = gpd.sjoin(point_gdf, gdf_district, how='left', predicate='within')
        if len(district_match) > 0 and not district_match.iloc[0].isna().all():
            if rayon_col in district_match.columns:
                gdf.at[idx, 'district_en'] = district_match.iloc[0][rayon_col]
            if 'ADM2_UA' in district_match.columns:
                gdf.at[idx, 'district_ua'] = district_match.iloc[0]['ADM2_UA']

        # Find oblast
        oblast_match = gpd.sjoin(point_gdf, gdf_oblast, how='left', predicate='within')
        if len(oblast_match) > 0 and not oblast_match.iloc[0].isna().all():
            if oblast_col in oblast_match.columns:
                gdf.at[idx, 'oblast_en'] = oblast_match.iloc[0][oblast_col]
            if 'ADM1_UA' in oblast_match.columns:
                gdf.at[idx, 'oblast_ua'] = oblast_match.iloc[0]['ADM1_UA']

    return gdf

