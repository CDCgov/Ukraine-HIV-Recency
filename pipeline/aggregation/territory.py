"""
Territory-level aggregation for the two-period detector.

:func:`aggregate_stats` builds the per-territory frame the model consumes:
current/baseline test and recent counts, the ``site_present`` flag
(structural vs sampling zeros), and per-territory testing intensity
(test-months) plus network-stability z-scores.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from pipeline.aggregation.testing_network import (
    calculate_testing_intensity,
    classify_network_stability,
)

logger = logging.getLogger(__name__)


def aggregate_stats(cfg, testing_sites_df, gdf_admin: gpd.GeoDataFrame, gdf_cases: gpd.GeoDataFrame,
                   start: pd.Timestamp, end: pd.Timestamp,
                   b_start: pd.Timestamp, b_end: pd.Timestamp) -> gpd.GeoDataFrame:
    """
    Aggregate HIV testing statistics.

    CRITICAL FIX: Now distinguishes between structural zeros (no testing site)
    and sampling zeros (site exists but 0 recent cases).

    Args:
        gdf_admin: GeoDataFrame with administrative boundaries
        gdf_cases: GeoDataFrame with HIV testing cases
        start: Start date of current analysis period
        end: End date of current analysis period
        b_start: Start date of baseline period
        b_end: End date of baseline period

    Returns:
        GeoDataFrame with aggregated statistics per territory
    """
    gdf_admin = gdf_admin.copy()
    gdf_admin['all_tested_curr'] = 0
    gdf_admin['recent_count_curr'] = 0
    gdf_admin['all_tested_hist'] = 0
    gdf_admin['recent_count_hist'] = 0

    # Initialize site_present flag to distinguish structural vs sampling zeros
    gdf_admin['site_present'] = False

    # Current period
    curr_all = gdf_cases[(gdf_cases['test_date'] >= start) & (gdf_cases['test_date'] <= end)]
    curr_recent = curr_all[curr_all['type'] == 'recent']

    # Baseline period
    hist_all = gdf_cases[(gdf_cases['test_date'] >= b_start) & (gdf_cases['test_date'] <= b_end)]
    hist_recent = hist_all[hist_all['type'] == 'recent']

    # Spatial joins - ONE TIME for all territories
    # Validate CRS before spatial operations — reproject cases to match hexagons
    for name, gdf in [('curr_all', curr_all), ('curr_recent', curr_recent),
                      ('hist_all', hist_all), ('hist_recent', hist_recent)]:
        if gdf.crs is None:
            logger.warning(f"{name} has no CRS — setting to EPSG:4326")
            gdf = gdf.set_crs('EPSG:4326', allow_override=True)
        if gdf.crs != gdf_admin.crs:
            logger.warning(f"CRS mismatch in {name}: {gdf.crs} vs {gdf_admin.crs} — reprojecting")
            gdf = gdf.to_crs(gdf_admin.crs)
        if name == 'curr_all': curr_all = gdf
        elif name == 'curr_recent': curr_recent = gdf
        elif name == 'hist_all': hist_all = gdf
        elif name == 'hist_recent': hist_recent = gdf

    # Log records lost due to sjoin failures
    n_curr_all_before = len(curr_all)
    n_curr_recent_before = len(curr_recent)
    n_hist_all_before = len(hist_all)
    n_hist_recent_before = len(hist_recent)

    joined_curr_all = gpd.sjoin(curr_all, gdf_admin[['geometry']], how='left', predicate='within')
    joined_curr_recent = gpd.sjoin(curr_recent, gdf_admin[['geometry']], how='left', predicate='within')
    joined_hist_all = gpd.sjoin(hist_all, gdf_admin[['geometry']], how='left', predicate='within')
    joined_hist_recent = gpd.sjoin(hist_recent, gdf_admin[['geometry']], how='left', predicate='within')

    # Count records with missing index_right (failed to match any polygon)
    n_curr_all_lost = joined_curr_all['index_right'].isna().sum()
    n_curr_recent_lost = joined_curr_recent['index_right'].isna().sum()
    n_hist_all_lost = joined_hist_all['index_right'].isna().sum()
    n_hist_recent_lost = joined_hist_recent['index_right'].isna().sum()

    # Log warnings if >1% records lost
    total_lost = n_curr_all_lost + n_curr_recent_lost + n_hist_all_lost + n_hist_recent_lost
    total_records = n_curr_all_before + n_curr_recent_before + n_hist_all_before + n_hist_recent_before

    if total_records > 0:
        pct_lost = (total_lost / total_records) * 100
        if pct_lost > 1.0:
            logger.warning(f"[WARN] SJOIN DATA LOSS: {total_lost}/{total_records} records ({pct_lost:.2f}%) failed to match any polygon")
            logger.warning(f"  Current all: {n_curr_all_lost}/{n_curr_all_before} ({n_curr_all_lost/n_curr_all_before*100:.1f}%)")
            logger.warning(f"  Current recent: {n_curr_recent_lost}/{n_curr_recent_before} ({n_curr_recent_lost/n_curr_recent_before*100 if n_curr_recent_before > 0 else 0:.1f}%)")
            logger.warning(f"  Historical all: {n_hist_all_lost}/{n_hist_all_before} ({n_hist_all_lost/n_hist_all_before*100 if n_hist_all_before > 0 else 0:.1f}%)")
            logger.warning(f"  Historical recent: {n_hist_recent_lost}/{n_hist_recent_before} ({n_hist_recent_lost/n_hist_recent_before*100 if n_hist_recent_before > 0 else 0:.1f}%)")
            logger.warning("  Possible causes: boundary sites, incorrect coordinates, missing polygons")
        elif total_lost > 0:
            logger.info(f"SJOIN: {total_lost}/{total_records} records ({pct_lost:.2f}%) outside polygons (acceptable)")
        else:
            logger.info(f"SJOIN: All {total_records} records matched successfully")

    # Aggregate using groupby (FAST - no loop!)
    curr_all_counts = joined_curr_all.groupby('index_right').size()
    curr_recent_counts = joined_curr_recent.groupby('index_right').size()
    hist_all_counts = joined_hist_all.groupby('index_right').size()
    hist_recent_counts = joined_hist_recent.groupby('index_right').size()

    # Map counts back to gdf_admin
    gdf_admin['all_tested_curr'] = gdf_admin.index.map(curr_all_counts).fillna(0).astype(int)
    gdf_admin['recent_count_curr'] = gdf_admin.index.map(curr_recent_counts).fillna(0).astype(int)
    gdf_admin['all_tested_hist'] = gdf_admin.index.map(hist_all_counts).fillna(0).astype(int)
    gdf_admin['recent_count_hist'] = gdf_admin.index.map(hist_recent_counts).fillna(0).astype(int)

    # (Testing intensity per territory/window is computed further down in this
    # function via calculate_testing_intensity; the two-period detector reads
    # testing_intensity_curr/hist from there.)

    # Optional FRR correction (config-driven, off by default)
    frr = cfg.get('bayesian', {}).get('frr')
    if frr is not None and frr > 0:
        correction = np.round(gdf_admin['all_tested_curr'] * frr).astype(int)
        before = gdf_admin['recent_count_curr'].sum()
        gdf_admin['recent_count_curr'] = np.maximum(0, gdf_admin['recent_count_curr'] - correction)
        after = gdf_admin['recent_count_curr'].sum()
        logger.info(f"FRR correction: removed {before - after} false recent cases (FRR={frr})")
        # Also correct historical counts for consistency
        correction_hist = np.round(gdf_admin['all_tested_hist'] * frr).astype(int)
        gdf_admin['recent_count_hist'] = np.maximum(0, gdf_admin['recent_count_hist'] - correction_hist)

    # Mark territories with testing sites active during CURRENT period
    # Uses testing_sites sheet with activation/deactivation dates
    # This correctly handles wartime closures: if a site was active historically
    # but is now closed (e.g., occupied territory), it's NOT site_present
    try:
        if testing_sites_df is not None and len(testing_sites_df) > 0:
            df_sites = testing_sites_df
            # Filter sites active during current analysis period
            active_mask = df_sites['activation_date'] <= end
            # Deactivation: either no deactivation date (still active) or deactivated after period start
            active_mask = active_mask & (
                df_sites['deactivation_date'].isna() | (df_sites['deactivation_date'] >= start)
            )
            df_active_sites = df_sites[active_mask]

            if len(df_active_sites) > 0:
                # Spatial join: which hexagons contain active sites?
                gdf_active_sites = gpd.GeoDataFrame(
                    df_active_sites,
                    geometry=gpd.points_from_xy(df_active_sites.longitude, df_active_sites.latitude),
                    crs='EPSG:4326'
                )
                # Reproject sites to match hexagons CRS (not the other way around!)
                if gdf_active_sites.crs != gdf_admin.crs:
                    gdf_active_sites = gdf_active_sites.to_crs(gdf_admin.crs)
                sites_in_hex = gpd.sjoin(gdf_active_sites, gdf_admin, how='inner', predicate='within')
                hex_with_active_sites = set(sites_in_hex.index_right.unique())

                gdf_admin['site_present'] = gdf_admin.index.isin(hex_with_active_sites)
                logger.info(f"{len(df_active_sites)} sites active during {start.date()}-{end.date()}")
                logger.info(f"  → {gdf_admin['site_present'].sum()} hexagons with active sites")
            else:
                gdf_admin['site_present'] = gdf_admin['all_tested_curr'] > 0
                logger.warning("No sites active during current period — falling back to test counts")
        else:
            # No testing_sites sheet — fallback to original logic
            gdf_admin['site_present'] = (gdf_admin['all_tested_curr'] > 0) | (gdf_admin['all_tested_hist'] > 0)
            logger.info("No testing_sites data — using original site_present logic")
    except (KeyError, ValueError, AttributeError, IOError, FileNotFoundError) as e:
        logger.warning(f"Error loading testing_sites: {e} — using original site_present logic")
        gdf_admin['site_present'] = (gdf_admin['all_tested_curr'] > 0) | (gdf_admin['all_tested_hist'] > 0)

    # Log structural vs sampling zeros
    n_structural_zeros = (~gdf_admin['site_present']).sum()
    n_sampling_zeros = (gdf_admin['site_present'] & (gdf_admin['recent_count_curr'] == 0)).sum()
    n_active = gdf_admin['site_present'].sum()

    logger.info(f"Territory classification:")
    logger.info(f"  - Active sites: {n_active} ({n_active/len(gdf_admin)*100:.1f}%)")
    logger.info(f"  - Structural zeros (no site): {n_structural_zeros} ({n_structural_zeros/len(gdf_admin)*100:.1f}%)")
    logger.info(f"  - Sampling zeros (site present, 0 recent): {n_sampling_zeros}")

    # Proportions
    gdf_admin['recent_proportion_curr'] = np.where(
        gdf_admin['all_tested_curr'] > 0,
        gdf_admin['recent_count_curr'] / gdf_admin['all_tested_curr'],
        0
    )
    gdf_admin['recent_proportion_hist'] = np.where(
        gdf_admin['all_tested_hist'] > 0,
        gdf_admin['recent_count_hist'] / gdf_admin['all_tested_hist'],
        np.nan  # Use NaN instead of 0 for territories without historical data
    )

    # Calculate testing intensity for each territory
    logger.info("Calculating testing intensity (testo-months)...")

    gdf_admin['testing_intensity_curr'] = 0.0
    gdf_admin['testing_intensity_hist'] = 0.0
    gdf_admin['n_active_months_curr'] = 0
    gdf_admin['n_active_months_hist'] = 0

    for idx in range(len(gdf_admin)):
        territory_geom = gdf_admin.iloc[idx].geometry

        # Filter cases for this territory
        territory_cases_curr = curr_all[curr_all.geometry.within(territory_geom)]
        territory_cases_hist = hist_all[hist_all.geometry.within(territory_geom)]

        # Calculate intensity for current period
        intensity_curr = calculate_testing_intensity(territory_cases_curr, start, end)
        gdf_admin.at[idx, 'testing_intensity_curr'] = intensity_curr['weighted_intensity']
        gdf_admin.at[idx, 'n_active_months_curr'] = intensity_curr['n_active_months']

        # Calculate intensity for baseline period
        intensity_hist = calculate_testing_intensity(territory_cases_hist, b_start, b_end)
        gdf_admin.at[idx, 'testing_intensity_hist'] = intensity_hist['weighted_intensity']
        gdf_admin.at[idx, 'n_active_months_hist'] = intensity_hist['n_active_months']

    logger.info(f"[OK] Testing intensity calculated")
    logger.info(f"  Mean intensity (current): {gdf_admin['testing_intensity_curr'].mean():.1f} test-months")
    logger.info(f"  Mean intensity (baseline): {gdf_admin['testing_intensity_hist'].mean():.1f} test-months")

    # Classify network stability using z-score approach
    logger.info("Classifying network stability...")

    # Collect all intensity values for z-score calculation
    all_intensities_curr = gdf_admin['testing_intensity_curr'].values
    all_intensities_hist = gdf_admin['testing_intensity_hist'].values

    # Initialize columns
    gdf_admin['network_stability'] = 'UNKNOWN'
    gdf_admin['network_stability_z'] = 0.0
    gdf_admin['intensity_change_pct'] = 0.0

    for idx in range(len(gdf_admin)):
        intensity_curr = gdf_admin.at[idx, 'testing_intensity_curr']
        intensity_hist = gdf_admin.at[idx, 'testing_intensity_hist']

        # Classify stability
        stability = classify_network_stability(
            intensity_curr, intensity_hist,
            all_intensities_curr, all_intensities_hist
        )

        gdf_admin.at[idx, 'network_stability'] = stability['stability']
        gdf_admin.at[idx, 'network_stability_z'] = stability['z_score'] if stability['z_score'] is not None else 0.0
        gdf_admin.at[idx, 'intensity_change_pct'] = stability['relative_change'] * 100 if stability['relative_change'] is not None else 0.0

    # Log stability distribution
    stability_counts = gdf_admin['network_stability'].value_counts()
    logger.info(f"[OK] Network stability classified:")
    for category, count in stability_counts.items():
        pct = count / len(gdf_admin) * 100
        logger.info(f"  {category}: {count} ({pct:.1f}%)")

    return gdf_admin

