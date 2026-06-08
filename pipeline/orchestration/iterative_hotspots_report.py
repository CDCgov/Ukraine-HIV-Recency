"""
Iterative-mode aggregated hotspots report.

After the rolling-window analysis finishes, this routine concatenates
the per-iteration hotspot GeoDataFrames, looks up each hex's community
/ rayon / oblast from ADM3, and joins case dates filtered to the
iteration's analysis window. The result is written as a two-sheet
Excel: ``All_Hotspots`` plus a ``Summary`` of counts by reliability
band.

ADM lookup is by point-in-polygon on the hex centroid (one read of
the ADM3 layer per matched hex, kept here for parity with the
original behaviour even though it's quadratic on hot data).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import h3
import pandas as pd

logger = logging.getLogger(__name__)


def _write_admin_iterative_report(df_all: pd.DataFrame, output_dir: Path,
                                  config: Dict[str, Any], run_timestamp: str,
                                  level_name: str) -> None:
    """Aggregated iterative report for an admin (oblast) level.

    The territory is the oblast itself, so there is no hex centroid to
    reverse-geocode: the oblast name comes straight off each row, and the
    recent-case count is the aggregated ``recent_count_curr``. Per-case dates
    (a hex-level enrichment keyed on H3 ids) are not produced here.
    """
    oblast_col = config.get('administrative_units', {}).get('oblast_col', 'ADM1_EN')
    rows = []
    for _, row in df_all.iterrows():
        terr = row.get(oblast_col) or row.get('territory_name') or 'Unknown'
        rows.append({
            'oblast': terr,
            'iteration': row.get('iteration'),
            'analysis_period': row.get('analysis_period'),
            'analysis_start_date': row.get('analysis_start_date'),
            'analysis_end_date': row.get('analysis_end_date'),
            'classification': row.get('classification', 'No Data'),
            'exceedance_prob': row.get('exceedance_prob'),
            'smr_mean': row.get('smr_mean'),
            'smr_median': row.get('smr_median'),
            'sir_mean': row.get('sir_mean'),
            'recent_count_curr': row.get('recent_count_curr'),
            'all_tested_curr': row.get('all_tested_curr'),
            'recent_proportion_curr': row.get('recent_proportion_curr'),
            'reliability_score': row.get('reliability_score'),
            'reliability_category': row.get('reliability_category'),
        })
    df_final = pd.DataFrame(rows).sort_values(
        ['iteration', 'exceedance_prob'], ascending=[True, False])

    output_file = output_dir / f'Iterative_Hotspots_Report_{level_name}_{run_timestamp}.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_final.to_excel(writer, sheet_name='All_Hotspots', index=False)
        summary = {
            'Total iterations': [df_final['iteration'].max()],
            'Total hotspot detections': [len(df_final)],
            'Unique oblasts': [df_final['oblast'].nunique()],
            'Date range': [f"{df_final['analysis_start_date'].min().date()} to "
                           f"{df_final['analysis_end_date'].max().date()}"],
            'Average reliability score': [df_final['reliability_score'].mean()],
        }
        pd.DataFrame(summary).to_excel(writer, sheet_name='Summary', index=False)

    logger.info(f"[OK] Iterative hotspots report saved: {output_file}")
    logger.info(f"   Total hotspots: {len(df_final)}  Unique oblasts: {df_final['oblast'].nunique()}")


def generate_iterative_hotspots_report(all_hotspots: List[gpd.GeoDataFrame],
                                       output_dir: Path,
                                       config: Dict[str, Any],
                                       run_timestamp: str,
                                       level=None) -> None:
    """Write the iterative-mode aggregated hotspots report.

    ``level`` is the analysis level (an H3 resolution int, or 'Oblast' for
    ADM1). For hex levels each hotspot hex is reverse-geocoded to its
    community / rayon / oblast and enriched with recent-case dates; for the
    ADM1 level the territory is the oblast itself, so that hex-specific
    enrichment is skipped. The filename carries the level so multiple levels
    in one run do not overwrite each other.
    """
    df_all = pd.concat(all_hotspots, ignore_index=True)
    logger.info(f"Total hotspots across all iterations: {len(df_all)}")

    level_name = (f'Hex_Res{level}' if isinstance(level, int) else str(level)) if level is not None else 'Hex'
    is_admin = (level is not None and not isinstance(level, int)) or ('h3_id' not in df_all.columns)
    if is_admin:
        _write_admin_iterative_report(df_all, output_dir, config, run_timestamp, level_name)
        return

    # Get first and last case dates for each hexagon
    excel_path = Path(config['excel_path'])
    df_cases = pd.read_excel(excel_path, sheet_name='hiv_cases')
    df_cases['test_date'] = pd.to_datetime(df_cases['test_date'])

    # Generate h3_id for each case based on coordinates (resolution 4)
    if 'latitude' in df_cases.columns and 'longitude' in df_cases.columns:
        logger.info("Generating H3 IDs (resolution 4) for cases in report...")
        df_cases['h3_id'] = df_cases.apply(
            lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], 4)
            if pd.notna(row['latitude']) and pd.notna(row['longitude']) else None,
            axis=1
        )
        h3_count = df_cases['h3_id'].notna().sum()
        logger.info(f"Generated H3 IDs for {h3_count}/{len(df_cases)} cases")
    else:
        logger.warning("latitude/longitude columns not found - case dates will not be populated")

    # Load the hexagons at the iterative resolution to get community names
    iter_res = int(config.get('iterative_resolution', 4))
    hex_path = Path(config['h3_hexagons'][f'res{iter_res}_path'])
    gdf_hex = gpd.read_file(hex_path)

    # Prepare final table
    final_rows = []

    for _, row in df_all.iterrows():
        hex_id = row.get('h3_id', row.get('territory_name'))

        # Get community name from centroid
        if hex_id in gdf_hex['h3_id'].values:
            hex_geom = gdf_hex[gdf_hex['h3_id'] == hex_id].iloc[0].geometry
            centroid = hex_geom.centroid

            # Find which admin unit contains this centroid
            adm3_path = Path(config['administrative_units']['adm3_path'])
            gdf_adm3 = gpd.read_file(adm3_path)

            community_name = "Unknown"
            oblast = "Unknown"
            rayon = "Unknown"

            for _, adm in gdf_adm3.iterrows():
                if adm.geometry.contains(centroid):
                    community_name = adm.get('ADM3_EN', 'Unknown')
                    oblast = adm.get('ADM1_EN', 'Unknown')
                    rayon = adm.get('ADM2_EN', 'Unknown')
                    break
        else:
            community_name = "Unknown"
            oblast = "Unknown"
            rayon = "Unknown"

        # Get case dates for this hexagon - filter by analysis period
        analysis_start = row['analysis_start_date']
        analysis_end = row['analysis_end_date']

        if 'h3_id' in df_cases.columns:
            # Filter cases for this hexagon within the analysis period
            hex_cases = df_cases[
                (df_cases['h3_id'] == hex_id) &
                (df_cases['test_date'] >= analysis_start) &
                (df_cases['test_date'] <= analysis_end)
            ]
        else:
            hex_cases = pd.DataFrame()  # Empty DataFrame if column doesn't exist

        if len(hex_cases) > 0:
            recent_cases = hex_cases[hex_cases['type'] == 'recent']
            if len(recent_cases) > 0:
                first_case_date = recent_cases['test_date'].min()
                last_case_date = recent_cases['test_date'].max()
                total_recent = len(recent_cases)
                # Collect all case_ids for recent cases
                case_ids = ', '.join(recent_cases['case_id'].astype(str).tolist())
            else:
                first_case_date = None
                last_case_date = None
                total_recent = 0
                case_ids = ''
        else:
            first_case_date = None
            last_case_date = None
            total_recent = 0
            case_ids = ''

        # Build row
        final_row = {
            'hex_id': hex_id,
            'community_name': community_name,
            'oblast': oblast,
            'rayon': rayon,
            'iteration': row['iteration'],
            'analysis_period': row['analysis_period'],
            'analysis_start_date': row['analysis_start_date'],
            'analysis_end_date': row['analysis_end_date'],
            'classification': row.get('classification', 'No Data'),
            'combined_z': row.get('combined_z', None),
            'exceedance_prob': row.get('exceedance_prob', None),
            'recent_count_curr': row.get('recent_count_curr', None),
            'all_tested_curr': row.get('all_tested_curr', None),
            'recent_proportion_curr': row.get('recent_proportion_curr', None),
            'predicted': row.get('predicted', None),
            'residual': row.get('residual', None),
            'reliability_score': row.get('reliability_score', None),
            'reliability_category': row.get('reliability_category', None),
            'data_adequacy_score': row.get('data_adequacy_score', None),
            'sample_size_score': row.get('sample_size_score', None),
            'model_quality_score': row.get('model_quality_score', None),
            'first_recent_case_date': first_case_date,
            'last_recent_case_date': last_case_date,
            'total_recent_cases': total_recent,
            'case_ids': case_ids
        }

        final_rows.append(final_row)

    # Create DataFrame
    df_final = pd.DataFrame(final_rows)

    # Sort by iteration (newest first), then by combined_z (highest first)
    df_final = df_final.sort_values(['iteration', 'exceedance_prob'], ascending=[True, False])

    # Save to Excel
    output_file = output_dir / f'Iterative_Hotspots_Report_{level_name}_{run_timestamp}.xlsx'

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Sheet 1: All hotspots
        df_final.to_excel(writer, sheet_name='All_Hotspots', index=False)

        # Sheet 2: Summary statistics
        summary_data = {
            'Total iterations': [df_final['iteration'].max()],
            'Total hotspots detected': [len(df_final)],
            'Unique hexagons': [df_final['hex_id'].nunique()],
            'Date range': [f"{df_final['analysis_start_date'].min().date()} to {df_final['analysis_end_date'].max().date()}"],
            'Average reliability score': [df_final['reliability_score'].mean()],
            'High reliability hotspots': [len(df_final[df_final['reliability_category'] == 'High'])],
            'Medium reliability hotspots': [len(df_final[df_final['reliability_category'] == 'Medium'])],
            'Low reliability hotspots': [len(df_final[df_final['reliability_category'] == 'Low'])]
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

    logger.info(f"[OK] Iterative hotspots report saved: {output_file}")
    logger.info(f"   Total hotspots: {len(df_final)}")
    logger.info(f"   Unique hexagons: {df_final['hex_id'].nunique()}")
    logger.info(f"   Iterations: {df_final['iteration'].max()}")
