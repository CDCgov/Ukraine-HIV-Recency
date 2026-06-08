"""
Per-level Excel report writer.

The :func:`write_report` routine writes the main ``Report_<level>_<period>.xlsx``
file with three sheets: the per-territory data (with ID + Ukrainian-name
columns, current/baseline counts, classification, reliability), the
mandatory facility-based-surveillance disclaimer, and a short metadata
block summarising the run. Column selection is driven by ``cfg`` so the
appropriate ID column (``otg_col`` / ``rayon_col`` / ``oblast_col``) is
chosen automatically per admin level.

The caller (analyzer wrapper) is responsible for:

* deciding whether the convergence gate fires (``convergence_fatal``) and writing
  the sentinel txt instead;
* filtering ``active`` and enriching it with parent-territory info and
  reliability scores;
* resolving ``output_path``;
* providing the facility-based disclaimer dict.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Union

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)


def write_report(cfg: Dict[str, Any], active: gpd.GeoDataFrame,
                 level_name: str, period_str: str,
                 output_path: Union[str, Path],
                 disclaimer: Dict[str, str]) -> None:
    """Pick the right columns for the level, then write the three-sheet Excel."""
    is_hex = 'Hex' in level_name

    if is_hex:
        report_cols = [
            'h3_id', 'hex_name_en', 'hex_name_ua',
            'community_en', 'community_ua', 'district_en', 'district_ua', 'oblast_en', 'oblast_ua',
            'all_tested_curr', 'recent_count_curr', 'recent_proportion_curr',
            'all_tested_hist', 'recent_count_hist', 'recent_proportion_hist',
            'national_baseline', 'deviation_pct',
            'z_national', 'z_residual', 'combined_z',
            'predicted', 'residual', 'classification',
            'reliability_score', 'reliability_category', 'reliability_flag',
        ]
    else:
        if 'Community' in level_name:
            report_cols = [
                cfg['administrative_units']['otg_col'], 'community_ua',
                'district_en', 'district_ua', 'oblast_en', 'oblast_ua',
                'all_tested_curr', 'recent_count_curr', 'recent_proportion_curr',
                'all_tested_hist', 'recent_count_hist', 'recent_proportion_hist',
                'national_baseline', 'deviation_pct',
                'z_national', 'z_residual', 'combined_z',
                'predicted', 'residual', 'classification',
                'reliability_score', 'reliability_category', 'reliability_flag',
            ]
        elif 'District' in level_name:
            report_cols = [
                cfg['administrative_units']['rayon_col'], 'district_ua',
                'oblast_en', 'oblast_ua',
                'all_tested_curr', 'recent_count_curr', 'recent_proportion_curr',
                'all_tested_hist', 'recent_count_hist', 'recent_proportion_hist',
                'national_baseline', 'deviation_pct',
                'z_national', 'z_residual', 'combined_z',
                'predicted', 'residual', 'classification',
                'reliability_score', 'reliability_category', 'reliability_flag',
            ]
        else:  # Oblast
            report_cols = [
                cfg['administrative_units']['oblast_col'], 'oblast_ua',
                'all_tested_curr', 'recent_count_curr', 'recent_proportion_curr',
                'all_tested_hist', 'recent_count_hist', 'recent_proportion_hist',
                'national_baseline', 'deviation_pct',
                'z_national', 'z_residual', 'combined_z',
                'predicted', 'residual', 'classification',
                'reliability_score', 'reliability_category', 'reliability_flag',
            ]

    if 'prob_lower' in active.columns:
        idx = report_cols.index('predicted') if 'predicted' in report_cols else len(report_cols)
        report_cols.insert(idx + 1, 'predicted_prob')
        report_cols.insert(idx + 2, 'prob_lower')
        report_cols.insert(idx + 3, 'prob_upper')

    # Audit M5 / D2: surface the raw SMR credible interval, the reliability CV
    # behind the 0-100 score, and the SIR-informativeness flag, so a reader
    # sees the actual uncertainty rather than only the reliability category.
    for _col in ['smr_mean', 'smr_median', 'smr_lower', 'smr_upper', 'sir_informative', 'reliability_cv']:
        if _col in active.columns and _col not in report_cols:
            report_cols.append(_col)

    # Combined burden + rate watch-list (pipeline.classification.add_watchlist):
    # an additive triage ranking that surfaces high-burden centres the rate
    # axis misses on sparse recency data.
    for _col in ['on_watchlist', 'watch_reason', 'watch_rank',
                 'burden_rank', 'rate_rank', 'burden_share_pct',
                 'burden_high', 'rate_high']:
        if _col in active.columns and _col not in report_cols:
            report_cols.append(_col)

    if 'high_outbreak' in active.columns:
        report_cols.extend(['high_outbreak', 'low_outbreak', 'testing_artifact',
                            'high_observed_curr', 'high_ci_upper',
                            'low_observed_curr', 'low_ci_upper'])

    if 'testing_intensity_curr' in active.columns:
        report_cols.extend(['testing_intensity_curr', 'testing_intensity_hist',
                            'n_active_months_curr', 'n_active_months_hist'])

    if 'network_stability' in active.columns:
        report_cols.extend(['network_stability', 'network_stability_z', 'intensity_change_pct'])

    available_cols = [col for col in report_cols if col in active.columns]
    df_report = active[available_cols].copy()

    if 'geometry' in df_report.columns:
        df_report = df_report.drop(columns=['geometry'])

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_report.to_excel(writer, sheet_name='Data', index=False)

        df_disclaimer = pd.DataFrame({
            'WARNING': ['FACILITY-BASED SURVEILLANCE DISCLAIMER'],
            'English': [disclaimer['en']],
            'Ukrainian': [disclaimer['ua']],
        })
        df_disclaimer.to_excel(writer, sheet_name='WARNING READ FIRST', index=False)

        metadata = {
            'Field': ['Analysis Period', 'Baseline Period', 'Level', 'Model Type',
                      'Total Territories', 'Active Sites', 'Structural Zeros',
                      'Data Type', 'Interpretation Warning'],
            'Value': [
                period_str,
                'Previous 12 months',
                level_name,
                'Bayesian',
                len(df_report),
                df_report['all_tested_curr'].gt(0).sum() if 'all_tested_curr' in df_report.columns else 'N/A',
                len(df_report) - df_report['all_tested_curr'].gt(0).sum() if 'all_tested_curr' in df_report.columns else 'N/A',
                'Facility-based surveillance',
                disclaimer['short_en'],
            ],
        }
        pd.DataFrame(metadata).to_excel(writer, sheet_name='Metadata', index=False)

    logger.info(f"[OK] Report saved with disclaimer: {output_path}")
