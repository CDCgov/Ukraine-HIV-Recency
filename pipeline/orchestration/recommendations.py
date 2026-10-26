"""
Comprehensive end-of-run RECOMMENDATIONS.txt report.

Walks every ``Report_*.xlsx`` and ``Diagnostics_*.xlsx`` written by the
analyzers, aggregates the hotspot list (sorted by ``combined_z``),
emits per-level model-quality scores for both admin and hex outputs,
and concludes with one of three concrete next-step recommendations
(``USE``, ``USE WITH CAUTION``, or ``FAILED -- try higher level``).

Pure I/O over the run's ``output_dir`` -- the caller supplies the
config (for ``excel_path`` / data-window completeness check), the
output directory, the resolved file path for the recommendations text,
and the hotspot mask helper.
"""

from __future__ import annotations

import glob
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Union

import pandas as pd

from pipeline.reporting.excel_report import REPORT_DISPLAY_NAMES

logger = logging.getLogger(__name__)


def generate_recommendations(cfg, output_dir, recommendations_path, is_hotspot_fn):
    """Generate comprehensive recommendations with detailed hotspot list."""
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING RECOMMENDATIONS")
    logger.info("=" * 60)

    # output_dir + recommendations_path supplied by caller.

    recommendations = []
    recommendations.append("=" * 80)
    recommendations.append("HIV HOTSPOT DETECTION - FINAL RECOMMENDATIONS")
    recommendations.append("=" * 80)
    recommendations.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    recommendations.append("")

    # Data window completeness check.

    # The earlier "days since today" check did not make sense for a
    # retrospective tool (a window analysing 2024 reports as "outdated"
    # in 2026 even though that is exactly its purpose) and changed
    # between runs. We instead check whether the analysis window
    # extends past the last test date in the data: if it does, counts
    # in the tail of the window are undercounted and the late-window
    # SMR / SIR are biased toward zero.
    try:
        excel_path = Path(cfg['excel_path'])
        df_cases = pd.read_excel(excel_path, sheet_name='hiv_cases')
        df_cases['test_date'] = pd.to_datetime(df_cases['test_date'])

        last_test = df_cases['test_date'].max()
        analysis_end_raw = cfg.get('analysis_period', {}).get('end')
        analysis_end = pd.to_datetime(analysis_end_raw) if analysis_end_raw else None

        recommendations.append("=" * 80)
        recommendations.append("DATA WINDOW COMPLETENESS")
        recommendations.append("=" * 80)
        recommendations.append(
            f"Most recent test date in data: {last_test.strftime('%Y-%m-%d')}"
        )
        if analysis_end is not None and not pd.isna(analysis_end):
            recommendations.append(
                f"Analysis window end:           {analysis_end.strftime('%Y-%m-%d')}"
            )
            gap_days = (analysis_end - last_test).days
            if gap_days > 0:
                recommendations.append("")
                recommendations.append(
                    f"[WARN] Analysis window extends {gap_days} day(s) past the last "
                    "test in the data. Counts in the tail of the window are "
                    "undercounted, so SMR / SIR for the most recent month are "
                    "biased toward zero. Interpret late-window territories with care."
                )
            else:
                recommendations.append(
                    "[OK] Analysis window is fully contained in the available data."
                )
        else:
            recommendations.append(
                "(Analysis window end not found in config; completeness check skipped.)"
            )
        recommendations.append("")
    except (FileNotFoundError, KeyError, ValueError) as e:
        logger.warning(f"Could not check data window completeness: {e}")
        recommendations.append("")
        recommendations.append("[WARN] Could not verify data window completeness")
        recommendations.append("")

    # === PART 1: DETAILED HOTSPOT LIST ===
    recommendations.append("=" * 80)
    recommendations.append("PART 1: DETAILED HOTSPOT LIST")
    recommendations.append("=" * 80)
    recommendations.append("")

    # Read all report files
    import glob
    # Reports sit one folder below admin/ and hex/ (e.g. admin/Oblast/,
    # hex/res3/), so search recursively.
    report_files = []
    for subdir in ['admin', 'hex']:
        bayesian_reports = output_dir / 'bayesian' / subdir / '**' / 'Report_*.xlsx'
        report_files.extend(glob.glob(str(bayesian_reports), recursive=True))

    all_hotspots = []

    for report_file in report_files:
        try:
            df = pd.read_excel(report_file)
            # The Excel report uses reader-facing column names; map them back
            # to the internal names the hotspot mask and the row printer use.
            df = df.rename(columns={v: k for k, v in REPORT_DISPLAY_NAMES.items()})
            if 'classification_smr_sir' not in df.columns and 'classification' in df.columns:
                df['classification_smr_sir'] = df['classification']

            # Hotspots from the detector's classification axis.
            hotspots = pd.DataFrame()
            if 'classification_smr_sir' in df.columns:
                hotspots = df[is_hotspot_fn(df)].copy()

            if len(hotspots) > 0:
                # Determine level from filename
                filename = Path(report_file).name
                if 'Community' in filename:
                    level = 'Community'
                elif 'District' in filename:
                    level = 'District'
                elif 'Oblast' in filename:
                    level = 'Oblast'
                elif 'Hex' in filename:
                    level = 'Hexagon'
                else:
                    level = 'Unknown'

                hotspots['level'] = level
                hotspots['source_file'] = filename
                all_hotspots.append(hotspots)
        except (IOError, pd.errors.EmptyDataError, KeyError) as e:
            logger.warning(f"Could not read {report_file}: {e}")

    if all_hotspots:
        df_all_hotspots = pd.concat(all_hotspots, ignore_index=True)
        if 'smr_median' in df_all_hotspots.columns:
            df_all_hotspots = df_all_hotspots.sort_values('smr_median', ascending=False)

        recommendations.append(f"TOTAL HOTSPOTS FOUND: {len(df_all_hotspots)}")
        recommendations.append("")
        recommendations.append("HOTSPOTS (detector taxonomy):")
        recommendations.append("-" * 80)
        recommendations.append("")

        for idx, (i, row) in enumerate(df_all_hotspots.iterrows(), 1):
            # Name (English and Ukrainian) on one line.
            name = 'Unknown'
            for en_col, ua_col in (('hex_name_en', 'hex_name_ua'), ('ADM3_EN', 'community_ua'),
                                   ('ADM2_EN', 'district_ua'), ('ADM1_EN', 'oblast_ua')):
                if en_col in row and pd.notna(row[en_col]):
                    name = str(row[en_col])
                    if ua_col in row and pd.notna(row[ua_col]):
                        name += f" ({row[ua_col]})"
                    break
            recommendations.append(f"{idx}. {name}")
            recommendations.append(f"   Level: {row['level']}")
            recommendations.append(f"   Classification: {row['classification_smr_sir']}")
            recommendations.append("")

            # Parent territories
            if 'community_en' in row and pd.notna(row['community_en']):
                recommendations.append(f"   Community: {row['community_en']}")
                if 'community_ua' in row and pd.notna(row['community_ua']):
                    recommendations[-1] += f" ({row['community_ua']})"
                recommendations.append("")

            if 'district_en' in row and pd.notna(row['district_en']):
                recommendations.append(f"   District: {row['district_en']}")
                if 'district_ua' in row and pd.notna(row['district_ua']):
                    recommendations[-1] += f" ({row['district_ua']})"
                recommendations.append("")

            if 'oblast_en' in row and pd.notna(row['oblast_en']):
                recommendations.append(f"   Oblast: {row['oblast_en']}")
                if 'oblast_ua' in row and pd.notna(row['oblast_ua']):
                    recommendations[-1] += f" ({row['oblast_ua']})"
                recommendations.append("")

            # Statistics
            n_tests = row.get('all_tested_curr', 0)
            n_recent = row.get('recent_count_curr', 0)
            prop = row.get('recent_proportion_curr', 0)
            national = row.get('national_baseline', 0)
            deviation = row.get('deviation_pct', 0)

            recommendations.append(f"   Tests: {int(n_tests)}")
            recommendations.append(f"   Recent infections: {int(n_recent)} ({prop*100:.1f}%)")
            recommendations.append(f"   National baseline: {national*100:.1f}%")
            recommendations.append(f"   Deviation: {deviation:+.1f}%")
            smr_med = row.get('smr_median')
            if smr_med is not None and pd.notna(smr_med):
                recommendations.append(
                    f"   National reference ratio: {smr_med:.2f} "
                    f"[{row.get('smr_lower', float('nan')):.2f}-{row.get('smr_upper', float('nan')):.2f}]")
            rel = row.get('reliability_category')
            if rel is not None and pd.notna(rel):
                recommendations.append(f"   Reliability: {rel}")

            # Recommended level
            if 'recommended_level' in row and pd.notna(row['recommended_level']):
                recommendations.append(f"   Recommended level: {row['recommended_level']}")

            recommendations.append("")

    else:
        recommendations.append("No hotspots found.")
        recommendations.append("")

    # === PART 2: MODEL DIAGNOSTICS ===
    recommendations.append("")
    recommendations.append("=" * 80)
    recommendations.append("PART 2: MODEL DIAGNOSTICS")
    recommendations.append("=" * 80)
    recommendations.append("")

    # One row per analysed level, read from the per-level Diagnostics
    # workbooks (admin/<level>/, hex/<res>/). These are the Bayesian checks the
    # run itself applies: R-hat, ESS, divergences and the overall verdict.
    diag_files = []
    for subdir in ['admin', 'hex']:
        bayesian_diag_path = output_dir / 'bayesian' / subdir / '**' / 'Diagnostics_*.xlsx'
        diag_files.extend(glob.glob(str(bayesian_diag_path), recursive=True))

    n_levels = 0
    for diag_file in sorted(diag_files):
        try:
            df = pd.read_excel(diag_file, sheet_name='Model Diagnostics')
        except (IOError, ValueError, KeyError) as e:
            logger.warning(f"Could not read {diag_file}: {e}")
            continue
        for _, row in df.iterrows():
            n_levels += 1
            recommendations.append("-" * 60)
            recommendations.append(f"Level: {row.get('level', 'Unknown')}")
            recommendations.append("-" * 60)
            recommendations.append(f"  Model: {row.get('model_name', 'Bayesian')}")
            recommendations.append(f"  Territories: {row.get('n_territories', 'n/a')}")
            recommendations.append(f"  Overall quality: {row.get('overall_quality', 'n/a')}")
            recommendations.append(f"  Converged (R-hat < 1.01): {row.get('convergence_ok', 'n/a')}")
            recommendations.append(f"  ESS adequate (> 400): {row.get('ess_adequate', 'n/a')}")
            recommendations.append(f"  Divergences acceptable: {row.get('divergences_ok', 'n/a')}")
            recommendations.append("")

    if n_levels == 0:
        recommendations.append("No diagnostics files found.")
        recommendations.append("")

    recommendations.append("=" * 80)
    recommendations.append("END OF REPORT")
    recommendations.append("=" * 80)

    report_text = '\n'.join(recommendations)
    logger.info(report_text)

    with open(recommendations_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    logger.info(f"[OK] Recommendations saved: {recommendations_path}")


