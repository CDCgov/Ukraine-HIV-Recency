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
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Union

import pandas as pd

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
        analysis_end_raw = cfg.get('data', {}).get('end')
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
    report_files = []
    for subdir in ['admin', 'hex']:
        bayesian_reports = output_dir / 'bayesian' / subdir / 'Report_*.xlsx'
        report_files.extend(glob.glob(str(bayesian_reports)))
        bayesian_cov_reports = output_dir / 'bayesian_covariates' / subdir / 'Report_*.xlsx'
        report_files.extend(glob.glob(str(bayesian_cov_reports)))

    all_hotspots = []

    for report_file in report_files:
        try:
            df = pd.read_excel(report_file)

            # Filter hotspots
            # Traditional: classification == 'Obvious Increase'
            # Bayesian Covariates: high_outbreak or low_outbreak
            hotspots = pd.DataFrame()

            if 'classification' in df.columns and 'combined_z' in df.columns:
                hotspots_traditional = df[is_hotspot_fn(df)].copy()
                if len(hotspots_traditional) > 0:
                    hotspots = pd.concat([hotspots, hotspots_traditional], ignore_index=True)

            # Check for Bayesian Covariates outbreaks
            if 'high_outbreak' in df.columns:
                hotspots_high = df[df['high_outbreak'] == True].copy()
                if len(hotspots_high) > 0:
                    hotspots_high['outbreak_type'] = 'HIGH-RISK GROUP'
                    hotspots = pd.concat([hotspots, hotspots_high], ignore_index=True)

            if 'low_outbreak' in df.columns:
                hotspots_low = df[df['low_outbreak'] == True].copy()
                if len(hotspots_low) > 0:
                    hotspots_low['outbreak_type'] = 'LOW-RISK GROUP'
                    hotspots = pd.concat([hotspots, hotspots_low], ignore_index=True)

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
        df_all_hotspots = df_all_hotspots.sort_values('combined_z', ascending=False)

        recommendations.append(f"TOTAL HOTSPOTS FOUND: {len(df_all_hotspots)}")
        recommendations.append("")
        recommendations.append("OBVIOUS INCREASE HOTSPOTS (Z > 2.0):")
        recommendations.append("-" * 80)
        recommendations.append("")

        for idx, (i, row) in enumerate(df_all_hotspots.iterrows(), 1):
            recommendations.append(f"{idx}. ", )

            # Name (English and Ukrainian)
            if 'hex_name_en' in row and pd.notna(row['hex_name_en']):
                recommendations.append(f"{row['hex_name_en']}")
                if 'hex_name_ua' in row and pd.notna(row['hex_name_ua']):
                    recommendations.append(f" ({row['hex_name_ua']})")
            elif 'ADM3_EN' in row and pd.notna(row['ADM3_EN']):
                recommendations.append(f"{row['ADM3_EN']}")
                if 'community_ua' in row and pd.notna(row['community_ua']):
                    recommendations.append(f" ({row['community_ua']})")
            elif 'ADM2_EN' in row and pd.notna(row['ADM2_EN']):
                recommendations.append(f"{row['ADM2_EN']}")
                if 'district_ua' in row and pd.notna(row['district_ua']):
                    recommendations.append(f" ({row['district_ua']})")
            elif 'ADM1_EN' in row and pd.notna(row['ADM1_EN']):
                recommendations.append(f"{row['ADM1_EN']}")
                if 'oblast_ua' in row and pd.notna(row['oblast_ua']):
                    recommendations.append(f" ({row['oblast_ua']})")

            recommendations[-1] = ''.join(recommendations[-1:])  # Join last line
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
            z_score = row.get('combined_z', 0)
            deviation = row.get('deviation_pct', 0)

            # Outbreak type (if from Bayesian Covariates)
            if 'outbreak_type' in row and pd.notna(row['outbreak_type']):
                recommendations.append(f"   OUTBREAK TYPE: {row['outbreak_type']}")
                if 'high_observed_curr' in row and pd.notna(row['high_observed_curr']):
                    recommendations.append(f"   High-risk group: {row['high_observed_curr']*100:.1f}% (expected max: {row.get('high_ci_upper', 0)*100:.1f}%)")
                if 'low_observed_curr' in row and pd.notna(row['low_observed_curr']):
                    recommendations.append(f"   Low-risk group: {row['low_observed_curr']*100:.1f}% (expected max: {row.get('low_ci_upper', 0)*100:.1f}%)")
                recommendations.append("")

            recommendations.append(f"   Tests: {int(n_tests)}")
            recommendations.append(f"   Recent infections: {int(n_recent)} ({prop*100:.1f}%)")
            recommendations.append(f"   National baseline: {national*100:.1f}%")
            recommendations.append(f"   Deviation: {deviation:+.1f}%")
            recommendations.append(f"   Z-score: {z_score:.2f}")

            # Recommended level
            if 'recommended_level' in row and pd.notna(row['recommended_level']):
                recommendations.append(f"   Recommended level: {row['recommended_level']}")

            recommendations.append("")

    else:
        recommendations.append("No obvious increase hotspots found.")
        recommendations.append("")

    # === PART 2: MODEL DIAGNOSTICS ===
    recommendations.append("")
    recommendations.append("=" * 80)
    recommendations.append("PART 2: MODEL DIAGNOSTICS")
    recommendations.append("=" * 80)
    recommendations.append("")

    # Read diagnostics from files
    diag_files = []
    for subdir in ['admin', 'hex']:
        bayesian_diag_path = output_dir / 'bayesian' / subdir / 'Diagnostics_*.xlsx'
        diag_files.extend(glob.glob(str(bayesian_diag_path)))
        bayesian_cov_diag_path = output_dir / 'bayesian_covariates' / subdir / 'Diagnostics_*.xlsx'
        diag_files.extend(glob.glob(str(bayesian_cov_diag_path)))

    # Collect diagnostics by category
    admin_diagnostics = {}
    hex_diagnostics = {}

    for diag_file in diag_files:
        try:
            df = pd.read_excel(diag_file, sheet_name='Model Diagnostics')
            for _, row in df.iterrows():
                level = row.get('level', 'Unknown')
                model_name = row.get('model_name', 'Bayesian')
                converged = row.get('converged', False)
                r2 = row.get('pseudo_r2', 0)
                normal = row.get('residuals_normal', 'No')

                diag_info = {
                    'model': model_name,
                    'converged': converged,
                    'r2': r2,
                    'normal': normal,
                    'file': diag_file
                }

                # Categorize by admin or hex
                if 'Hex' in level:
                    hex_diagnostics[level] = diag_info
                else:
                    admin_diagnostics[level] = diag_info

        except (IOError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not read {diag_file}: {e}")

    # === ADMINISTRATIVE UNITS RECOMMENDATIONS ===
    if admin_diagnostics:
        recommendations.append("=" * 80)
        recommendations.append("ADMINISTRATIVE UNITS - LEVEL RECOMMENDATIONS")
        recommendations.append("=" * 80)
        recommendations.append("")

        admin_order = ['Community', 'District', 'Oblast']
        admin_quality = {}

        for level in admin_order:
            if level in admin_diagnostics:
                diag = admin_diagnostics[level]
                converged = diag['converged']
                r2 = diag['r2']
                normal = diag['normal']
                model = diag['model']

                recommendations.append("-" * 60)
                recommendations.append(f"Level: {level}")
                recommendations.append("-" * 60)
                recommendations.append(f"  Model Used: {model}")
                recommendations.append(f"  Converged: {converged}")
                if r2:
                    recommendations.append(f"  Pseudo R²: {r2:.4f}")
                recommendations.append(f"  Residuals Normal: {normal}")

                # Quality score
                quality_score = 0
                if converged:
                    quality_score += 1
                if r2 and r2 > 0.05:
                    quality_score += 2
                if normal == 'Yes':
                    quality_score += 1

                admin_quality[level] = quality_score

                if quality_score >= 3:
                    recommendations.append(f"  [OK] Quality: GOOD")
                elif quality_score >= 2:
                    recommendations.append(f"  [WARN]  Quality: ACCEPTABLE")
                else:
                    recommendations.append(f"  [FAIL] Quality: POOR")
                recommendations.append("")

        # Overall recommendation for admin levels
        recommendations.append("=" * 60)
        recommendations.append("RECOMMENDED ADMINISTRATIVE LEVEL FOR DECISION-MAKING:")
        recommendations.append("=" * 60)

        if admin_quality:
            best_level = max(admin_quality, key=admin_quality.get)
            best_score = admin_quality[best_level]

            if best_score >= 3:
                recommendations.append(f"→ USE {best_level.upper()} LEVEL")
                recommendations.append(f"  Reason: Good model fit, reliable results")
            elif best_score >= 2:
                recommendations.append(f"→ USE {best_level.upper()} LEVEL WITH CAUTION")
                recommendations.append(f"  Reason: Acceptable fit, but verify hotspots manually")
            else:
                # All levels poor - recommend higher aggregation
                if best_level == 'Community':
                    recommendations.append(f"→ COMMUNITY LEVEL FAILED - TRY DISTRICT LEVEL")
                    recommendations.append(f"  Reason: Insufficient data at community level")
                elif best_level == 'District':
                    recommendations.append(f"→ DISTRICT LEVEL FAILED - TRY OBLAST LEVEL")
                    recommendations.append(f"  Reason: Insufficient data at district level")
                else:
                    recommendations.append(f"→ ALL LEVELS SHOW POOR FIT")
                    recommendations.append(f"  Reason: Consider extending analysis period or checking data quality")

            # Specific recommendations for failed levels
            for level in admin_order:
                if level in admin_quality and admin_quality[level] < 2:
                    if level == 'Community' and 'District' in admin_quality:
                        recommendations.append(f"  [WARN]  Community level unreliable → Use District level instead")
                    elif level == 'District' and 'Oblast' in admin_quality:
                        recommendations.append(f"  [WARN]  District level unreliable → Use Oblast level instead")

        recommendations.append("")

    # === H3 HEXAGONS RECOMMENDATIONS ===
    if hex_diagnostics:
        recommendations.append("=" * 80)
        recommendations.append("H3 HEXAGONS - RESOLUTION RECOMMENDATIONS")
        recommendations.append("=" * 80)
        recommendations.append("")

        hex_order = ['Hex_Res3', 'Hex_Res4']
        hex_quality = {}

        for level in hex_order:
            if level in hex_diagnostics:
                diag = hex_diagnostics[level]
                converged = diag['converged']
                r2 = diag['r2']
                normal = diag['normal']
                model = diag['model']

                recommendations.append("-" * 60)
                recommendations.append(f"Level: {level}")
                recommendations.append("-" * 60)
                recommendations.append(f"  Model Used: {model}")
                recommendations.append(f"  Converged: {converged}")
                if r2:
                    recommendations.append(f"  Pseudo R²: {r2:.4f}")
                recommendations.append(f"  Residuals Normal: {normal}")

                # Quality score
                quality_score = 0
                if converged:
                    quality_score += 1
                if r2 and r2 > 0.05:
                    quality_score += 2
                if normal == 'Yes':
                    quality_score += 1

                hex_quality[level] = quality_score

                if quality_score >= 3:
                    recommendations.append(f"  [OK] Quality: GOOD")
                elif quality_score >= 2:
                    recommendations.append(f"  [WARN]  Quality: ACCEPTABLE")
                else:
                    recommendations.append(f"  [FAIL] Quality: POOR")
                recommendations.append("")

        # Overall recommendation for hex resolutions
        recommendations.append("=" * 60)
        recommendations.append("RECOMMENDED H3 RESOLUTION FOR DECISION-MAKING:")
        recommendations.append("=" * 60)

        if hex_quality:
            best_level = max(hex_quality, key=hex_quality.get)
            best_score = hex_quality[best_level]

            if best_score >= 3:
                recommendations.append(f"→ USE {best_level.upper()}")
                recommendations.append(f"  Reason: Good model fit, reliable spatial clustering")
            elif best_score >= 2:
                recommendations.append(f"→ USE {best_level.upper()} WITH CAUTION")
                recommendations.append(f"  Reason: Acceptable fit, but verify clusters manually")
            else:
                # All resolutions poor - recommend coarser resolution
                if best_level == 'Hex_Res4':
                    recommendations.append(f"→ RES4 FAILED - TRY RES3 (COARSER)")
                    recommendations.append(f"  Reason: Insufficient data at this resolution")
                else:
                    recommendations.append(f"→ ALL RESOLUTIONS SHOW POOR FIT")
                    recommendations.append(f"  Reason: Consider using administrative units instead")

            # Specific recommendations for failed resolutions
            for level in hex_order:
                if level in hex_quality and hex_quality[level] < 2:
                    if level == 'Hex_Res4' and 'Hex_Res3' in hex_quality:
                        recommendations.append(f"  [WARN]  Res4 unreliable → Use Res3 instead")

        recommendations.append("")

    recommendations.append("=" * 80)
    recommendations.append("END OF REPORT")
    recommendations.append("=" * 80)

    report_text = '\n'.join(recommendations)
    logger.info(report_text)

    with open(recommendations_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    logger.info(f"[OK] Recommendations saved: {recommendations_path}")


