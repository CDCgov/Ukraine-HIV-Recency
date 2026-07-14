# -*- coding: utf-8 -*-
"""
Supplementary risk-group attribution report (oblast level).

Spatial-joins the case table onto oblast boundaries and runs the four
:mod:`pipeline.standardization.group_attribution` products: the tested case-mix
shift over time, the escalating-window power table for the national group rates,
the per-oblast high-risk vs general recency rates (with a two-proportion test),
and the per-oblast indirect standardization. Prints full tables (all oblasts) and
writes them to an Excel workbook.

Run:
    python validation/group_attribution_report.py [config.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.aggregation.periods import baseline_months_for
from pipeline.standardization import (
    escalating_group_rates,
    oblast_group_rates,
    oblast_standardization,
    period_composition,
)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / 'config.json')
    cfg = json.load(open(cfg_path, encoding='utf-8'))

    cases = pd.read_excel(ROOT / cfg['excel_path'], sheet_name='hiv_cases')
    cases['test_date'] = pd.to_datetime(cases['test_date'])

    cur_s = pd.Timestamp(cfg['analysis_period']['start'])
    cur_e = pd.Timestamp(cfg['analysis_period']['end'])

    gob = gpd.read_file(ROOT / cfg['administrative_units']['adm1_path'])
    obcol = cfg['administrative_units']['oblast_col']
    pts = gpd.GeoDataFrame(cases, geometry=gpd.points_from_xy(cases['longitude'], cases['latitude']),
                           crs='EPSG:4326')
    joined = gpd.sjoin(pts.to_crs(gob.crs), gob[[obcol, 'geometry']], how='inner', predicate='within')

    pd.set_option('display.width', 240)
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.max_rows', 60)

    comp = period_composition(cases, split_year=2024)
    esc = escalating_group_rates(cases, cur_s)
    rates = oblast_group_rates(joined, obcol)
    std = oblast_standardization(joined, obcol, cur_s, cur_e)

    print("=" * 92)
    print("(1) TESTED CASE-MIX SHIFT OVER TIME (well powered)")
    print("=" * 92)
    print(comp['by_year'].to_string(index=False))
    print(f"\nyear x group chi-square: chi2={comp['chi2']:.1f}, p={comp['chi2_p']:.2e}")
    print(f"monotonic trend in high-share: r={comp['trend_r']:.3f}, p={comp['trend_p']:.4f}")
    print(f"pre-{comp['split_year']}: {comp['pre_pct']:.1f}% "
          f"[{comp['pre_ci'][0]:.1f},{comp['pre_ci'][1]:.1f}] (N={comp['pre_n']})   "
          f"{comp['split_year']}+: {comp['post_pct']:.1f}% "
          f"[{comp['post_ci'][0]:.1f},{comp['post_ci'][1]:.1f}] (N={comp['post_n']})   "
          f"z={comp['contrast_z']:.1f}, p={comp['contrast_p']:.2e}")

    print("\n" + "=" * 92)
    print("(2) NATIONAL GROUP RATES at escalating windows (adequate = >=10 recent in each group)")
    print("=" * 92)
    print(esc.to_string(index=False))

    print("\n" + "=" * 92)
    print("(3) PER-OBLAST high-risk vs general recency rate (whole pool; 'powered' = conclusive)")
    print("=" * 92)
    print(rates.to_string(index=False))
    pw = rates[rates['powered']]
    print(f"\nPowered oblasts with a significant group difference (p<0.05): "
          f"{', '.join(pw[pw['p_diff'] < 0.05]['oblast'].tolist()) or 'none'}")

    print("\n" + "=" * 92)
    print("(4a) PER-OBLAST INDIRECT STANDARDIZATION (current window -- cross-checks the detector)")
    print("comp_ratio = mix effect (reliable); smr = excess beyond mix. Current window is sparse.")
    print("=" * 92)
    print(std.to_string(index=False))

    # Whole-pool standardization: where the SMR is actually powered.
    pool_s = pd.Timestamp('2021-01-01')
    std_pool = oblast_standardization(joined, obcol, pool_s, cur_e)
    print("\n" + "=" * 92)
    print("(4b) PER-OBLAST INDIRECT STANDARDIZATION (whole pool 2021-2026 -- powered)")
    print("=" * 92)
    print(std_pool.to_string(index=False))

    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    xlsx = out / 'Group_Attribution_Report.xlsx'
    with pd.ExcelWriter(xlsx) as w:
        comp['by_year'].to_excel(w, sheet_name='composition_by_year', index=False)
        esc.to_excel(w, sheet_name='escalating_group_rates', index=False)
        rates.to_excel(w, sheet_name='oblast_group_rates', index=False)
        std.to_excel(w, sheet_name='oblast_std_current', index=False)
        std_pool.to_excel(w, sheet_name='oblast_std_pool', index=False)
    print(f"\n[OK] Written: {xlsx}")


if __name__ == '__main__':
    main()
