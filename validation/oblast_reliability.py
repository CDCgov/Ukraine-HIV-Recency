# -*- coding: utf-8 -*-
"""
Oblast-level reliability illustration (offline, NOT the production pipeline).

The pipeline is hexagon-only; it does not analyse oblasts as units. This
standalone script answers a single what-if for the paper/discussion: *if we
aggregated the same recency data to the oblast level, which oblasts would
reach a usable reliability tier?* The expectation is that the high-volume
regions (Dnipro / Odesa / Kyiv, which hold ~40-50% of all recency tests)
clear LOW while the sparse oblasts stay LOW -- i.e. reliability is bounded by
how many recent infections accumulate per unit, not by the geographic split.

Method (closed-form, the same shrinkage + CV-based reliability the pipeline
uses): assign each test to its oblast by point-in-polygon, aggregate
recent/tested over the analysis window, form a Beta posterior centred on the
national rate (concentration K), and score reliability as
``100 * exp(-CV)`` of the SMR posterior with the pipeline's 80/60 tiers.
This is an illustration; the production reliability uses the EB-fitted K and
the full MCMC posterior, but the CV behaviour is the same.

Run:
    python validation/oblast_reliability.py
    python validation/oblast_reliability.py config.json --prior-k 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

# Project root on sys.path so this script runs from the validation/ subfolder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.constants import DEFAULT_CONFIG
from pipeline.io.cases import load_cases_from_disk


def main() -> None:
    parser = argparse.ArgumentParser(description="Oblast-level reliability illustration")
    parser.add_argument('config', nargs='?', default='config.json',
                        help='Config JSON (default: config.json)')
    parser.add_argument('--prior-k', type=float, default=20.0,
                        help='EB prior concentration K (default 20, illustrative)')
    args = parser.parse_args()

    if Path(args.config).exists():
        config = json.load(open(args.config, encoding='utf-8'))
    else:
        config = dict(DEFAULT_CONFIG)

    target_crs = config.get('target_crs', 'EPSG:3857')
    adm1_path = config['administrative_units']['adm1_path']
    period = config['analysis_period']
    start = pd.to_datetime(period['start'])
    end = pd.to_datetime(period['end'])

    print(f"Analysis window: {start.date()} to {end.date()}  |  prior K={args.prior_k}")
    print("Loading cases and oblast boundaries ...")
    gdf_cases = load_cases_from_disk(config['excel_path'], target_crs)
    gdf_obl = gpd.read_file(adm1_path).to_crs(target_crs)
    name_col = 'ADM1_EN' if 'ADM1_EN' in gdf_obl.columns else gdf_obl.columns[0]

    # Window filter + point-in-oblast assignment.
    win = gdf_cases[(gdf_cases['test_date'] >= start) & (gdf_cases['test_date'] <= end)].copy()
    joined = gpd.sjoin(win, gdf_obl[[name_col, 'geometry']], predicate='within', how='inner')
    joined['is_recent'] = (joined['type'].astype(str).str.lower() == 'recent').astype(int)

    agg = joined.groupby(name_col).agg(tested=('is_recent', 'size'),
                                       recent=('is_recent', 'sum')).reset_index()
    tested_total = float(agg['tested'].sum())
    recent_total = float(agg['recent'].sum())
    p0 = max(recent_total / tested_total, 1e-3) if tested_total > 0 else 1e-3

    K = args.prior_k
    a0, b0 = p0 * K, (1.0 - p0) * K

    rows = []
    for _, r in agg.iterrows():
        tested, recent = float(r['tested']), float(r['recent'])
        a, b = a0 + recent, b0 + (tested - recent)
        p_mean = a / (a + b)
        p_lo = beta_dist.ppf(0.025, a, b)
        p_hi = beta_dist.ppf(0.975, a, b)
        cv = (p_hi - p_lo) / (3.92 * p_mean) if p_mean > 0 else float('nan')
        score = 100.0 * float(np.exp(-cv)) if np.isfinite(cv) else float('nan')
        tier = "HIGH" if score >= 80 else ("MODERATE" if score >= 60 else "LOW")
        rows.append((r[name_col], int(tested), int(recent), p_mean / p0, score, tier))

    rows.sort(key=lambda x: x[1], reverse=True)

    print("=" * 86)
    print(f"Oblast-level reliability (window {start.date()}..{end.date()})  "
          f"national recent% = {100*p0:.2f}")
    print("=" * 86)
    print(f"{'oblast':<26}{'tested':>8}{'recent':>8}{'SMR':>8}{'reliability':>13}{'tier':>10}")
    print("-" * 86)
    for name, tested, recent, smr, score, tier in rows:
        sc = f"{score:.1f}" if np.isfinite(score) else "n/a"
        print(f"{str(name)[:25]:<26}{tested:>8}{recent:>8}{smr:>8.2f}{sc:>13}{tier:>10}")
    print("-" * 86)
    n_mod = sum(1 for r in rows if r[5] in ('MODERATE', 'HIGH'))
    print(f"Oblasts reaching MODERATE+ : {n_mod}/{len(rows)}")
    print(f"Total: {int(tested_total)} tests, {int(recent_total)} recent across {len(rows)} oblasts")
    print("\nReading it: reliability tracks the per-oblast test volume; the few "
          "high-volume oblasts\ncan clear LOW, the sparse ones cannot -- the binding "
          "constraint is the number of\nrecent infections that accumulate per unit, "
          "not the geographic resolution.")


if __name__ == '__main__':
    main()
