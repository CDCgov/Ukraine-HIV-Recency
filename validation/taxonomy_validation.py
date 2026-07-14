# -*- coding: utf-8 -*-
"""
Taxonomy coverage test: does the pipeline recover every classification label?

For each taxonomy category (the seven SMR/SIR labels, the trend-uncertain label,
No Data, and the two watch-list reasons) this builds a SEPARATE synthetic
dataset: a fixed low-rate background of normal hexes plus ONE engineered target
hex designed to land in that category. Each dataset is run through the full
pipeline with the production Joint Two-Period Beta-Binomial model, and the
target hex's classification and watch-list status are read back. The result is
a table showing how the detector catches each engineered signal.

Separate runs (one strong target per run) keep the national rate and the
FDR-controlled cut-offs from being distorted by other extreme hexes, so each
category is tested cleanly.

Run:
    python validation/taxonomy_validation.py <scratch_dir>
Outputs a table to stdout and writes taxonomy_results.csv into the scratch dir.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_taxo_scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)

HIST0, HIST1 = pd.Timestamp('2025-03-15'), pd.Timestamp('2026-02-15')
CUR0, CUR1 = pd.Timestamp('2026-03-15'), pd.Timestamp('2026-05-15')
PERIOD = '202603'

# (key, intended label, n_hist, r_hist, n_curr, r_curr, high_frac_curr, high_frac_hist)
DESIGNS = [
    ('established', 'Established hotspot',                     250, 8,  250, 45, 0.40, 0.40),
    ('emerging',    'Emerging hotspot',                       250, 2,  250, 9,  0.35, 0.30),
    ('stable',      'Stable high-burden',                     250, 45, 250, 45, 0.40, 0.40),
    ('declining',   'Declining from high-burden',             250, 60, 250, 30, 0.40, 0.40),
    ('emerg_dec',   'Emerging decrease',                      250, 20, 250, 6,  0.30, 0.40),
    ('sig_dec',     'Significant decrease',                   250, 8,  300, 1,  0.20, 0.30),
    ('normal',      'Normal',                                 250, 6,  250, 7,  0.25, 0.25),
    ('elev_unc',    'Elevated vs national (trend uncertain)', 4,   0,  250, 40, 0.40, 0.25),
    ('nodata',      'No Data',                                250, 6,  0,   0,  0.25, 0.25),
    ('watch_burden','Watch: high burden (avg rate)',          800, 24, 800, 24, 0.30, 0.30),
    ('watch_rate',  'Watch: high rate (few events)',          12,  1,  12,  4,  0.50, 0.40),
    ('composition', 'Testing composition shift',              250, 10, 250, 26, 0.60, 0.20),
]


REAL = ROOT / 'data' / 'synthetic_input_data.xlsx'


def _load_universe():
    gj = gpd.read_file(ROOT / 'data' / 'h3_hexagons_res3.geojson')
    idcol = 'h3_id' if 'h3_id' in gj.columns else gj.columns[0]
    return sorted(str(x) for x in gj[idcol].tolist())


def _real_background():
    """The real synthetic dataset provides a HETEROGENEOUS background.

    A homogeneous background collapses the hierarchical between-hex variance to
    zero, and then even a strong injected hotspot is shrunk to the common mean
    (it lands in Normal). Real data carries natural variation, so an injected
    outlier can actually express.
    """
    cases = pd.read_excel(REAL, sheet_name='hiv_cases')
    sites = pd.read_excel(REAL, sheet_name='testing_sites')
    cases['test_date'] = pd.to_datetime(cases['test_date'])
    used = set(h3.latlng_to_cell(la, lo, 3)
               for la, lo in zip(cases['latitude'], cases['longitude']))
    return cases, sites, used


UNIVERSE = _load_universe()
_BG_CASES, _BG_SITES, _USED = _real_background()
# one reserved empty cell per category (distinct from the real data's cells)
TARGET_HEXES = [h for h in UNIVERSE if h not in _USED][:len(DESIGNS)]


def _dates(rng, n, a, b):
    if n <= 0:
        return []
    return [a + pd.Timedelta(days=int(x)) for x in rng.integers(0, (b - a).days + 1, size=n)]


def _emit(rows, sites, rng, cid, hx, sid, nh, rh, nc, rc, hic, hih):
    lat, lon = h3.cell_to_latlng(hx)
    sites.append(dict(site_id=sid, longitude=lon, latitude=lat,
                      activation_date=pd.Timestamp('2025-01-01'),
                      deactivation_date=pd.Timestamp('2026-12-31')))
    for i, d in enumerate(_dates(rng, nh, HIST0, HIST1)):
        rows.append(dict(case_id=cid, type=('recent' if i < rh else 'long-term'), test_date=d,
                         longitude=lon, latitude=lat,
                         risk_group=('high' if rng.random() < hih else 'low'), site_id=sid)); cid += 1
    for i, d in enumerate(_dates(rng, nc, CUR0, CUR1)):
        rows.append(dict(case_id=cid, type=('recent' if i < rc else 'long-term'), test_date=d,
                         longitude=lon, latitude=lat,
                         risk_group=('high' if rng.random() < hic else 'low'), site_id=sid)); cid += 1
    return cid


def build_dataset(design, target_hex, path):
    """One dataset: the real (heterogeneous) background + a single target hex."""
    rng = np.random.default_rng(2026)
    rows, sites, cid = [], [], 20_000_000
    _, _, nh, rh, nc, rc, hic, hih = design
    _emit(rows, sites, rng, cid, target_hex, f'T_{design[0]}', nh, rh, nc, rc, hic, hih)
    allc = pd.concat([_BG_CASES, pd.DataFrame(rows)], ignore_index=True)
    # unify case_id dtype (real ids are strings, injected ids are ints) so the
    # pipeline's parquet cache can serialise the column.
    allc['case_id'] = allc['case_id'].astype(str)
    alls = pd.concat([_BG_SITES, pd.DataFrame(sites)], ignore_index=True)
    with pd.ExcelWriter(path) as w:
        allc.to_excel(w, sheet_name='hiv_cases', index=False)
        alls.to_excel(w, sheet_name='testing_sites', index=False)


def run_pipeline(xlsx):
    """Run the pipeline crude-only (two-period model); return (report_df, outdir) for the target."""
    cfg = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    cfg['excel_path'] = str(xlsx)
    cfg['analysis_type'] = 'standard'
    cfg['analysis_levels'] = [3]
    cfg['two_period_model'] = True
    # leave-one-out national rate: the single strong target must not inflate its
    # own comparison denominator, or its category would blur.
    cfg['smr_leave_one_out'] = True
    # single-core sampling: multiprocessing chains are unreliable in a very long
    # session (children get killed); one core is slower but robust.
    cfg.setdefault('sampling', {})['cores'] = 1
    # fast preset (2 chains x 500 draws) -- enough for the classification label,
    # and ~4x faster than the full posterior, which matters for 24 runs.
    cfg['fast_sampling'] = True
    cfgp = SCRATCH / 'cfg_two_period.json'
    cfgp.write_text(json.dumps(cfg), encoding='utf-8')

    before = set((ROOT / 'output').glob('*')) if (ROOT / 'output').exists() else set()
    subprocess.run([sys.executable, str(ROOT / 'validation' / 'service_run.py'), str(cfgp)],
                   cwd=str(ROOT), capture_output=True, text=True)
    after = set((ROOT / 'output').glob('*'))
    new_dirs = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if not new_dirs:
        return 'RUN-FAILED', '-'
    outdir = new_dirs[-1]
    report = outdir / 'bayesian' / 'hex' / 'res3' / f'Report_Hex_Res3_{PERIOD}.xlsx'
    if not report.exists():
        return 'NO-REPORT', outdir
    try:
        df = pd.read_excel(report, sheet_name='Data')
        return df, outdir  # caller matches the target hex
    except Exception as exc:  # noqa: BLE001 -- keep the batch going on a bad file
        print(f'   [warn] could not read report: {exc}')
        return 'READ-FAILED', outdir


def main():
    # Resumable: reuse any category already recorded (the environment kills long
    # runs, so this is driven in short chunks that each append a few categories).
    out_csv = SCRATCH / 'taxonomy_results.csv'
    results = []
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        results = prev.to_dict('records')
        done = set(prev['intended'])
        print(f"resuming: {len(done)} categories already done", flush=True)
    limit = int(os.environ.get('TAXO_CHUNK', '99'))  # max NEW categories this run
    new_count = 0
    for design, thex in zip(DESIGNS, TARGET_HEXES):
        key, intended = design[0], design[1]
        if intended in done:
            continue
        if new_count >= limit:
            break
        new_count += 1
        xlsx = SCRATCH / f'ds_{key}.xlsx'
        build_dataset(design, thex, xlsx)
        row = {'intended': intended, 'target_hex': thex[:8]}
        out = run_pipeline(xlsx)
        if isinstance(out, tuple) and isinstance(out[0], pd.DataFrame):
            df, outdir = out
            m = df[df['h3_id'].astype(str) == thex]
            if len(m):
                r = m.iloc[0]
                cls = str(r.get('Classification', r.get('classification', '?')))
                onwl = bool(r.get('on_watchlist', False))
                wr = str(r.get('watch_reason', '') or '')
                row['detected'] = cls
                row['watch'] = (wr if onwl and wr else ('yes' if onwl else '-'))
            else:
                row['detected'] = 'HEX-NOT-FOUND'; row['watch'] = '-'
            shutil.rmtree(outdir, ignore_errors=True)
        else:
            row['detected'] = 'RUN-FAILED'; row['watch'] = '-'
        results.append(row)
        # write incrementally so a long run is never lost and progress is visible
        pd.DataFrame(results).to_csv(SCRATCH / 'taxonomy_results.csv', index=False)
        print(f"[done] {intended:42s} detected={row['detected']}", flush=True)

    res = pd.DataFrame(results)
    res.to_csv(SCRATCH / 'taxonomy_results.csv', index=False)
    print("\n" + "=" * 120)
    print("TAXONOMY COVERAGE — engineered target per category, two-period model (crude)")
    print("=" * 120)
    print(res.to_string(index=False))


if __name__ == '__main__':
    main()
