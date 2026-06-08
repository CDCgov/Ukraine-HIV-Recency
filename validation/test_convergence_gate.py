"""Convergence-gate smoke test — verify save_report / plot_map gate on convergence_fatal.

This is not a unit-test framework test, just a standalone smoke check
that exercises the ``convergence_fatal`` branches that the regular
``--test`` never reaches (the test data converges cleanly). It builds a
minimal analyzer, injects ``convergence_fatal=True`` and confirms both
safety behaviours actually fire:

    1. ``save_report`` writes ``Report_*_SKIPPED.txt`` and skips the Excel.
    2. ``plot_map`` writes ``Map_*_EN_UNRELIABLE.png`` (oblast outlines +
       WARN banner) and skips the colour-filled choropleth.

The boundary-only fallback map loads the oblast layer for context, so the
test stubs ``load_geodata`` with a tiny synthetic oblast frame -- that keeps
the check hermetic (no dependency on the real GeoJSON) while still driving
the real rendering code. The scenario uses a hex level (the only analysis
mode the pipeline supports), as a production run would.

Exit code is 0 only if both artefacts are produced; non-zero otherwise.

Run via ``python validation/test_convergence_gate.py``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Project root on sys.path so this script runs from the validation/ subfolder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

import run_hotspots as m


def _fake_gdf() -> gpd.GeoDataFrame:
    """Two-territory result frame with the columns save_report / plot_map read."""
    geom = [Point(0, 0).buffer(0.1), Point(1, 1).buffer(0.1)]
    return gpd.GeoDataFrame(
        {
            'h3_id': ['841e5ddffffffff', '8411965ffffffff'],
            'all_tested_curr': [100, 200],
            'recent_count_curr': [3, 5],
            'recent_proportion_curr': [0.03, 0.025],
            'all_tested_hist': [80, 180],
            'recent_count_hist': [2, 4],
            'recent_proportion_hist': [0.025, 0.022],
            'classification': ['Normal', 'Normal'],
            'predicted': [0.03, 0.025],
            'residual': [0.0, 0.0],
            'reliability_score': [80.0, 85.0],
            'reliability_category': ['HIGH', 'HIGH'],
            'reliability_flag': ['[OK]', '[OK]'],
        },
        geometry=geom,
        crs="EPSG:4326",
    )


def _fake_oblast_gdf() -> gpd.GeoDataFrame:
    """Minimal oblast layer for the boundary-only UNRELIABLE map.

    ``add_oblast_labels`` only needs an ``ADM1_EN`` column and geometry with
    a CRS, so two labelled polygons are enough to render the fallback map.
    """
    geom = [Point(0, 0).buffer(0.5), Point(2, 2).buffer(0.5)]
    return gpd.GeoDataFrame(
        {'ADM1_EN': ['Kyiv', 'Lviv']},
        geometry=geom,
        crs="EPSG:4326",
    )


def main() -> None:
    out_root = Path('_convergence_smoke_out')
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir()

    cfg = dict(m.DEFAULT_CONFIG)
    cfg['administrative_units'] = {'oblast_col': 'ADM1_EN'}
    cfg['output_dir'] = str(out_root)

    analyzer = m.BayesianAnalyzer.__new__(m.BayesianAnalyzer)
    analyzer.cfg = cfg
    analyzer.config = cfg
    analyzer.run_timestamp = '00000000'
    analyzer.diagnostics = None
    analyzer.mode_suffix = 'hex'
    analyzer._cached_geodata = {}
    # The boundary-only fallback map calls load_geodata('Oblast'); stub it with
    # a synthetic oblast frame so the test does not depend on the real GeoJSON.
    analyzer.load_geodata = lambda *args, **kwargs: _fake_oblast_gdf()

    def fake_get_output_path(*args, is_hex=False, **kwargs):
        flat = [a for a in args if isinstance(a, str)]
        fname = flat[-1]
        path = out_root / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    analyzer.get_output_path = fake_get_output_path

    fatal_diag = {
        'convergence_fatal': True,
        'pct_divergences': 12.5,
        'rhat_max': 1.18,
        'ess_alpha_min': 42,
        'convergence_ok': 'No',
    }

    level = 'Hex_Res4'
    period_str = '202602'
    gdf = _fake_gdf()
    start = pd.Timestamp('2026-02-01')
    end = pd.Timestamp('2026-04-30')

    print('Calling save_report with fatal diagnostics ...')
    analyzer.save_report(gdf, level, period_str, diagnostics=fatal_diag)

    print('Calling plot_map with fatal diagnostics ...')
    analyzer.plot_map(gdf, level, start, end, model_name='Bayesian',
                      diagnostics=fatal_diag)

    print()
    print('Outputs:')
    for path in sorted(out_root.glob('*')):
        print(f'  {path.name}  ({path.stat().st_size} bytes)')

    # Assertions: both safety artefacts must exist.
    sentinel = out_root / f'Report_{level}_{period_str}_SKIPPED.txt'
    unreliable_map = out_root / f'Map_{level}_{period_str}_EN_UNRELIABLE.png'

    failures = []
    if sentinel.exists():
        print()
        print('--- sentinel contents ---')
        print(sentinel.read_text(encoding='utf-8'))
    else:
        failures.append(f'missing sentinel: {sentinel.name}')

    excel = out_root / f'Report_{level}_{period_str}.xlsx'
    if excel.exists():
        failures.append(f'Excel report should have been skipped: {excel.name}')

    if not unreliable_map.exists():
        failures.append(f'missing UNRELIABLE map: {unreliable_map.name}')

    print()
    if failures:
        for f in failures:
            print(f'FAIL: {f}')
        sys.exit(1)
    print('PASS: convergence-fatal gate skipped the Excel report and drew the '
          'UNRELIABLE map.')


if __name__ == '__main__':
    main()
