"""Precompute the data files the JupyterLite notebooks read.

    python jupyterlite/build_content.py            # run the pipeline, then export
    python jupyterlite/build_content.py --export   # export only (reuse earlier runs)

Everything here runs on the PUBLIC synthetic dataset
(``data/synthetic_input_data.xlsx``); no real record is read at any step. The
production pipeline is run, unchanged, on

* the standard levels (H3 res-3 hexagons and oblasts), and
* each of the eight DGGS grids in ``jupyterlite/grids/`` (level 3 only), the
  grid-sensitivity (MAUP) check,

for one analysis window, and the results are exported as small GeoJSON/CSV
files into ``jupyterlite/content/data/``. The pipeline runs go to
``jupyterlite/_runs/`` (git-ignored).
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LITE = REPO / 'jupyterlite'
GRIDS = LITE / 'grids'
RUNS = LITE / '_runs'
DATA = LITE / 'content' / 'data'
SYNTHETIC = REPO / 'data' / 'synthetic_input_data.xlsx'

WINDOW = ('2025-12-01', '2026-05-31')
GRID_SLUGS = ['h3', 's2', 'a5', 'rhealpix', 'healpix_xdggs', 'dggrid_isea4h', 'dggrid_isea3h',
              'openeaggr_isea4t']
REPORT_COLS = {
    'h3_id': 'cell_id', 'hex_name_en': 'name', 'oblast_en': 'oblast',
    'all_tested_curr': 'tests_curr', 'recent_count_curr': 'recent_curr',
    'all_tested_hist': 'tests_hist', 'recent_count_hist': 'recent_hist',
    'testing_intensity_curr': 'test_months_curr', 'testing_intensity_hist': 'test_months_hist',
    'Classification': 'label', 'reliability_category': 'reliability', 'reliability_score': 'reliability_score',
    'National reference ratio (median)': 'ratio_median', 'National reference ratio (2.5%)': 'ratio_lo',
    'National reference ratio (97.5%)': 'ratio_hi', 'on_watchlist': 'on_watchlist',
}


def run_one(slug: str) -> None:
    """Run the production pipeline once ('production' = H3 res3 + oblast; else a DGGS grid)."""
    import os
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import pipeline.bootstrap  # noqa: F401
    from pipeline.logging_setup import setup_logging
    from pipeline.orchestrator import PipelineOrchestrator

    out = RUNS / slug
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(log_to_stdout=False, log_to_file=True, log_file=out / 'pipeline.log', log_level='INFO')
    orch = PipelineOrchestrator(None, run_timestamp=datetime.now().strftime('%Y%m%d%H%M%S'), output_base=out)
    cfg = orch.config
    cfg['excel_path'] = str(SYNTHETIC)
    cfg['analysis_period'] = {'start': WINDOW[0], 'end': WINDOW[1]}
    cfg.setdefault('sampling', {})['cores'] = 1
    if slug == 'production':
        cfg.update(analysis_levels=[3, 'Oblast'], hex_resolutions=[3], admin_levels=['Oblast'],
                   analysis_mode='h3_hexagons')
    else:
        cfg['h3_hexagons']['res3_path'] = str(GRIDS / f'{slug}.geojson')
        cfg['analysis_levels'] = [3]
    orch.run_full_pipeline()


def run_all() -> None:
    """Run every configuration in its own process, a few at a time."""
    def launch(slug):
        r = subprocess.run([sys.executable, __file__, '--run', slug], capture_output=True, text=True)
        print(f"{'DONE' if r.returncode == 0 else 'FAILED'} {slug}", flush=True)
        if r.returncode != 0:
            print(r.stderr[-2000:])
        return r.returncode
    with ThreadPoolExecutor(max_workers=5) as ex:
        codes = list(ex.map(launch, ['production'] + GRID_SLUGS))
    if any(codes):
        sys.exit("Some pipeline runs failed; see jupyterlite/_runs/<run>/pipeline.log")


def _report(slug: str, level: str):
    import pandas as pd
    sub = 'admin/Oblast' if level == 'oblast' else 'hex/res3'
    path = next((RUNS / slug).glob(f'*/bayesian/{sub}/Report_*.xlsx'), None) or \
        next((RUNS / slug).glob(f'bayesian/{sub}/Report_*.xlsx'))
    df = pd.read_excel(path, dtype={'h3_id': str})
    return df


def _round_geometry(gdf, ndigits=4):
    from shapely import set_precision
    gdf = gdf.copy()
    gdf['geometry'] = set_precision(gdf.geometry.values, 10 ** -ndigits)
    return gdf


def export() -> None:
    import geopandas as gpd
    import pandas as pd

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / 'grids').mkdir(exist_ok=True)

    # Synthetic inputs (already public) and the oblast boundaries for maps.
    cases = pd.read_excel(SYNTHETIC, sheet_name='hiv_cases', parse_dates=['test_date'])
    sites = pd.read_excel(SYNTHETIC, sheet_name='testing_sites')
    sites.to_csv(DATA / 'synthetic_sites.csv', index=False)
    cases[['case_id', 'type', 'test_date', 'risk_group', 'site_id']].to_csv(
        DATA / 'synthetic_cases.csv', index=False)
    oblasts = gpd.read_file(REPO / 'data' / 'Ukraine_Adm1_Oblast.geojson')[['ADM1_EN', 'geometry']]
    oblasts = oblasts.rename(columns={'ADM1_EN': 'oblast'})
    oblasts['geometry'] = oblasts.simplify(0.01)
    _round_geometry(oblasts).to_file(DATA / 'oblasts.geojson', driver='GeoJSON')

    # Production results: H3 res3 cells and oblasts.
    h3 = gpd.read_file(GRIDS / 'h3.geojson')[['h3_id', 'geometry']]
    h3['h3_id'] = h3['h3_id'].astype(str)
    rep = _report('production', 'hex')
    keep = [c for c in REPORT_COLS if c in rep.columns]
    res = h3.merge(rep[keep], on='h3_id', how='left').rename(columns=REPORT_COLS)
    _round_geometry(res).to_file(DATA / 'results_h3.geojson', driver='GeoJSON')
    orep = _report('production', 'oblast')
    okeep = [c for c in REPORT_COLS if c in orep.columns and c != 'h3_id']
    ores = orep[okeep].rename(columns=REPORT_COLS)
    name_col = next(c for c in orep.columns if c.lower() in ('territory_name', 'adm1_en', 'oblast_en', 'name'))
    ores.insert(0, 'oblast_name', orep[name_col])
    ores.to_csv(DATA / 'results_oblast.csv', index=False)

    # Grid-sensitivity results: every cell of every grid with its label.
    rows = []
    for slug in GRID_SLUGS:
        grid = gpd.read_file(GRIDS / f'{slug}.geojson')
        grid['h3_id'] = grid['h3_id'].astype(str)
        _round_geometry(grid[['h3_id', 'geometry']].rename(columns={'h3_id': 'cell_id'})).to_file(
            DATA / 'grids' / f'{slug}.geojson', driver='GeoJSON')
        r = _report(slug, 'hex')
        r = r[[c for c in REPORT_COLS if c in r.columns]].rename(columns=REPORT_COLS)
        r.insert(0, 'grid', slug)
        rows.append(r)
    pd.concat(rows).drop(columns=['name', 'oblast'], errors='ignore').to_csv(
        DATA / 'grid_results.csv', index=False)

    # Grid descriptions without anything derived from site locations.
    summary = pd.read_csv(GRIDS / 'grids_summary.csv')
    summary[['grid', 'slug', 'resolution', 'n_cells', 'area_median_km2', 'area_min_km2', 'area_max_km2']].to_csv(
        DATA / 'grids_summary.csv', index=False)

    make_jgis(res, oblasts, sites)

    meta = {'window_start': WINDOW[0], 'window_end': WINDOW[1],
            'source': 'data/synthetic_input_data.xlsx (fully synthetic)',
            'built': datetime.now().strftime('%Y-%m-%d')}
    (DATA / 'build_info.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f"Exported to {DATA}")


LABEL_COLORS = {
    'Established hotspot': '#d62728', 'Emerging hotspot': '#fb8500', 'Stable high-burden': '#ffd60a',
    'Elevated vs national (trend uncertain)': '#c77dff', 'Declining from high-burden': '#48cae4',
    'Emerging decrease': '#a0e0a8', 'Below national': '#06d6a0', 'Normal': '#d3d3d3',
}


def _rgba(hex_color: str, alpha: float = 1.0) -> list:
    h = hex_color.lstrip('#')
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [alpha]


def make_jgis(cells, oblasts, sites) -> None:
    """Write ``hotspots_map.jGIS``: a JupyterGIS project with the data embedded.

    One vector layer per label (so each has its own colour and can be toggled),
    an outline layer marking LOW-reliability cells, oblast borders, the
    synthetic sites and an OpenStreetMap basemap.
    """
    import uuid
    import geopandas as gpd

    sources, layers, tree = {}, {}, []

    def add(name, data, symbology, visible=True, opacity=1.0):
        sid, lid = str(uuid.uuid4()), str(uuid.uuid4())
        sources[sid] = {'name': f'{name} source', 'type': 'GeoJSONSource', 'parameters': {'data': data}}
        layers[lid] = {'name': name, 'type': 'VectorLayer', 'visible': visible,
                       'parameters': {'source': sid, 'opacity': opacity, 'symbologyState': symbology}}
        tree.append(lid)

    osm_s, osm_l = str(uuid.uuid4()), str(uuid.uuid4())
    sources[osm_s] = {'name': 'OpenStreetMap source', 'type': 'RasterSource', 'parameters': {
        'url': 'https://tile.openstreetmap.org/{z}/{x}/{y}.png', 'maxZoom': 19, 'minZoom': 0,
        'attribution': '(C) OpenStreetMap contributors', 'htmlAttribution': '', 'provider': 'OpenStreetMap',
        'bounds': [], 'urlParameters': {}}}
    layers[osm_l] = {'name': 'OpenStreetMap', 'type': 'RasterLayer', 'visible': True,
                     'parameters': {'source': osm_s, 'opacity': 1}}
    tree.append(osm_l)

    active = cells[cells['label'].notna()]
    props = ['cell_id', 'name', 'oblast', 'label', 'reliability', 'tests_curr', 'recent_curr',
             'tests_hist', 'recent_hist', 'ratio_median', 'ratio_lo', 'ratio_hi']
    for label, color in LABEL_COLORS.items():
        sub = active[active['label'] == label]
        if sub.empty:
            continue
        add(label, json.loads(sub[[c for c in props if c in sub.columns] + ['geometry']].round(3).to_json()),
            {'fillColor': _rgba(color, 0.75), 'strokeColor': [80, 80, 80, 1], 'strokeWidth': 0.6})
    low = active[active['reliability'] == 'LOW']
    if not low.empty:
        add('Low reliability (outline)', json.loads(low[['cell_id', 'geometry']].to_json()),
            {'fillColor': [0, 0, 0, 0], 'strokeColor': [20, 20, 20, 1], 'strokeWidth': 2.2})
    add('Oblast borders', json.loads(oblasts.to_json()),
        {'fillColor': [0, 0, 0, 0], 'strokeColor': [110, 110, 110, 1], 'strokeWidth': 0.8})
    pts = gpd.GeoDataFrame(sites[['site_id']], geometry=gpd.points_from_xy(sites.longitude, sites.latitude),
                           crs='EPSG:4326')
    add('Synthetic testing sites', json.loads(pts.to_json()),
        {'fillColor': [0, 0, 0, 1], 'strokeColor': [255, 255, 255, 1], 'strokeWidth': 0.5, 'radius': 3})

    doc = {
        'schemaVersion': '0.7.2', 'layers': layers, 'sources': sources, 'layerTree': tree,
        'options': {'latitude': 48.6, 'longitude': 31.3, 'zoom': 5.4, 'bearing': 0, 'pitch': 0,
                    'projection': 'EPSG:3857'},
        'metadata': {'title': 'HIV recency hotspots (synthetic data)'},
    }
    try:  # validate against the JupyterGIS schema when the package is available
        from jupytergis_core.schema.interfaces.project.jgis import IJGISContent
        from jupytergis_core.schema.interfaces.project.layers.rasterLayer import IRasterLayer
        from jupytergis_core.schema.interfaces.project.layers.vectorLayer import IVectorLayer
        from jupytergis_core.schema.interfaces.project.sources.geoJsonSource import IGeoJSONSource
        from jupytergis_core.schema.interfaces.project.sources.rasterSource import IRasterSource
    except ImportError:
        print('jupytergis not installed: hotspots_map.jGIS written without schema validation')
    else:
        IJGISContent.model_validate(doc)
        for s in sources.values():
            {'GeoJSONSource': IGeoJSONSource, 'RasterSource': IRasterSource}[s['type']].model_validate(
                s['parameters'])
        for lyr in layers.values():
            {'VectorLayer': IVectorLayer, 'RasterLayer': IRasterLayer}[lyr['type']].model_validate(
                lyr['parameters'])
    (LITE / 'content' / 'hotspots_map.jGIS').write_text(json.dumps(doc), encoding='utf-8')


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--run':
        run_one(sys.argv[2])
    else:
        if '--export' not in sys.argv:
            run_all()
        export()
