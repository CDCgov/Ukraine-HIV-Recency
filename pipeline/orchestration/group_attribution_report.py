"""
Generate the supplementary risk-group attribution report as part of a run.

Runs the :mod:`pipeline.standardization.group_attribution` products (tested
case-mix shift over time, national group recency rates at escalating windows,
per-oblast high-risk vs general rates with a significance test, and the
per-oblast indirect standardization / SMR) and writes them to
``Group_Attribution_Report.xlsx`` next to the run's summary output.

This is a supplementary explanatory analysis at the OBLAST level (where the
sparse recent-event counts have enough power); it does not touch the detector's
per-unit classification. It is wrapped so a failure never aborts the main run.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd
import pandas as pd

from pipeline.standardization import (
    escalating_group_rates,
    oblast_group_rates,
    oblast_standardization,
    period_composition,
)

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path_str: str) -> Path:
    """Resolve a config path as given, else relative to the repo root."""
    p = Path(path_str)
    return p if p.exists() else (_ROOT / path_str)


def generate_group_attribution_report(config: Dict[str, Any], out_xlsx: Path,
                                      group_col: str = 'risk_group',
                                      high_label: str = 'high') -> Optional[str]:
    """Compute the oblast-level attribution products and write the workbook.

    Returns a short text summary (for the run log) or ``None`` when the report
    cannot be produced (e.g. no ``risk_group`` column, or no oblast layer).
    """
    excel_path = _resolve(config['excel_path'])
    cases = pd.read_excel(excel_path, sheet_name='hiv_cases')
    cases['test_date'] = pd.to_datetime(cases['test_date'])

    if group_col not in cases.columns:
        logger.info(f"Group attribution skipped: no '{group_col}' column in the case data.")
        return None
    if not {'longitude', 'latitude'}.issubset(cases.columns):
        logger.info("Group attribution skipped: case data has no coordinates.")
        return None

    adm = config.get('administrative_units', {})
    adm1_path = adm.get('adm1_path')
    oblast_col = adm.get('oblast_col')
    if not adm1_path or not oblast_col:
        logger.info("Group attribution skipped: no ADM1 layer configured.")
        return None
    gdf_ob = gpd.read_file(_resolve(adm1_path))
    if oblast_col not in gdf_ob.columns:
        logger.info(f"Group attribution skipped: '{oblast_col}' not in the ADM1 layer.")
        return None

    cur_s = pd.Timestamp(config['analysis_period']['start'])
    cur_e = pd.Timestamp(config['analysis_period']['end'])

    pts = gpd.GeoDataFrame(cases, geometry=gpd.points_from_xy(cases['longitude'], cases['latitude']),
                           crs='EPSG:4326')
    joined = gpd.sjoin(pts.to_crs(gdf_ob.crs), gdf_ob[[oblast_col, 'geometry']],
                       how='inner', predicate='within')

    comp = period_composition(cases, group_col=group_col, high_label=high_label, split_year=2024)
    esc = escalating_group_rates(cases, cur_s, group_col=group_col, high_label=high_label)
    rates = oblast_group_rates(joined, oblast_col, group_col=group_col, high_label=high_label)
    std_cur = oblast_standardization(joined, oblast_col, cur_s, cur_e,
                                     group_col=group_col, high_label=high_label)
    std_pool = oblast_standardization(joined, oblast_col, pd.Timestamp('2021-01-01'), cur_e,
                                      group_col=group_col, high_label=high_label)

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx) as w:
        comp['by_year'].to_excel(w, sheet_name='composition_by_year', index=False)
        esc.to_excel(w, sheet_name='escalating_group_rates', index=False)
        rates.to_excel(w, sheet_name='oblast_group_rates', index=False)
        std_cur.to_excel(w, sheet_name='oblast_std_current', index=False)
        std_pool.to_excel(w, sheet_name='oblast_std_pool', index=False)

    # Short log summary: powered oblasts that run credibly hot (pooled SMR CI > 1).
    hot = std_pool[(std_pool['powered']) & (std_pool['smr_lo'] > 1.0)]
    hot_txt = ', '.join(f"{r.oblast} SMR={r.smr:.2f} [{r.smr_lo:.2f}-{r.smr_hi:.2f}]"
                        for r in hot.itertuples()) or 'none'
    summary = (f"Group attribution written: {out_xlsx.name}. "
               f"Composition shift pre/post-2024: {comp['pre_pct']:.0f}%->{comp['post_pct']:.0f}% high-risk "
               f"(p={comp['contrast_p']:.1e}). Oblasts credibly hot beyond composition (pooled): {hot_txt}.")
    logger.info(f"[OK] {summary}")
    return summary
