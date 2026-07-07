"""
Generate a SYNTHETIC input_data.xlsx from the non-personal site profiles.

Reads the aggregated per-site counts (site_profiles.csv) and the oblast
(province) polygons, then builds a completely made-up dataset that has the same
STRUCTURE as the real one but contains no real record and no real coordinate:

  * each site is given an invented coordinate — a random point INSIDE its own
    oblast, so province-level geography is preserved while the exact clinic
    location is hidden;
  * cases are laid down MONTH BY MONTH: within each calendar month we take that
    month's recent/long-term and high/low-risk shares (from site_monthly.csv)
    and draw from a binomial, so the result is close to the real share but not
    an exact copy. This reproduces the TIME course of the recency share, so the
    hotspot detection (current window versus history) sees the same signal on
    the synthetic data that it does on the real data;
  * each month's case dates fall inside that month, trimmed to the site's
    operating window (activation/deactivation);
  * activation_date / deactivation_date are carried over verbatim.

The result is synthetic_input_data.xlsx with the hiv_cases and testing_sites
sheets — exactly the columns the pipeline expects. The run is reproducible
(fixed SEED). The public notebook overrides the input/output paths below to run
this on the shipped synthetic file instead of the local defaults.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

SEED = 2026

PROJECT = Path(__file__).resolve().parent.parent
ADM1 = PROJECT / "data" / "Ukraine_Adm1_Oblast.geojson"
PROFILES = Path(__file__).resolve().parent / "site_profiles.csv"
MONTHLY = Path(__file__).resolve().parent / "site_monthly.csv"
OUT = Path(__file__).resolve().parent / "synthetic_input_data.xlsx"

OBLAST_COL = "ADM1_EN"
JITTER_DEG = 0.02  # ~2 km: spread each case around its site (stand-in for residence)


def random_point_in(poly, rng: np.random.Generator) -> Point:
    """Return a random point inside a polygon (rejection sampling on its bounding box)."""
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(10000):
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if poly.contains(p):
            return p
    return poly.representative_point()  # fallback


def month_bounds(ym: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last day of the calendar month given as 'YYYY-MM'."""
    start = pd.Period(ym, freq="M").start_time.normalize()
    end = pd.Period(ym, freq="M").end_time.normalize()
    return start, end


def clip_to_operation(lo: pd.Timestamp, hi: pd.Timestamp,
                      act, deact) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Clip the range [lo, hi] to the site's operating window (activation/deactivation)."""
    if pd.notna(act):
        lo = max(lo, pd.to_datetime(act))
    if pd.notna(deact):
        hi = min(hi, pd.to_datetime(deact))
    return lo, hi


def main() -> None:
    rng = np.random.default_rng(SEED)
    prof = pd.read_csv(PROFILES, parse_dates=["case_date_min", "case_date_max",
                                              "activation_date", "deactivation_date"])
    monthly = pd.read_csv(MONTHLY, dtype={"year_month": str})
    months_by_site = {sid: g for sid, g in monthly.groupby("syn_site_id")}
    oblasts = gpd.read_file(ADM1)[[OBLAST_COL, "geometry"]].to_crs("EPSG:4326")
    geom_by_oblast = dict(zip(oblasts[OBLAST_COL], oblasts.geometry))

    site_rows, case_rows = [], []
    case_counter = 0

    for _, row in prof.iterrows():
        sid = row["syn_site_id"]
        poly = geom_by_oblast.get(row["oblast"])
        if poly is None:  # site with no oblast match — skip it rather than misplace it
            continue
        site_pt = random_point_in(poly, rng)

        site_rows.append({
            "site_id": sid,
            "longitude": round(site_pt.x, 6),
            "latitude": round(site_pt.y, 6),
            "activation_date": row["activation_date"],
            "deactivation_date": row["deactivation_date"],
        })

        site_months = months_by_site.get(sid)
        if site_months is None:
            continue  # site exists but has no cases

        # draw MONTH BY MONTH: the recency/risk shares are taken per month
        for _, m in site_months.iterrows():
            n_m = int(m["n_recent"]) + int(m["n_longterm"]) + int(m["n_other"])
            if n_m == 0:
                continue

            denom_rec = int(m["n_recent"]) + int(m["n_longterm"])
            p_recent = (m["n_recent"] / denom_rec) if denom_rec > 0 else 0.0
            denom_risk = int(m["n_high"]) + int(m["n_low"])
            p_high = (m["n_high"] / denom_risk) if denom_risk > 0 else 0.0

            # binomial draw -> shares come out slightly different but close
            n_recent = int(rng.binomial(n_m, p_recent))
            types = np.array(["recent"] * n_recent + ["long-term"] * (n_m - n_recent))
            rng.shuffle(types)
            n_high = int(rng.binomial(n_m, p_high))
            risks = np.array(["high"] * n_high + ["low"] * (n_m - n_high))
            rng.shuffle(risks)

            # dates WITHIN this month, trimmed to the site's operating window
            lo, hi = month_bounds(m["year_month"])
            lo, hi = clip_to_operation(lo, hi, row["activation_date"], row["deactivation_date"])
            if hi < lo:  # month falls entirely outside the operating window — keep it within the month
                lo, hi = month_bounds(m["year_month"])
            span_days = max(int((hi - lo).days), 0)
            offsets = rng.integers(0, span_days + 1, size=n_m) if span_days > 0 else np.zeros(n_m, int)
            dates = pd.to_datetime(lo) + pd.to_timedelta(offsets, unit="D")

            # case coordinates: scattered around the site with a small offset (residence stand-in)
            lons = site_pt.x + rng.normal(0, JITTER_DEG, n_m)
            lats = site_pt.y + rng.normal(0, JITTER_DEG, n_m)

            for k in range(n_m):
                case_counter += 1
                case_rows.append({
                    "case_id": f"CASE_{case_counter:06d}",
                    "type": types[k],
                    "test_date": dates[k],
                    "longitude": round(float(lons[k]), 6),
                    "latitude": round(float(lats[k]), 6),
                    "risk_group": risks[k],
                    "site_id": sid,
                })

    cases_df = pd.DataFrame(case_rows, columns=["case_id", "type", "test_date",
                                               "longitude", "latitude", "risk_group", "site_id"])
    sites_df = pd.DataFrame(site_rows, columns=["site_id", "longitude", "latitude",
                                               "activation_date", "deactivation_date"])

    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        cases_df.to_excel(xw, sheet_name="hiv_cases", index=False)
        sites_df.to_excel(xw, sheet_name="testing_sites", index=False)

    rt = (cases_df["type"] == "recent").sum() + (cases_df["type"] == "long-term").sum()
    print(f"[OK] Synthetic file -> {OUT.name}")
    print(f"     cases:           {len(cases_df)}")
    print(f"     sites:           {len(sites_df)} (with cases: {cases_df['site_id'].nunique()})")
    print(f"     recent share:    {(cases_df['type'] == 'recent').sum() / rt:.4f}" if rt else "")
    print(f"     high-risk share: {(cases_df['risk_group'] == 'high').mean():.4f}")
    print(f"     date range:      {cases_df['test_date'].min().date()} -> {cases_df['test_date'].max().date()}")


if __name__ == "__main__":
    main()
