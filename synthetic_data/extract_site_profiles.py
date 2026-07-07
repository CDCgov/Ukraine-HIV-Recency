"""
Extract NON-PERSONAL profiles of the testing sites.

Run LOCALLY on the real data/input_data.xlsx to build site_profiles.csv — a
table of AGGREGATE counts for each testing site. The output contains none of:
  * any individual record,
  * any real coordinate,
  * any real site identifier (they are replaced with SITE_0001, SITE_0002, ...).

It keeps only what the generator needs to rebuild the STRUCTURE:
  - the oblast (ADM1 province) the site sits in, for geographic plausibility;
  - how many recent / long-term cases (i.e. the recency share per site);
  - how many high / low risk (i.e. the risk share per site);
  - the site's real active window (when it had cases): date_min..date_max;
  - the operating dates activation_date / deactivation_date (site lifespan).

Besides the summary profiles (site_profiles.csv) it also writes a MONTHLY
breakdown (site_monthly.csv): the same recent/long-term/high/low counts for each
(site, calendar month). The generator needs this so it can reproduce the TIME
course of the recency share, not just the total. Hotspot detection compares a
current window with history (splitting cases strictly by test_date), so without
the monthly structure the synthetic data would show the same share in both
windows and lose the hotspot signal. Monthly granularity reproduces any window
boundary, because it is finer than the analysis boundaries (which are monthly).

site_profiles.csv / site_monthly.csv are safe to keep locally; they are
aggregate numbers per site (the monthly breakdown is finer, so some cells may be
small), so the decision to commit them to a repository is the data owner's.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

# --- paths (relative to the project root) -----------------------------------
PROJECT = Path(__file__).resolve().parent.parent
EXCEL = PROJECT / "data" / "input_data.xlsx"
ADM1 = PROJECT / "data" / "Ukraine_Adm1_Oblast.geojson"
OUT = Path(__file__).resolve().parent / "site_profiles.csv"
OUT_MONTHLY = Path(__file__).resolve().parent / "site_monthly.csv"

OBLAST_COL = "ADM1_EN"


def main() -> None:
    cases = pd.read_excel(EXCEL, sheet_name="hiv_cases")
    sites = pd.read_excel(EXCEL, sheet_name="testing_sites")
    cases["test_date"] = pd.to_datetime(cases["test_date"])
    sites["activation_date"] = pd.to_datetime(sites["activation_date"])
    sites["deactivation_date"] = pd.to_datetime(sites["deactivation_date"])

    # --- oblast of each site via a spatial join ------------------------------
    oblasts = gpd.read_file(ADM1)[[OBLAST_COL, "geometry"]].to_crs("EPSG:4326")
    gsites = gpd.GeoDataFrame(
        sites.copy(),
        geometry=gpd.points_from_xy(sites["longitude"], sites["latitude"]),
        crs="EPSG:4326",
    )
    gsites = gpd.sjoin(gsites, oblasts, how="left", predicate="within")
    gsites = gsites.drop_duplicates(subset="site_id")  # a point on a border can match twice

    # --- per-site case aggregates -------------------------------------------
    def agg(group: pd.DataFrame) -> pd.Series:
        t = group["type"].astype(str).str.strip().str.lower()
        r = group["risk_group"].astype(str).str.strip().str.lower()
        return pd.Series({
            "n_recent": int((t == "recent").sum()),
            "n_longterm": int((t == "long-term").sum()),
            "n_other": int((~t.isin(["recent", "long-term"])).sum()),
            "n_high": int((r == "high").sum()),
            "n_low": int((r == "low").sum()),
            "date_min": group["test_date"].min(),
            "date_max": group["test_date"].max(),
        })

    by_site = cases.groupby("site_id", dropna=False).apply(agg, include_groups=False)

    # --- summary: one row per site (including sites with zero cases) ---------
    # syn_id_map: real site_id -> SITE_0001... (deterministic order by site_id).
    # The same mapping is reused for the monthly breakdown below.
    ordered = gsites.sort_values("site_id").reset_index(drop=True)
    syn_id_map = {srow["site_id"]: f"SITE_{i:04d}"
                  for i, (_, srow) in enumerate(ordered.iterrows(), start=1)}

    rows = []
    for _, srow in ordered.iterrows():
        sid = srow["site_id"]
        prof = by_site.loc[sid] if sid in by_site.index else None
        rows.append({
            "syn_site_id": syn_id_map[sid],
            "oblast": srow.get(OBLAST_COL),
            "n_recent": int(prof["n_recent"]) if prof is not None else 0,
            "n_longterm": int(prof["n_longterm"]) if prof is not None else 0,
            "n_other": int(prof["n_other"]) if prof is not None else 0,
            "n_high": int(prof["n_high"]) if prof is not None else 0,
            "n_low": int(prof["n_low"]) if prof is not None else 0,
            "case_date_min": prof["date_min"] if prof is not None else pd.NaT,
            "case_date_max": prof["date_max"] if prof is not None else pd.NaT,
            "activation_date": srow["activation_date"],
            "deactivation_date": srow["deactivation_date"],
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8")

    # --- monthly breakdown: one row per (site, calendar month) --------------
    # Sites with cases only; syn_site_id from the same mapping. Gives the
    # generator the time course of the recency share (current vs history window).
    mcases = cases[cases["site_id"].isin(syn_id_map)].copy()
    mcases["year_month"] = mcases["test_date"].dt.to_period("M").astype(str)
    t = mcases["type"].astype(str).str.strip().str.lower()
    r = mcases["risk_group"].astype(str).str.strip().str.lower()
    mcases["_recent"] = (t == "recent").astype(int)
    mcases["_longterm"] = (t == "long-term").astype(int)
    mcases["_other"] = (~t.isin(["recent", "long-term"])).astype(int)
    mcases["_high"] = (r == "high").astype(int)
    mcases["_low"] = (r == "low").astype(int)
    monthly = (
        mcases.groupby(["site_id", "year_month"], dropna=False)[
            ["_recent", "_longterm", "_other", "_high", "_low"]
        ].sum().reset_index()
    )
    monthly.insert(0, "syn_site_id", monthly["site_id"].map(syn_id_map))
    monthly = monthly.drop(columns="site_id").rename(columns={
        "_recent": "n_recent", "_longterm": "n_longterm", "_other": "n_other",
        "_high": "n_high", "_low": "n_low",
    })
    monthly = monthly.sort_values(["syn_site_id", "year_month"]).reset_index(drop=True)
    monthly.to_csv(OUT_MONTHLY, index=False, encoding="utf-8")

    # --- short console report (aggregate) -----------------------------------
    n_cases = out["n_recent"].sum() + out["n_longterm"].sum() + out["n_other"].sum()
    rt = out["n_recent"].sum() + out["n_longterm"].sum()
    print(f"[OK] Saved profiles for {len(out)} sites -> {OUT.name}")
    print(f"[OK] Monthly breakdown {len(monthly)} rows ({monthly['year_month'].nunique()} months) -> {OUT_MONTHLY.name}")
    print(f"     sites with cases: {(out[['n_recent','n_longterm','n_other']].sum(axis=1) > 0).sum()}")
    print(f"     total cases:      {n_cases}")
    print(f"     recent share:     {out['n_recent'].sum() / rt:.4f}" if rt else "     recent share: n/a")
    print(f"     oblasts covered:  {out['oblast'].nunique(dropna=True)}")
    missing = out["oblast"].isna().sum()
    if missing:
        print(f"     [warning] sites with no oblast (point outside all polygons): {missing}")


if __name__ == "__main__":
    main()
