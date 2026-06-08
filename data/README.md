# Data Directory

This directory holds the geometry layers used by the pipeline and is where
the case data file is placed locally. **The case data file is not part of
this repository** — it is git-ignored and shared with authorised
collaborators separately.

## Files

### 1. Case data — `input_data.xlsx` (NOT in the repository)

Place this file here locally; it is excluded by `.gitignore` and must never
be committed or pushed.

**About the data.** Records are **case-level**: one row per test, keyed by a
unique identifier that is held only by the dataset owner. The file contains
**no direct personal identifiers** (no name, address, or national ID). The
`longitude` / `latitude` are the coordinates of the **testing site** (the
health-care facility), **not** a patient residence — so all geographic
attribution is to the catchment of a testing site.

**Structure** — two sheets:

- **`hiv_cases`** (required) — one row per test. Columns the loader reads:
  - `test_date` — date of the test (parsed with `pd.to_datetime`)
  - `type` — test result: `recent`, `long-term`, or `negative`
    (`type == 'recent'` marks a recent infection)
  - `longitude`, `latitude` — testing-site coordinates (WGS84 / EPSG:4326)
  - optional: `site_id`, `risk_group`
- **`testing_sites`** (optional) — one row per site with coordinates and
  `activation_date` / `deactivation_date`. Used to decide which sites were
  open in a given period; the pipeline runs without it (falls back to
  observed presence).

**Geographic assignment** is done by a **spatial join of the testing-site
coordinates onto the H3 hexagons** — there are no oblast/rayon name columns
in the case sheet, and none are required.

### 2. H3 hexagon grid (in the repository)

- `h3_hexagons_res4.geojson` — primary analysis grid (Res4)
- `h3_hexagons_res3.geojson` / `h3_hexagons_res5.geojson` — optional
  coarser / finer grids

GeoJSON with an `h3_id` property and `geometry` (hexagon polygons). Public,
non-sensitive geometry.

### 3. Administrative boundaries (in the repository — labelling only)

- `Ukraine_Adm3_OTG.geojson` — Community (ADM3)
- `Ukraine_Adm2_Rayon.geojson` — District (ADM2)
- `Ukraine_Adm1_Oblast.geojson` — Oblast (ADM1)
- `Ukraine_Adm0_Country.geojson` — Country outline

GeoJSON with `ADM3_EN` / `ADM2_EN` / `ADM1_EN` name properties and
`geometry`. These are **not** an analysis mode: they are used only to
(a) label each hexagon with its oblast/rayon/community name in the iterative
report (reverse geocoding by hexagon centroid) and (b) draw oblast outlines
on the fallback map when a fit fails to converge. Public, non-sensitive
boundaries.

## Data format

**Excel file** (`input_data.xlsx`, sheet `hiv_cases`) — one row per test:

```
test_date  | type      | longitude | latitude  | site_id | risk_group
-----------|-----------|-----------|-----------|---------|------------
2026-01-15 | recent    | 30.5234   | 50.4501   | KY-01   | high
2026-01-20 | negative  | 24.0297   | 49.8397   | LV-03   | low
2026-01-22 | long-term | 36.2304   | 49.9935   | KH-02   | high
...
```

`type` must be one of `recent` / `long-term` / `negative`. Coordinates are
WGS84 (EPSG:4326); the pipeline reprojects to the configured `target_crs`.

**GeoJSON files**: standard GeoJSON. The H3 grid carries an `h3_id`
property; the administrative layers carry `ADM3_EN` / `ADM2_EN` / `ADM1_EN`.

## Handling the case file

Although the repository is public, the case file itself stays out of it and
is handled according to the project's data-use agreement:

- Keep `input_data.xlsx` only in this local directory; verify it is matched
  by `.gitignore` before any commit.
- Do not email the file or copy it to unsecured locations; transfer it to
  collaborators through the agreed secure channel.
- Prefer encrypted storage and follow the institutional retention policy;
  delete local copies when no longer needed.

## Troubleshooting

**File-not-found errors** — check that `config.json` paths are correct, that
`input_data.xlsx` exists in this directory, that the boundary files are
present, and that names match exactly (case-sensitive).

**Data-format errors** — verify the `hiv_cases` sheet has the expected
columns (`test_date`, `type`, `longitude`, `latitude`), that dates parse
(YYYY-MM-DD recommended), and that coordinates are valid WGS84 lon/lat
inside Ukraine (cases outside every hexagon are dropped by the spatial join).
