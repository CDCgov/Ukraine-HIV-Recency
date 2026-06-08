# Ukraine HIV Recency Hotspot Detection

**⚠️ INTERPRETATION NOTE:**
This pipeline identifies areas with an elevated proportion of *recent* HIV
infections among those tested — NOT geographic areas where infected
individuals reside. Results describe testing-network performance and must be
combined with local knowledge for decision-making.

A Bayesian surveillance pipeline that flags hotspots of recent HIV
infection in Ukraine from facility-based recency-testing data, aggregated
onto an **H3 hexagonal grid** (res3 / res4) or **ADM1 oblasts** — selectable
per run, individually or in combination.

## Data

The raw case table (`data/input_data.xlsx`) is **not included** in this
repository — it is git-ignored and distributed separately to authorised
collaborators. The pipeline reads facility-based recency-testing records
(one row per test, located to the testing site); see
[`data/README.md`](data/README.md) for the expected schema and handling
rules. Only the public geometry layers (H3 grid, administrative boundaries)
are version-controlled.

---

## What it does

For each **unit** (H3 hexagon or oblast) the pipeline estimates the
proportion of recent infections among newly-diagnosed and compares it against
the national picture along two axes:

- **SMR** (Standardised Morbidity Ratio) — the unit's current
  proportion versus the **current** national rate. Answers *"is the
  recency proportion here higher than the country right now?"*
- **SIR** (Standardised Incidence Ratio) — the unit's current
  proportion versus its **own Empirical-Bayes-shrunken history**, adjusted
  for the national trend. Answers *"is this area rising relative to where
  it used to be?"*

Both axes are evaluated against FDR-controlled exceedance probabilities and
crossed into a seven-category taxonomy that separates a fresh rise from a
sustained high level from a wind-down (see *Classification* below).

Because the recent-event counts per unit are often small, the rigorous
classification is complemented by a **watch-list** that ranks units for
field triage by burden and relative rate (see *Watch-list* below).

The per-unit rate is fitted with a hierarchical **Beta-Binomial**
model in PyMC (partial pooling across units, prior centred on the
national rate). The Beta-Binomial recovers the Binomial as the
concentration parameter grows, so overdispersion is handled without a
separate model-selection step.

The model is **exchangeable**, not spatial: units borrow strength from
the national pool, not from their geographic neighbours. For facility-based
testing data, adjacent units can serve completely different
populations, so spatial smoothing across neighbours would be misleading.

---

## Models

| Model | Role |
|-------|------|
| **Bayesian (crude)** | The **primary detector**. Estimates the recency proportion per unit with no covariate adjustment. Drives the hotspot list, the maps and the recommendations. |
| **Bayesian + covariates** | An **explanatory layer**, reported alongside the crude result (never overriding it). Adjusts for `proportion_high_risk` to ask *"is the burden higher than the risk-group mix predicts?"*. The adjustment is descriptive, not causal. |
| **Truncated Binomial** | Optional branch (`--use-hurdle`) for very sparse data dominated by structural zeros — fits the Beta-Binomial only on active testing sites. |

The covariate model is descriptive on purpose: `proportion_high_risk` lies
on the causal pathway from local environment to recent infection, so
treating it as a confounder would mask the signal the system is meant to
catch.

---

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `geopandas`, `shapely`,
`matplotlib`, `contextily`, `statsmodels`, `scipy`, `pymc`, `arviz`,
`openpyxl`. A working C compiler is needed for PyMC/PyTensor; the run logs
a warning at startup if none is found.

---

## Input data

Place the following in `data/`:

- `input_data.xlsx` with two sheets:
  - `hiv_cases` — one row per test (`test_date`, `longitude`, `latitude`,
    `type` where `type == 'recent'` marks a recent infection; optional
    `site_id`, `risk_group`). Unit assignment is by a **spatial join on
    the test coordinates**, not by any place-name column. The coordinates
    are the **testing-site** (health-facility) location, not a patient
    residence.
  - `testing_sites` (optional) — site coordinates plus `activation_date` /
    `deactivation_date`. The deactivation dates are what let the pipeline
    drop sites closed by the war when deciding `site_present` per period.
    If the sheet is absent the pipeline still runs (falls back to observed
    presence).
- H3 geometry: `h3_hexagons_res4.geojson` (and `res3` / `res5` when used).
- `Ukraine_Adm*.geojson` boundary layers are **not** an analysis mode — they
  are used only to (a) label each hexagon with its oblast/rayon/community
  name in the iterative report and (b) draw oblast outlines on the fallback
  map when a fit fails to converge.

---

## Running

The entry point is `run_hotspots.py`. Step-by-step guides live in `docs/`
([`docs/QUICK_START.md`](docs/QUICK_START.md),
[`docs/HOW_TO_RUN.md`](docs/HOW_TO_RUN.md)).

```bash
# Smoke test on the built-in default config — runs straight through, no wizard
python run_hotspots.py --test

# Full run on a config file
python run_hotspots.py config.json
```

### The interactive wizard

`python run_hotspots.py config.json` **always** runs the wizard (there is no
"use defaults?" shortcut and no config-driven auto-start). It asks:

1. **Analysis type** — standard (single window) or iterative (sliding windows).
2. **Levels** — any combination of `res3`, `res4`, `adm1` (oblasts). Each
   selected level is analysed separately, with its own reports and maps.
3. **Analysis window**
   - *iterative:* 3 / 6 / 9 / 12 months.
   - *standard:* you enter the period start/end (window ≤ 12 months).
   The **baseline length is derived** from the window: 1-6 m → 12 m,
   7-9 m → 18 m, 10-12 m → 24 m. Baselines never start before the
   **2023-01-01 floor**; a period/range whose baseline would cross it is
   rejected (single mode) or yields the "earliest usable date" error
   (iterative, which also requires ≥ 2 iterations).
4. *standard only:* Bayesian parametrization and model selection.

The period, baseline and levels come from the wizard, not the config file.
Scripted / non-interactive runs (e.g. background validation) bypass the
wizard via [`validation/service_run.py`](validation/service_run.py) using a
fully-specified config.

### Command-line options

| Flag | Effect |
|------|--------|
| `--test` | Run on the built-in `DEFAULT_CONFIG` (a config file is optional); no wizard. |
| `--use-loo-ic` | Use LOO-IC for model selection instead of the heuristic score. |
| `--use-hurdle` | Enable the Truncated-Binomial branch for sparse data. |
| `--hurdle-threshold N` | Structural-zero percentage that triggers the hurdle suggestion (default 70). |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | Console / file log verbosity (default INFO). |
| `--no-log-stdout` / `--no-log-file` | Disable console or file logging. |

---

## Configuration (`config.json`)

`config.json` holds the **stable** settings — data paths, CRS, geometry,
columns, colours, the Bayesian prior block, and the detection / watch-list
thresholds. The **run-specific** choices (analysis type, levels, analysis
window, period / date range — and the derived baseline) are picked in the
wizard, not stored in the config. Key fields:

```json
{
  "excel_path": "data/input_data.xlsx",
  "analysis_mode": "h3_hexagons",
  "output_dir": "output",
  "target_crs": "EPSG:3857",
  "administrative_units": { "adm1_path": "data/Ukraine_Adm1_Oblast.geojson", "oblast_col": "ADM1_EN", "...": "..." },
  "h3_hexagons": { "res3_path": "...", "res4_path": "...", "...": "..." },
  "bayesian": {
    "use_non_centered": true,
    "auto_select_parametrization": true
  },
  "detection": { "smr_threshold": 2.0, "sir_threshold": 1.5 },
  "watchlist": { "burden_top_frac": 0.80, "rate_percentile": 0.80 }
}
```

- **Levels** are chosen in the wizard: any combination of `res3`, `res4`,
  `adm1` (oblasts). `analysis_mode` stays `h3_hexagons` (a geometry flag);
  the oblast level is driven by the level choice and reads
  `administrative_units` / `adm1_path`.
- **Analysis window & baseline** are wizard choices too (see *The interactive
  wizard*): the baseline length is derived from the window (1-6 m → 12,
  7-9 m → 18, 10-12 m → 24) and never starts before 2023-01-01.
- **`detection`** — the epidemiological cut-offs for the SMR/SIR exceedance
  taxonomy. A unit is flagged on an axis when `P(ratio > threshold)` clears
  its FDR cut-off. `smr_threshold = 2.0` (a doubling vs national) and
  `sir_threshold = 1.5` are the conventional elevated / moderately-elevated
  levels; tune them here.
- **`watchlist`** — triage knobs for the burden + rate watch-list (see
  *Watch-list*). `burden_top_frac` (default 0.80) sets the cumulative share of
  the recent caseload counted as "high burden"; `rate_percentile` (default
  0.80) sets the relative-rate cut (top 20% of the posterior SMR). These do
  **not** affect the rigorous `classification`.
- **`bayesian.resolution_sigma_multiplier`** (optional) — a map from level
  name (e.g. `"Hex_Res4"` or `"Oblast"`) to a multiplier on the prior width;
  larger = weaker shrinkage. Absent → `1.0`.
- **`bayesian.frr`** (optional) — false-recent-rate correction; off by
  default and not part of the standard protocol (the indicator is a
  proportion, not an incidence estimate).

Scripted (non-interactive) runs put the run-specific fields directly in a
config JSON and use `validation/service_run.py`.

Tunable constants with their literature sources live in
`pipeline/constants.py` (`ANALYSIS_CONSTANTS`).

---

## Output

Each run writes to a timestamped folder `output/<YYYYMMDDhhmmss>/`:

```
output/<timestamp>/
├── bayesian/hex/<resN>/...
│   ├── Report_Hex_<resN>_<period>.xlsx          # per-unit results + disclaimer
│   ├── Diagnostics_Hex_<resN>_<period>.xlsx     # convergence / quality metrics
│   ├── Map_Hex_<resN>_<period>_EN.png           # anomaly choropleth (EN + UA)
│   ├── Reliability_Map_*_*.png                  # HIGH/MODERATE/LOW reliability
│   ├── Watchlist_Map_*_*.png                    # burden + rate triage map
│   ├── Interpretation_*.txt / Specification_Analysis_*.txt
│   └── *_PPC.png / *_Forest.png / *_Pairs.png   # diagnostic plots
├── bayesian_covariates/hex/<resN>/...           # parallel explanatory layer
├── summary/
│   ├── Dashboard_*.png                          # one-page overview
│   └── Results_*.json                           # snapshot for historical comparison
└── pipeline.log
```

The per-unit `Report_*.xlsx` carries, alongside the counts and the
classification: the posterior SMR **mean and median** with its 95% credible
interval, the reliability score/category, and the watch-list columns
(`on_watchlist`, `watch_reason`, `watch_rank`, `burden_rank`, `rate_rank`,
`burden_share_pct`). The oblast level writes the same files under
`bayesian/admin/Oblast/`.

In iterative mode the rolling-window hotspot report (with each hexagon
labelled by its oblast/rayon/community name) and a
`SkippedWindows_convergence_fatal.csv` (windows whose posterior failed to
converge) are written under `iterative/`.

---

## Classification

The SIR × SMR cross yields seven labels:

| Label | Meaning |
|-------|---------|
| 🔴 **Established hotspot** | High on both axes — sustained, currently elevated. |
| 🟠 **Emerging hotspot** | Rising vs its own history, not yet above national — early signal. |
| 🟡 **Stable high-burden** | Above national but not rising — chronic level. |
| 🔵 **Declining from high-burden** | Falling from a high level. |
| 🟢 **Emerging decrease** / **Significant decrease** | Downward trends. |
| ⚪ **Normal** | No signal on either axis. |

New hexagons (no historical data) are marked with a `○` symbol on the map
and classified on SMR only — the trend axis is undefined for them.

### Watch-list (burden + rate triage)

On sparse recency data the binary rate axis is under-powered: a unit with a
genuinely high rate but only a handful of recent events rarely clears the
FDR floor, and a high-burden centre whose *rate* is near average is never
flagged at all. The watch-list is an **additive triage layer** (it does not
change the classification above) that surfaces both:

- **Burden** — recent-case count as a share of the level-wide total;
  `burden_high` marks the units carrying the top `burden_top_frac` (default
  80%) of the recent caseload.
- **Rate (relative)** — `rate_high` marks units whose posterior SMR sits in
  the top `1 - rate_percentile` (default top 20%) of the active distribution,
  or that are already an FDR-flagged hotspot.

A unit is on the list (`on_watchlist`) if it is notable on **either** axis,
recorded in `watch_reason` as `burden` / `rate` / `both`; `watch_rank` orders
the list by priority (best standing on either axis). This is a **ranking for
triage, not a significance test** — read it next to the classification, not
instead of it. The `Watchlist_Map_*.png` colours units by reason and labels
them with their rank.

### Reliability

Every unit carries a reliability rating driven by a hard convergence
gate plus the coefficient of variation of its posterior rate:

- **Hard gate → UNRELIABLE / NaN** if the fit is unhealthy:
  `convergence_fatal` (>5% divergences after adaptive retries), R-hat ≥
  1.01 anywhere, or minimum ESS < 400 (Vehtari et al. 2021).
- Otherwise **HIGH / MODERATE / LOW** from the posterior CV.

When the hard gate trips, the Excel report is skipped (a `_SKIPPED.txt`
sentinel is written instead) and the map degrades to oblast outlines plus a
WARN banner — unhealthy results are never drawn as if trustworthy.

---

## Verification

The standalone verification scripts live in `validation/`:

```bash
python run_hotspots.py --test                       # full pipeline on the default config
python validation/test_convergence_gate.py          # convergence-gate smoke test
python validation/simulation_validation.py          # synthetic FDR / sensitivity / specificity check
python validation/multiseed_stability.py config.json --seeds 42 43 44 --resolution 4
```

`validation/multiseed_stability.py` refits the model under several random seeds and
reports how stable each hexagon's classification is — a check that the
labels are not an artefact of the seed.

---

## Methodology

Core principles as implemented in this codebase:

- The measured quantity is the **proportion of recent infections among the
  newly-diagnosed who were recency-tested** (RITA: rapid recency assay +
  viral load, with ART-experienced / previously-known positives excluded).
  It is a proportion, not an incidence estimate.
- Each unit is compared to the national baseline along two independent
  axes (SMR vs current national, SIR vs own EB-shrunken history), each
  FDR-controlled.
- The hierarchical model is **exchangeable** (no spatial structure), which
  is the appropriate choice for facility-based surveillance where adjacent
  units need not be epidemiologically similar.
- Geocoding is by **test location**, so spatial attribution is to the
  catchment of a testing site, not a residence — an ecological limitation
  noted in the reports.
- A fixed-seed configuration (`random_seed`) makes every run reproducible.

**Interpretation caveats:**

- **Case-mix over time.** The composition of who is recency-tested has
  shifted across the programme (declining share of key populations). Because
  risk groups differ in their recent-infection share, this can confound
  comparisons of a place against its own past (the SIR axis). A
  decomposition of the observed national decline attributes only a small
  part (~5–18%) to this composition shift and the large majority to a
  genuine within-group decline — but local comparisons should still be read
  with the case-mix change in mind.
- **Assay change.** Recent-fraction levels are **not comparable across a
  change of recency assay** (e.g. Asante → LAg): different assays imply a
  different mean duration of recent infection. Treat a post-switch period as
  a fresh baseline; do not compare SIR across the switch.

---

## References

- Gelman & Hill (2007) — *Data Analysis Using Regression and
  Multilevel/Hierarchical Models*.
- Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021) — Rank-normalised
  R-hat and ESS, *Bayesian Analysis*.
- Cameron & Trivedi (2013) — *Regression Analysis of Count Data*.
- Benjamini & Hochberg (1995) — FDR control.

---

## Related documents

* [Open Practices](open_practices.md)
* [Rules of Behavior](rules_of_behavior.md)
* [Thanks and Acknowledgements](thanks.md)
* [Disclaimer](DISCLAIMER.md)
* [Contribution Notice](CONTRIBUTING.md)
* [Code of Conduct](code-of-conduct.md)

## General Disclaimer

This repository was created for use by CDC programs to collaborate on public
health related projects in support of the [CDC mission](https://www.cdc.gov/about/cdc/#cdc_about_cio_mission-our-mission).
GitHub is not hosted by the CDC, but is a third party website used by CDC and
its partners to share information and collaborate on software. CDC use of
GitHub does not imply an endorsement of any one particular service, product,
or enterprise.

## Public Domain Standard Notice

This repository constitutes a work of the United States Government and is not
subject to domestic copyright protection under 17 USC § 105. This repository is in
the public domain within the United States, and copyright and related rights in
the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
All contributions to this repository will be released under the CC0 dedication. By
submitting a pull request you are agreeing to comply with this waiver of
copyright interest.

## License Standard Notice

The repository utilizes code licensed under the terms of the Apache Software
License and therefore is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it and/or modify it under
the terms of the Apache Software License version 2, or (at your option) any
later version.

This source code in this repository is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the Apache Software License for more details.

You should have received a copy of the Apache Software License along with this
program. If not, see http://www.apache.org/licenses/LICENSE-2.0.html

The source code forked from other open source projects will inherit its license.

## Privacy Standard Notice

This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](DISCLAIMER.md)
and [Code of Conduct](code-of-conduct.md).
For more information about CDC's privacy policy, please visit [http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).

## Contributing Standard Notice

Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo)
and submitting a pull request. (If you are new to GitHub, you might start with a
[basic tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you grant a world-wide, royalty-free, perpetual, irrevocable,
non-exclusive, transferable license to all users under the terms of the
[Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or
later.

All comments, messages, pull requests, and other submissions received through
CDC including this GitHub page may be subject to applicable federal law, including but not limited to the Federal Records Act, and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice

This repository is not a source of government records, but is a copy to increase
collaboration and collaborative potential. All government records will be
published through the [CDC web site](http://www.cdc.gov).

## Additional Standard Notices

Please refer to [CDC's Template Repository](https://github.com/CDCgov/template) for more information about [contributing to this repository](https://github.com/CDCgov/template/blob/main/CONTRIBUTING.md), [public domain notices and disclaimers](https://github.com/CDCgov/template/blob/main/DISCLAIMER.md), and [code of conduct](https://github.com/CDCgov/template/blob/main/code-of-conduct.md).
