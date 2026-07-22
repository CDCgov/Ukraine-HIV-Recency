# Synthetic Data for Testing the Pipeline

This folder lets anyone run the hotspot pipeline end to end **without the
confidential case data**. The repository already ships a fully synthetic dataset
in the input folder, plus a notebook that turns it into the file the pipeline
runs on and demonstrates the geographic-privacy tools.

## Getting started (step by step)

You do not need any confidential data. Everything you need is in the repository.

**1. Get the code and the test data.**
Clone the repository. The synthetic test dataset ships with it at
`data/synthetic_input_data.xlsx`, so it is already in the input folder — there
is nothing extra to download or move. (If you ever obtain that file separately,
just place it at `data/synthetic_input_data.xlsx`.)

**2. Install the dependencies.**
From the repository root:

```bash
pip install -r requirements.txt
```

**3. Generate the file the pipeline will run on.**
Open and run every cell of
[`synthetic_pipeline_public.ipynb`](synthetic_pipeline_public.ipynb). It reads
`data/synthetic_input_data.xlsx`, generates a second synthetic dataset, and
writes it to `data/synthetic_input_data_double.xlsx`. (The same notebook also
runs the optional encryption and donut-masking demos; those need the extra
libraries noted at the bottom, but the generation step does not.)

**4. Point the pipeline at the generated file.**
In `config.json` set:

```json
"excel_path": "data/synthetic_input_data_double.xlsx"
```

**5. Run the pipeline.**
From the repository root:

```bash
python run_hotspots.py config.json
```

**6. Collect the outputs.**
Everything is written to the `output/` folder: hotspot maps, the Excel report,
the watch-list, and run diagnostics. On the synthetic data the result mirrors
the real data — the same power limit (few recent events) that is the study's
main finding.

## What the two synthetic files are

- **`data/synthetic_input_data.xlsx`** — the **input** that ships with the
  repository. Think of it as a stand-in for the confidential local data. This
  notebook reads it but never treats it as something to publish further.
- **`data/synthetic_input_data_double.xlsx`** — the file the notebook
  **generates** from the input, one more step removed from the real data. This
  is what the pipeline runs on.

## How the synthetic data was made (provenance)

The real, confidential data is used **only once, offline**, by a separate
local-only notebook, to produce `data/synthetic_input_data.xlsx`. That file
already contains no real record: coordinates are random points inside the
correct oblast (province), and case counts are binomial draws from non-personal
per-site, per-month aggregates. The public notebook here then produces the
double-synthetic file from it, with a second independent layer of random
coordinates and noise, and reads no real data at any point.

## Files in this folder

- **`synthetic_pipeline_public.ipynb`** — the generator plus the geoprivacy
  demo (runs entirely on synthetic data).
- **`synthetic_data_demo.ipynb`** — a light notebook to explore the synthetic
  dataset (summary counts, the monthly recency trend, a map of sites).
- **`extract_site_profiles.py`, `generate_synthetic.py`** — the generation
  scripts the notebooks call.

## Privacy, stated honestly

No individual real record and no exact real location can be reconstructed from
either synthetic file. Coordinates are invented random points within a province
and dates are drawn within a month, so there is no link back to any real
individual, and exact clinic locations never enter the data. What the files
deliberately preserve is *aggregate shape* — the approximate recent-infection
share per anonymised site per month — which is coarse, anonymised to province
level, and already reported in the published results; the double-synthetic step
blurs even this a little further. These are properties under any realistic
threat model, not an absolute mathematical guarantee.

## Optional geoprivacy demonstration

The last section of `synthetic_pipeline_public.ipynb` demonstrates the **Map
Encryption Library** (Jim and Herman,
`github.com/PHI-Case-Studies/2026-Map-Encryption-Library`): reversible
encryption of coordinates (exact recovery with a key, a display scattered across
the globe without it) and donut geomasking (a short, structure-preserving
shift). To run this section, clone that library next to this repository or
install it; the donut part also needs `geopy`. The generation steps above do
not require any of these.
