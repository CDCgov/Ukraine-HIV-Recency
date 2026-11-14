# Browser notebooks (JupyterLite, JupyterGIS, Jupyter Book)

This folder builds a website that explains the hotspot method with notebooks that
run **entirely in the browser**: nothing to install, and no data leave the page. All
results shown are computed on the **fully synthetic** dataset
`data/synthetic_input_data.xlsx`; no real record is read at any step.

The same notebooks are published two ways:

- **JupyterLite** (`/`): an in-browser JupyterLab with a Python (Pyodide) kernel, so
  every cell can be re-run and changed; `hotspots_map.jGIS` opens as an interactive
  map in **JupyterGIS**;
- **Jupyter Book** (`/book/`): a static, readable version with the saved outputs.

## Contents

| Notebook | What it shows |
|---|---|
| `01_synthetic_data.ipynb` | the synthetic dataset, the analysis and baseline windows |
| `02_how_the_detector_works.ipynb` | level and trend, the 0.80 rule, labels, gates, reliability |
| `03_results_maps.ipynb` | H3 and oblast maps, the areas with a signal, the watch-list |
| `04_grid_sensitivity.ipynb` | the same data on eight DGGS grids (MAUP check) |
| `05_live_model.ipynb` | the joint two-period PyMC model fitted live in the browser |
| `hotspots_map.jGIS` | interactive JupyterGIS map of the H3 results |

`05_live_model.ipynb` runs PyMC in pure-Python mode in the browser (no numba, no C
compiler); with the reduced settings it took about 6 minutes in our test.

## Layout

```
jupyterlite/
  build_content.py      runs the pipeline on the synthetic data, exports content/data/ and the .jGIS map
  make_notebooks.py     writes the notebooks in content/ (edit text there, then re-run)
  grids/                the eight DGGS grids (geometry only, cells of about 6-21 thousand km²)
  content/              what the site serves: notebooks, hotspots_lite.py, data/, myst.yml (book)
  jupyter_lite_config.json, requirements-build.txt
```

## Rebuilding

1. Precompute the results (needs the full pipeline environment with PyMC):

   ```bash
   python jupyterlite/build_content.py          # 9 pipeline runs, then export
   python jupyterlite/make_notebooks.py         # only if the notebook text changed
   ```

   Then execute the notebooks once so that their saved outputs (shown in the book
   and on first opening) match the data.

2. Build the site locally (separate environment recommended):

   ```bash
   pip install -r jupyterlite/requirements-build.txt
   cd jupyterlite && jupyter lite build            # -> jupyterlite/_output
   cd content && jupyter book build --html         # -> jupyterlite/content/_build/html
   ```

3. Publishing: `.github/workflows/jupyterlite.yml` builds both and deploys them to
   GitHub Pages (manual trigger, or a push to `main` that touches `jupyterlite/`).
   A repository admin must first set *Settings → Pages → Source* to *GitHub Actions*.

## Privacy

Only synthetic data and public boundaries are shipped, and the notebooks say nothing
about results on real data. Grid files contain cell geometry only. How the synthetic
dataset was generated is described in `synthetic_data/README.md`.
