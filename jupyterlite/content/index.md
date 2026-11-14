# Ukraine HIV recency hotspots: method notebooks

These notebooks explain how the hotspot pipeline in
[CDCgov/Ukraine-HIV-Recency](https://github.com/CDCgov/Ukraine-HIV-Recency) turns
recent-infection test results into maps, using a **fully synthetic** dataset that
contains no real record and no real location.

1. [The synthetic dataset](01_synthetic_data.ipynb)
2. [How the detector works](02_how_the_detector_works.ipynb)
3. [Results maps](03_results_maps.ipynb)
4. [Does the answer depend on the grid?](04_grid_sensitivity.ipynb)
5. [Run the Bayesian model in your browser](05_live_model.ipynb)

In the interactive version (JupyterLite) every notebook runs in the browser; the
interactive map `hotspots_map.jGIS` opens in JupyterGIS.
