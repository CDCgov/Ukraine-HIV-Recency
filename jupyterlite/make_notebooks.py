"""Write the JupyterLite notebooks in ``jupyterlite/content/``.

    python jupyterlite/make_notebooks.py

The notebooks are generated from this file so their structure stays
consistent; edit the text here and re-run rather than editing the .ipynb files.
"""
from pathlib import Path

import nbformat as nbf

CONTENT = Path(__file__).resolve().parent / 'content'

INSTALL = '''# In the browser (JupyterLite) install the packages this notebook needs.
# On a normal Python kernel this cell does nothing.
import sys
if sys.platform == "emscripten":
    import piplite
    await piplite.install({packages})'''

GEO = '["pandas", "geopandas", "shapely", "pyproj", "matplotlib", "scipy"]'


def md(text):
    return nbf.v4.new_markdown_cell(text.strip('\n'))


def code(text):
    return nbf.v4.new_code_cell(text.strip('\n'))


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        'kernelspec': {'display_name': 'Python (Pyodide)', 'language': 'python', 'name': 'python'},
        'language_info': {'name': 'python'},
    }
    nb.cells = cells
    nbf.write(nb, CONTENT / name)
    print('wrote', name)


# ---------------------------------------------------------------------------
write('01_synthetic_data.ipynb', [
    md('''
# 1. The synthetic dataset

These notebooks show how the **Ukraine HIV recency hotspot pipeline** works, using
only a **fully synthetic** dataset. They run in your browser (JupyterLite): nothing
is installed on your computer and no data leaves the page.

**What the data are.** The pipeline uses results of recent-infection testing among
people newly diagnosed with HIV. Each record is one test at one testing site, marked
*recent* or *long-term*. The synthetic file has exactly that structure, but **no
record and no location is real**: sites are random points inside an oblast and
case counts are random draws. In this dataset about one to two per cent of tests are
recent.

**What the notebooks cover.**

1. this notebook: the synthetic data;
2. how the detector turns test counts into labels;
3. maps of the results;
4. does the answer depend on the grid? (eight different grids);
5. running the Bayesian model live in the browser.

The precomputed results in `data/` were produced by running the unchanged pipeline
on this synthetic dataset; see `jupyterlite/build_content.py` in the repository.
'''),
    code(INSTALL.format(packages=GEO)),
    code('''
import matplotlib.pyplot as plt
import pandas as pd
from hotspots_lite import build_info, load_cases, load_sites, load_oblasts

info = build_info()
cases = load_cases()
sites = load_sites()
oblasts = load_oblasts()

recent = cases["type"].eq("recent")
pd.Series({
    "tests": len(cases),
    "testing sites": len(sites),
    "recent share (all years)": round(recent.mean(), 4),
    "first test": cases["test_date"].min().date(),
    "last test": cases["test_date"].max().date(),
    "analysis window used in notebooks 2-5": f"{info['window_start']} to {info['window_end']}",
}, dtype=object)
'''),
    md('''
## Two windows: current and baseline

The detector compares an **analysis window** (here six months) with a **baseline
window** just before it. The baseline length follows from the window length: one to
six months of analysis get a twelve-month baseline. The plot shows the monthly recent
share with both windows shaded. Note how noisy a monthly share is when only one or two
per cent of tests are recent; this sparsity shapes every design choice that follows.
'''),
    code('''
start, end = pd.Timestamp(info["window_start"]), pd.Timestamp(info["window_end"])
b_start, b_end = start - pd.DateOffset(months=12), start - pd.Timedelta(days=1)

monthly = (cases.assign(recent=recent, month=cases["test_date"].dt.to_period("M"))
           .groupby("month")["recent"].agg(["sum", "count"]))
monthly["share"] = monthly["sum"] / monthly["count"]
monthly = monthly[monthly["count"] >= 20]   # drop months with very few tests

fig, ax = plt.subplots(figsize=(11, 4))
ax.axvspan(b_start, b_end, color="#bcd4e6", alpha=0.6, label="baseline window")
ax.axvspan(start, end, color="#f6c28b", alpha=0.6, label="analysis window")
ax.plot(monthly.index.to_timestamp(), monthly["share"], marker="o", ms=3, color="#333333")
ax.set_ylabel("recent / all tested")
ax.set_title("Synthetic monthly share of recent infections (months with at least 20 tests)")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
plt.show()

for name, (a, b) in {"baseline": (b_start, b_end), "analysis": (start, end)}.items():
    w = cases[(cases.test_date >= a) & (cases.test_date <= b)]
    print(f"{name:9} {a.date()} to {b.date()}: {len(w):5} tests, {w.type.eq('recent').sum():3} recent")
'''),
    md('''
## Where the sites are

Each synthetic site is a random point inside an oblast; none of these points is a
clinic location.
'''),
    code('''
ax = oblasts.to_crs("EPSG:3035").boundary.plot(figsize=(9, 6.5), color="0.6", linewidth=0.5)
sites.to_crs("EPSG:3035").plot(ax=ax, color="#d62728", markersize=12)
ax.set_title("Synthetic testing sites")
ax.set_axis_off()
plt.show()
'''),
])

# ---------------------------------------------------------------------------
write('02_how_the_detector_works.ipynb', [
    md('''
# 2. How the detector works

The pipeline answers two questions for every area:

* **Level:** is the share of recent infections here higher (or lower) than in the
  rest of the country now?
* **Trend:** has it risen (or fallen) compared with this area's own past?

The answers are combined into one label, and every label carries a **reliability**
rating that says how much data stands behind it. This notebook walks through the
steps on the synthetic data.
'''),
    code(INSTALL.format(packages=GEO)),
    code('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
from hotspots_lite import build_info, load_results, load_oblasts, plot_labels, LABEL_COLORS

info = build_info()
cells = load_results()
active = cells[cells["tests_curr"].fillna(0) > 0].copy()
print(f"Analysis window {info['window_start']} to {info['window_end']}")
print(f"{len(cells)} H3 cells cover Ukraine; {len(active)} have testing in the window")
active[["name", "oblast", "tests_curr", "recent_curr", "tests_hist", "recent_hist"]].sort_values(
    "recent_curr", ascending=False).head(10)
'''),
    md('''
## Step 1. From sites to areas

Every test belongs to a testing site, and every site falls in one area. Here the
areas are **H3 hexagons at resolution 3** (about 12,000 km² each); the pipeline also
works on oblasts. An area's data are simply the sums over its sites: tests and recent
results in the analysis window and in the baseline window. The denominator is the
number of **tests**, not the population, because nobody knows exactly whom a site
serves.

## Step 2. Level: compared with the rest of the country

For each area the model estimates its recent share *p* and asks how likely it is that
*p* is above the national share **computed without that area** (leave-one-out, so a
large area does not pull the reference towards itself). An area is called *above* (or
*below*) when that posterior probability is at least **0.80**. This is the decision
rule of Richardson et al. (2004) for disease maps; a simulation on sparse data like
ours picked 0.80 as the best balance between missed signals and false alarms. Where
the data allow it, the cutoff is raised further so that the expected share of false
calls among flagged areas stays within 5 %.

The production model is a hierarchical Bayesian model (notebook 5 runs it). The cell
below shows the same rule on a much simpler posterior, a Beta distribution per area,
so you can see the mechanics in a second.
'''),
    code('''
recent_total, tests_total = active["recent_curr"].sum(), active["tests_curr"].sum()
national = recent_total / tests_total
loo = (recent_total - active["recent_curr"]) / (tests_total - active["tests_curr"])

prior_strength = 20                      # pseudo-tests pulling small areas towards the national share
a0, b0 = national * prior_strength, (1 - national) * prior_strength
post_a = a0 + active["recent_curr"]
post_b = b0 + active["tests_curr"] - active["recent_curr"]
active["P(above rest of country)"] = 1 - beta.cdf(loo, post_a, post_b)

print(f"National recent share in the window: {national:.3%} ({int(recent_total)} of {int(tests_total)} tests)")
cols = ["name", "tests_curr", "recent_curr", "P(above rest of country)", "label"]
active.sort_values("P(above rest of country)", ascending=False)[cols].head(8).round(3)
'''),
    md('''
Two things stand out. First, with one or two per cent of tests recent, an area needs
both a fair number of tests and several recent results before the probability passes
0.80. Second, the simple posterior and the pipeline's label (last column) mostly agree
but not always: the hierarchical model learns how much areas really differ and shrinks
noisy areas more firmly towards the national level.

## Step 3. Trend: compared with the area's own past

The production model fits both windows **jointly**: each area has its own level and
its own change between the baseline and the analysis window. That change is measured
beyond the national change, so a rise the whole country shares is not a local trend,
and it is partially pooled towards zero. An area is *rising* when the posterior
probability of a positive change is at least 0.80. Pooling matters here, because many
areas have no recent results at all in one of the windows.

## Step 4. Labels

| Trend \\ Level | above the country | similar | below the country |
|---|---|---|---|
| **rising** | Established hotspot | Emerging hotspot | Normal |
| **stable** | Stable high-burden | Normal | Below national |
| **falling** | Declining from high-burden | Emerging decrease | Below national |

Two gates protect against false alarms caused by zeros:

* **presence gate:** a hotspot label needs at least **two recent results** in the
  analysis window; an area cannot be a hotspot of recent infection without them;
* **cold-cell gate:** an area with no recent results in either window has no trend to
  speak of and is *Normal*.

When an area's own history is too thin to judge a trend, an elevated area is labelled
*Elevated vs national (trend uncertain)* rather than being given a direction.
'''),
    code('''
cells["label"].value_counts(dropna=True).rename("areas").to_frame()
'''),
    md('''
## Step 5. Reliability

Every estimate comes with a rating that depends on how wide its 95 % credible interval
is relative to the estimate itself (its coefficient of variation, CV):

score = 100 × exp(−CV); **HIGH** ≥ 80, **MODERATE** ≥ 60, otherwise **LOW**.

A calibration study showed that HIGH estimates are almost always decisive about the
national reference and LOW ones almost never. Maps **hatch** LOW areas so that a reader
sees at once which colours rest on too little data. Separately, if the sampler does not
converge (R-hat ≥ 1.01 or effective sample size ≤ 400), the run is marked unreliable as
a whole.
'''),
    code('''
print(cells["reliability"].value_counts().to_string())
plot_labels(cells, oblasts=load_oblasts(),
            title=f"H3 res 3, {info['window_start']} to {info['window_end']} (synthetic data)")
plt.show()
'''),
    md('''
## What the method cannot do

With a few dozen recent results per window across the country, most areas are LOW
reliability **by nature of the data**, not because of the thresholds. The method is
built to say so openly: it labels only what the data can support, hatches what they
cannot, and never invents a trend from a single count. Locations are those of testing
sites, not of where people live.
'''),
])

# ---------------------------------------------------------------------------
write('03_results_maps.ipynb', [
    md('''
# 3. Results maps

The labels from one pipeline run on the synthetic data, for H3 hexagons and for
oblasts. For an interactive version, open **`hotspots_map.jGIS`** in the file browser
(JupyterGIS): you can pan, zoom, switch layers on and off and click a cell to see its
numbers.
'''),
    code(INSTALL.format(packages=GEO)),
    code('''
import pandas as pd
import matplotlib.pyplot as plt
from hotspots_lite import build_info, load_results, load_oblasts, load_sites, plot_labels, HOTSPOT_LABELS, DATA

info = build_info()
cells, oblasts, sites = load_results(), load_oblasts(), load_sites()
period = f"{info['window_start']} to {info['window_end']}"
plot_labels(cells, oblasts=oblasts, sites=sites, title=f"H3 hexagons, {period} (synthetic data)")
plt.show()
'''),
    md('''
## Areas with a signal

Every area that is not *Normal*, with its data and the national reference ratio
(the area's recent share divided by the national share, median and 95 % interval).
'''),
    code('''
cols = ["name", "oblast", "label", "reliability", "tests_curr", "recent_curr",
        "ratio_median", "ratio_lo", "ratio_hi"]
cells[cells["label"].notna() & cells["label"].ne("Normal")][cols].sort_values(
    "ratio_median", ascending=False).round(2)
'''),
    md('''
## Oblasts

The same model on oblasts. Larger areas hold more tests, so their estimates are
usually more reliable, but a signal concentrated in one city is diluted across a whole
oblast.
'''),
    code('''
obl = pd.read_csv(DATA / "results_oblast.csv")
o = oblasts.merge(obl, left_on="oblast", right_on="oblast_name", how="left")
plot_labels(o, title=f"Oblasts, {period} (synthetic data)")
plt.show()
obl[obl["label"].ne("Normal")][["oblast_name", "label", "reliability", "tests_curr", "recent_curr",
                                "ratio_median", "ratio_lo", "ratio_hi"]].round(2)
'''),
    md('''
## Watch-list

Some areas never pass the 0.80 rule simply because they are small, yet carry a large
share of the country's recent results or an unusually high rate. The pipeline lists
them separately as a **watch-list**: not hotspots, but places to keep an eye on.
'''),
    code('''
w = cells[cells.get("on_watchlist", False) == True]
w[["name", "oblast", "label", "tests_curr", "recent_curr"]] if len(w) else "No cells on the watch-list in this window."
'''),
])

# ---------------------------------------------------------------------------
write('04_grid_sensitivity.ipynb', [
    md('''
# 4. Does the answer depend on the grid?

Counting cases in areas and then mapping the areas can change the answer when the
areas change: the **modifiable areal unit problem** (MAUP). Here every test is tied to
the location of its testing site, so the question becomes: *which sites end up in the
same cell?* A different grid groups sites differently.

To check this, the unchanged pipeline was run on **eight discrete global grid
systems** (DGGS) with cells of roughly the same size as H3 resolution 3: H3, S2, A5,
rHEALPix, HEALPix, DGGRID ISEA4H and ISEA3H, and OpenEAGGR ISEA4T. Exact sizes differ,
because each system steps between resolutions differently (from about 6 to 21
thousand km²). A signal that appears on most grids is robust; a signal that appears on
one grid only is more likely an artefact of where the cell borders fall.
'''),
    code(INSTALL.format(packages=GEO)),
    code('''
import itertools
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hotspots_lite import build_info, load_oblasts, load_sites, load_cases, plot_labels, HOTSPOT_LABELS, DATA

info = build_info()
summary = pd.read_csv(DATA / "grids_summary.csv")
results = pd.read_csv(DATA / "grid_results.csv", dtype={"cell_id": str})
oblasts = load_oblasts()
summary
'''),
    code('''
fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
grids = {}
for ax, row in zip(axes.flat, summary.itertuples()):
    g = gpd.read_file(DATA / "grids" / f"{row.slug}.geojson")
    g["cell_id"] = g["cell_id"].astype(str)
    g = g.merge(results[results.grid == row.slug], on="cell_id", how="left")
    grids[row.slug] = g
    plot_labels(g, ax=ax, oblasts=oblasts, legend=(row.Index == 0),
                title=f"{row.grid} ({row.resolution}), ~{row.area_median_km2 / 1000:.1f} thousand km²")
plt.suptitle(f"The same synthetic data on eight grids, {info['window_start']} to {info['window_end']}"
             " (hatched = low reliability)", fontsize=14)
plt.tight_layout()
plt.show()
'''),
    md('''
## Grid support for each site

Each site inherits the label of the cell it falls in, on every grid. A site's **grid
support** is the number of grids on which its cell is a hotspot. Only sites that tested
in the analysis window count.
'''),
    code('''
cases = load_cases()
win = cases[(cases.test_date >= info["window_start"]) & (cases.test_date <= info["window_end"])]
act = win.groupby("site_id").agg(tests=("type", "size"), recent=("type", lambda x: (x == "recent").sum()))
sites = load_sites()
sites = sites[sites.site_id.isin(act.index)]

support = []
for slug, g in grids.items():
    j = gpd.sjoin(sites[["site_id", "geometry"]], g[["label", "geometry"]], how="left", predicate="within")
    support.append(j.assign(grid=slug, hot=j["label"].isin(HOTSPOT_LABELS))[["site_id", "grid", "hot"]])
support = pd.concat(support)
per_site = support.groupby("site_id")["hot"].sum().rename("grids_hot").to_frame().join(act)
per_site[per_site.grids_hot > 0].sort_values(["grids_hot", "recent"], ascending=False)
'''),
    md('''
## How similar are the grids?

The Jaccard index of the sets of sites that each pair of grids places in a hotspot
(1 = identical, 0 = no overlap).
'''),
    code('''
sets = {s: set(support[(support.grid == s) & support.hot].site_id) for s in grids}
names = list(grids)
jac = pd.DataFrame(np.nan, index=names, columns=names)
for a, b in itertools.product(names, names):
    u = sets[a] | sets[b]
    jac.loc[a, b] = len(sets[a] & sets[b]) / len(u) if u else np.nan
jac.round(2)
'''),
    md('''
## How to read the comparison

* A signal that appears on most grids does not depend on where the cell borders fall,
  so the choice of grid did not create it.
* A signal that appears on one or two grids only deserves caution: it may come from a
  particular way of grouping neighbouring sites.
* What the grid changes most is **which neighbouring sites share a cell** with the
  site that produces the signal, and therefore the outline and the name of the hotspot
  on the map. It is therefore worth reporting a hotspot together with the sites that
  produce its recent results, not only with the name of the area at the cell's centre.
* In the pipeline H3 is the primary grid; the multi-grid comparison is a sensitivity
  check.
'''),
])

# ---------------------------------------------------------------------------
write('05_live_model.ipynb', [
    md('''
# 5. Run the Bayesian model in your browser

This notebook fits the pipeline's **joint two-period Beta-Binomial model** to the
synthetic H3 data, live, in your browser. It is the same model as in
`pipeline/analyzers/bayesian.py` (`run_two_period_model`).

**Be patient.** The browser cannot use the fast compiled back-ends that PyMC normally
relies on, so sampling runs in pure Python. With the reduced settings below it took
about **6 minutes** in our test (slower computers and browsers take longer); the
production settings (4 chains × 1,500 draws) would take about an hour. On a normal Python installation the same code runs in under a
minute.
'''),
    code('''
import os, sys
if sys.platform == "emscripten":
    # No numba / C compiler in the browser: run PyTensor in pure-Python mode.
    os.environ["PYTENSOR_FLAGS"] = "mode=FAST_COMPILE,linker=py,cxx="
    import piplite
    await piplite.install(["numpy", "scipy", "pandas", "geopandas", "matplotlib", "xarray",
                           "filelock", "setuptools", "cachetools", "cloudpickle", "rich",
                           "threadpoolctl", "typing-extensions", "arviz"])
    # pytensor and pymc pull in numba, which does not exist for the browser.
    await piplite.install("pytensor", deps=False)
    await piplite.install("pymc", deps=False)
'''),
    code('''
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from hotspots_lite import build_info, load_results

CHAINS, DRAWS, TUNE = 2, 400, 400     # production: 4 chains, 1,500 draws, 1,500 tuning steps

info = build_info()
cells = load_results()
d = cells[cells["tests_curr"].fillna(0) > 0].reset_index(drop=True)
y, n = d["recent_curr"].astype(int).to_numpy(), d["tests_curr"].astype(int).to_numpy()
y_h = d["recent_hist"].fillna(0).astype(int).to_numpy()
n_h = d["tests_hist"].fillna(0).astype(int).to_numpy()
has_hist = n_h > 0
K = len(d)

national = y.sum() / n.sum()
prior_mu = float(np.log(np.clip(national, 1e-4, 0.5) / (1 - np.clip(national, 1e-4, 0.5))))

# Testing-intensity normalisation (standardised log test-months, pooled over both windows).
li_c = np.log1p(d["test_months_curr"].fillna(0).to_numpy(float))
li_h = np.log1p(d["test_months_hist"].fillna(0).to_numpy(float))
pool = np.concatenate([li_c, li_h])
z_c, z_h = (li_c - pool.mean()) / pool.std(), (li_h - pool.mean()) / pool.std()
print(f"{K} cells, national recent share {national:.3%}; {has_hist.sum()} cells have a baseline")
'''),
    md('''
## The model

For area *i* and window *t* (baseline or current):

logit *p*<sub>it</sub> = μ<sub>t</sub> + *a*<sub>i</sub> + δ<sub>i</sub>·[t = current] + β·*z*<sub>it</sub>

* μ<sub>t</sub>: national level in each window;
* *a*<sub>i</sub>: the area's own level, pooled across areas;
* δ<sub>i</sub>: the area's change between windows beyond the national change μ<sub>current</sub> − μ<sub>history</sub>, pooled towards zero;
* β·*z*<sub>it</sub>: adjustment for testing effort (test-months), so the trend is not
  driven by sites opening or closing;
* recent counts are Beta-Binomial with a concentration κ learned from the data.
'''),
    code('''
with pm.Model() as model:
    mu_hist = pm.Normal("mu_hist", mu=prior_mu, sigma=1.5)
    mu_curr = pm.Normal("mu_curr", mu=prior_mu, sigma=1.5)
    sigma_a = pm.HalfNormal("sigma_a", sigma=2)
    sigma_delta = pm.HalfNormal("sigma_delta", sigma=1.0)
    a = pm.Deterministic("a", sigma_a * pm.Normal("a_offset", 0, 1, shape=K))
    delta = pm.Deterministic("delta", sigma_delta * pm.Normal("d_offset", 0, 1, shape=K))
    beta_intensity = pm.Normal("beta_intensity", mu=0.0, sigma=1.0)
    p_hist = pm.Deterministic("p_hist", pm.math.invlogit(mu_hist + a + beta_intensity * z_h))
    p_curr = pm.Deterministic("p_curr", pm.math.invlogit(mu_curr + a + delta + beta_intensity * z_c))
    kappa = pm.Gamma("kappa", alpha=2, beta=0.01)
    pm.BetaBinomial("y_obs", alpha=p_curr * kappa, beta=(1 - p_curr) * kappa, n=n, observed=y)
    bb_h = pm.BetaBinomial.dist(alpha=p_hist * kappa, beta=(1 - p_hist) * kappa, n=np.maximum(n_h, 1))
    pm.Potential("y_hist_ll", (pm.logp(bb_h, y_h) * has_hist.astype(float)).sum())

    idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=1, target_accept=0.95,
                      random_seed=42, progressbar=False)

s = az.summary(idata, var_names=["mu_curr", "sigma_a", "sigma_delta", "kappa"])
print(f"max R-hat {s['r_hat'].max():.3f} (needs < 1.01), min bulk ESS {s['ess_bulk'].min():.0f} (needs > 400)")
s
'''),
    md('''
**Check the diagnostics before reading anything else.** With the reduced browser
settings the sampler often stops short of the pipeline's thresholds (R-hat below 1.01,
effective sample size above 400). The pipeline treats such a run as unreliable and
does not write a report. Here we carry on for teaching purposes; for real use, raise
`CHAINS`, `DRAWS` and `TUNE` towards the production values (4, 1,500, 1,500) and run
on a normal Python installation.
'''),
    md('''
## From the posterior to labels

The level compares each area's current share with the national share **without that
area**; the trend reads the change δ directly. Each axis uses the 0.80 rule, raised
when a stricter cutoff keeps the expected share of false calls within 5 %. The presence
and cold-cell gates come last. (The pipeline has one more gate that withholds a trend
when an area's history is too thin; it needs extra inputs and is left out here, so a
few labels can differ from the pipeline's.)
'''),
    code('''
pc = idata.posterior["p_curr"].values.reshape(-1, K)
dl = idata.posterior["delta"].values.reshape(-1, K)
loo = (y.sum() - y) / (n.sum() - n)

exc = {
    "level_high": (pc > loo).mean(0), "level_low": (pc < loo).mean(0),
    "trend_up": (dl > 0).mean(0), "trend_down": (dl < 0).mean(0),
}

def fdr_cutoff(p, start=0.80, max_fdr=0.05, step=0.005, ceiling=0.99):
    """Lowest cutoff >= start at which the Bayesian FDR of the flagged areas is <= 5 %."""
    t = start
    while t <= ceiling:
        sel = p[p > t]
        if len(sel) == 0:
            break
        if np.mean(1 - sel) <= max_fdr:
            return t
        t += step
    return start

flags = {k: v > fdr_cutoff(v) for k, v in exc.items()}
level = np.where(flags["level_high"], "high", np.where(flags["level_low"], "low", "norm"))
trend = np.where(flags["trend_up"], "high", np.where(flags["trend_down"], "low", "norm"))
label_map = {("high", "high"): "Established hotspot", ("high", "norm"): "Emerging hotspot",
             ("high", "low"): "Normal", ("norm", "high"): "Stable high-burden",
             ("norm", "norm"): "Normal", ("norm", "low"): "Below national",
             ("low", "high"): "Declining from high-burden", ("low", "norm"): "Emerging decrease",
             ("low", "low"): "Below national"}
hot = {"Established hotspot", "Emerging hotspot", "Stable high-burden"}
labels = []
for i in range(K):
    if y[i] == 0 and y_h[i] == 0:
        labels.append("Normal")                       # cold-cell gate
        continue
    lab = label_map[(trend[i], level[i])]
    labels.append("Normal" if lab in hot and y[i] < 2 else lab)   # presence gate
d["live_label"] = labels
d["P(level high)"] = exc["level_high"].round(3)
d["P(trend up)"] = exc["trend_up"].round(3)
print(f"Agreement with the precomputed pipeline labels: {(d.live_label == d.label).mean():.0%}")
d[["name", "tests_curr", "recent_curr", "P(level high)", "P(trend up)", "live_label", "label"]].sort_values(
    "P(level high)", ascending=False).head(12)
'''),
    md('''
Differences from the precomputed labels come from four places: fewer posterior draws
here, the omitted thin-history gate, ordinary Monte Carlo noise near the 0.80 cutoff,
and the false-discovery step. That last one can drop a borderline area even above
0.80: if the areas above 0.80 together carry more than 5 % expected false calls, the
cutoff rises until they do not, and the weakest of them falls out. An area whose label flips between runs is, by definition, a borderline case;
the reliability rating and the watch-list are there to show that.
'''),
])
