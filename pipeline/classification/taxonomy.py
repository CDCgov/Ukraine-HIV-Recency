"""
Hotspot classification taxonomy.

Two routines live here:

``classify_with_exceedance`` — the legacy single-axis classifier that maps
``exceedance_prob = P(theta > baseline | data)`` to five labels (Obvious /
Slight Increase, No Difference, Slight / Obvious Decrease). It is kept
because some downstream code paths still rely on the older labels and the
function is the natural fallback when ``classification_smr_sir`` is not
present.

``classify_with_smr_sir`` — the two-dimensional taxonomy that combines
SMR (current vs national-current) and SIR (current vs EB-shrunken local
history adjusted for the national trend). The cross of those two axes
yields seven labels that separate a fresh rise (early signal) from an
endemically high level (sustained burden) and from a wind-down — situations
that call for different programmatic responses.

The module also exposes the canonical hotspot set used by ``is_hotspot``
(applied by reporting and recommendation code to mask DataFrames uniformly)
and ``add_smr_sir_counts`` which mirrors the legacy single-axis counters
with directly-comparable SIR/SMR-aware versions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# SIR/SMR taxonomy labels that count as a "hotspot" in summaries and
# recommendations.
HOTSPOT_LABELS = frozenset({
    'Established hotspot',  # SIR_high and SMR_high
    'Emerging hotspot',     # SIR_high, SMR not yet high
    'Stable high-burden',   # SMR_high without recent rise
})


# Full SIR/SMR label set with stable diagnostics keys.
SMR_SIR_LABELS = (
    ('Established hotspot',        'established_hotspot'),
    ('Emerging hotspot',           'emerging_hotspot'),
    ('Stable high-burden',         'stable_high_burden'),
    ('Declining from high-burden', 'declining_from_high_burden'),
    ('Emerging decrease',          'emerging_decrease'),
    ('Significant decrease',       'significant_decrease'),
    ('Normal',                     'normal'),
)


def classify_with_exceedance(row: pd.Series, threshold: float = 0.95) -> str:
    """Classify a territory by its one-axis exceedance probability.

    Thresholds:
        - > threshold:  "Obvious Increase" (FDR-controlled via ``bayesian_fdr_threshold``)
        - > 0.80:       "Slight Increase"
        - < 0.05:       "Obvious Decrease"
        - < 0.20:       "Slight Decrease"
        - else:         "No Difference"

    Returns ``"No Data"`` if the window contained no tests for the territory.
    """
    if 'all_tested_curr' in row and row['all_tested_curr'] == 0:
        return "No Data"

    if 'exceedance_prob' in row and not pd.isna(row['exceedance_prob']):
        exc_prob = row['exceedance_prob']
        if exc_prob > threshold:
            return "Obvious Increase"
        elif exc_prob > 0.80:
            return "Slight Increase"
        elif exc_prob < 0.05:
            return "Obvious Decrease"
        elif exc_prob < 0.20:
            return "Slight Decrease"
        else:
            return "No Difference"

    return "No Data"


def classify_with_smr_sir(row: pd.Series,
                          cutoff_smr_high: float,
                          cutoff_sir_high: float,
                          cutoff_smr_low: float,
                          cutoff_sir_low: float) -> str:
    """Two-dimensional SIR x SMR taxonomy on FDR-controlled exceedance probs.

    Each axis is evaluated independently against its own FDR cut-off, so the
    "elevated" call on SMR and the "elevated" call on SIR each carry their
    own discovery budget. The cross of the three states per axis (high /
    normal / low) is collapsed to seven semantically distinct labels.
    """
    if 'all_tested_curr' in row and row['all_tested_curr'] == 0:
        return "No Data"

    smr_high = float(row.get('exc_prob_smr', 0.0)) > cutoff_smr_high
    sir_high = float(row.get('exc_prob_sir', 0.0)) > cutoff_sir_high
    smr_low = float(row.get('exc_prob_smr_low', 0.0)) > cutoff_smr_low
    sir_low = float(row.get('exc_prob_sir_low', 0.0)) > cutoff_sir_low

    smr_state = 'high' if smr_high else ('low' if smr_low else 'norm')
    sir_state = 'high' if sir_high else ('low' if sir_low else 'norm')

    label_map = {
        ('high', 'high'): "Established hotspot",
        ('high', 'norm'): "Emerging hotspot",
        ('high', 'low'):  "Normal",   # rises but still below national
        ('norm', 'high'): "Stable high-burden",
        ('norm', 'norm'): "Normal",
        ('norm', 'low'):  "Significant decrease",
        ('low',  'high'): "Declining from high-burden",
        ('low',  'norm'): "Emerging decrease",
        ('low',  'low'):  "Significant decrease",
    }
    return label_map.get((sir_state, smr_state), "Normal")


def is_hotspot(df_in: pd.DataFrame) -> pd.Series:
    """Boolean mask of hotspot rows.

    Prefers ``classification_smr_sir`` when the taxonomy is available and
    falls back to the legacy ``classification == 'Obvious Increase'`` so
    older single-axis outputs keep behaving as before.
    """
    if 'classification_smr_sir' in df_in.columns:
        return df_in['classification_smr_sir'].isin(HOTSPOT_LABELS)
    return df_in['classification'] == 'Obvious Increase'


def add_watchlist(df: pd.DataFrame, top_burden_frac: float = 0.80,
                  rate_pctile: float = 0.80) -> pd.DataFrame:
    """Combined burden + rate "watch-list" ranking (additive — it does not
    alter the SMR/SIR classification or reliability).

    Recency hotspot detection on the single rate axis is underpowered when the
    recent-event count per territory is tiny: the posterior rate is dominated
    by the global level and almost nothing clears the FDR floor, so genuinely
    high-burden centres (many recent infections but a near-average *rate*) stay
    invisible. To keep the output useful for field triage this adds two
    triage axes and a combined ranking:

      * **burden** — ``recent_count_curr`` as a share of the level-wide recent
        total. ``burden_high`` marks the territories that together carry the
        top ``top_burden_frac`` of the recent caseload (data-adaptive, no fixed
        cut-off, so it tracks the falling recency rate over time).
      * **rate** — *relative* elevation: ``burden``-independent, ``rate_high``
        marks territories whose posterior SMR sits in the top
        ``1 - rate_pctile`` of the active distribution, OR that are already an
        FDR-flagged hotspot. This is a ranking for triage, **not** a
        significance test — the rigorous FDR-controlled call stays in
        ``classification`` and is left untouched. The relative flag surfaces
        the highest-rate territories (e.g. a 2.6x oblast on 4 events) that the
        FDR floor cannot confirm.

    A territory lands on the watch-list (``on_watchlist``) when it is notable
    on EITHER axis (OR semantics, recorded in ``watch_reason`` as
    ``'burden'`` / ``'rate'`` / ``'both'``). ``watch_rank`` (1 = highest
    priority) is the best (smallest) of the two per-axis ranks, so high-burden
    centres and relatively-elevated communities both surface. Territories with
    no current tests are excluded (NaN rank).
    """
    if len(df) == 0:
        return df

    rc = pd.to_numeric(df.get('recent_count_curr'), errors='coerce').fillna(0.0)
    tested = pd.to_numeric(df.get('all_tested_curr'), errors='coerce').fillna(0.0)
    active = tested > 0
    total_recent = float(rc.where(active, 0.0).sum())

    # Burden axis: share of the level-wide recent caseload.
    if total_recent > 0:
        df['burden_share_pct'] = (rc.where(active, 0.0) / total_recent * 100.0).round(2)
    else:
        df['burden_share_pct'] = 0.0

    # Cumulative-share membership: sort active territories by count (desc) and
    # keep each whose *preceding* cumulative share is still below the fraction,
    # so the territory that crosses the line is included.
    burden_high = pd.Series(False, index=df.index)
    if total_recent > 0:
        order = rc.where(active, 0.0).sort_values(ascending=False)
        prev_cum = (order.cumsum() - order) / total_recent
        burden_high.loc[order.index] = (prev_cum < top_burden_frac) & (order > 0)
    df['burden_high'] = burden_high

    # Rate axis (relative triage, NOT a significance test): top (1 - rate_pctile)
    # of the active posterior-SMR distribution, OR an FDR-flagged hotspot (so a
    # rigorous call is never dropped from the list). Rank on the posterior
    # MEDIAN when available -- more robust than the tail-inflated mean on the
    # skewed SMR posteriors of sparse-count territories.
    _smr_col = 'smr_median' if 'smr_median' in df.columns else 'smr_mean'
    smr = pd.to_numeric(df.get(_smr_col), errors='coerce')
    smr_active = smr.where(active)
    if int(smr_active.notna().sum()) > 0:
        cut = float(smr_active.quantile(rate_pctile))
        pct_high = (smr_active >= cut) & active
    else:
        pct_high = pd.Series(False, index=df.index)
    label_col = 'classification_smr_sir' if 'classification_smr_sir' in df.columns else (
        'classification' if 'classification' in df.columns else None)
    fdr_high = (df[label_col].isin(HOTSPOT_LABELS) & active) if label_col is not None \
        else pd.Series(False, index=df.index)
    df['rate_high'] = pct_high | fdr_high

    # Two transparent per-axis ranks (1 = top), over active territories only,
    # so a reader sees a territory's standing on each dimension directly.
    df['burden_rank'] = rc.where(active).rank(ascending=False, method='min')
    df['rate_rank'] = smr_active.rank(ascending=False, method='min')

    df['on_watchlist'] = df['burden_high'] | df['rate_high']

    # Priority *within the watch-list* (1 = top); territories not on the list
    # get no rank (NaN), so the column is a clean 1..N_listed ordering and
    # sorting by it yields the watch-list in priority order. Priority = best
    # (smallest) rank on EITHER axis, ties broken by the rank sum so a
    # territory strong on both axes outranks one strong on a single axis.
    best = df[['burden_rank', 'rate_rank']].min(axis=1)
    tie = df['burden_rank'].fillna(1e9) + df['rate_rank'].fillna(1e9)
    order_key = (best * 1e6 + tie).where(df['on_watchlist'])
    df['watch_rank'] = order_key.rank(method='min')

    df['watch_reason'] = np.select(
        [df['burden_high'] & df['rate_high'], df['burden_high'], df['rate_high']],
        ['both', 'burden', 'rate'],
        default='',
    )
    return df


def add_smr_sir_counts(diagnostics: dict, df_in: pd.DataFrame, total: int,
                       include_n: bool = False) -> None:
    """Mirror legacy single-axis counters with SIR/SMR-aware versions.

    For each of the seven categories writes ``pct_<key>`` (and optionally
    ``n_<key>``) into ``diagnostics``. Silent no-op when the frame lacks the
    ``classification_smr_sir`` column or has no rows.
    """
    if 'classification_smr_sir' not in df_in.columns or total == 0:
        return
    counts = df_in['classification_smr_sir'].value_counts()
    for label, key in SMR_SIR_LABELS:
        n = int(counts.get(label, 0))
        diagnostics['pct_' + key] = float(n / total * 100)
        if include_n:
            diagnostics['n_' + key] = n
