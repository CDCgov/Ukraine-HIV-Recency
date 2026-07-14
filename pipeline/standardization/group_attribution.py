"""
Risk-group attribution of the recency signal (supplementary explanatory analysis).

The detector answers "where and how much"; this module answers the explanatory
question "is the recent-infection signal carried by the key-population (high-risk)
group or by the general (low-risk) population, and did the tested case-mix shift?".

Why this lives OUTSIDE the per-run detector, at the oblast level:
  * Recent events are very sparse (~a dozen nationally per quarter), so a per-hex
    or per-quarter split by risk group is undefined almost everywhere. Composition
    (who is tested), by contrast, is measured on thousands of tests and is precise.
  * So we compute composition finely but pool the recency RATES up to the oblast
    level and over time, escalating the window until each group clears a minimum
    recent-event count (``min_rec``).

Three products, each with its honest power caveat:

1. ``period_composition`` -- did the tested risk-mix change over time? (Very well
   powered: thousands of tests.) Returns the yearly high-risk share, a chi-square
   test of independence, a monotonic-trend test, and the pre/post contrast.

2. ``oblast_group_rates`` -- in each oblast, is the high-risk group's recency rate
   different from the general population's? Two-proportion test per oblast; only
   the few oblasts with enough recent events in BOTH groups are conclusive.

3. ``oblast_standardization`` -- indirect standardization. ``comp_ratio`` (reliable)
   is how much the oblast's testing mix alone raises/lowers expected recency at
   national group rates; ``smr`` (observed / expected) is the excess beyond the mix.
   NOTE: because the national group rates are close, ``comp_ratio`` is ~1 and the
   ``smr`` nearly equals the detector's own "National reference ratio" -- this
   analysis CONFIRMS the detector's level axis rather than adding a new one. The
   genuinely new information is the per-group breakdown in (2).

Methods: Kitagawa (1955) decomposition; indirect standardization / SMR (Breslow &
Day 1987; Rothman, Greenland & Lash, Modern Epidemiology); the pooling-for-power
logic follows small-area estimation (Fay & Herriot 1979).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.aggregation.periods import baseline_months_for


def _prep(cases: pd.DataFrame, group_col: str, high_label: str) -> pd.DataFrame:
    df = cases.copy()
    df['_date'] = pd.to_datetime(df['test_date'])
    df['_high'] = df[group_col].astype(str).str.lower() == high_label.lower()
    df['_rec'] = df['type'].astype(str).str.lower() == 'recent'
    return df


def _group_stats(d: pd.DataFrame) -> Dict[str, Any]:
    """Counts, shares and per-group recency rates for one slice."""
    n = len(d)
    nh = int(d['_high'].sum())
    nl = n - nh
    rh = int(d.loc[d['_high'], '_rec'].sum())
    rl = int(d.loc[~d['_high'], '_rec'].sum())
    return {
        'n': n, 'n_high': nh, 'n_low': nl,
        'rec_high': rh, 'rec_low': rl, 'recent': rh + rl,
        's_high': (nh / n) if n else np.nan,
        'r_high': (rh / nh) if nh else np.nan,
        'r_low': (rl / nl) if nl else np.nan,
        'p_overall': ((rh + rl) / n) if n else np.nan,
    }


def period_composition(cases: pd.DataFrame, group_col: str = 'risk_group',
                       high_label: str = 'high', start_year: int = 2021,
                       split_year: int = 2024) -> Dict[str, Any]:
    """Yearly high-risk testing share + tests of change over time.

    Returns a dict with ``by_year`` (DataFrame), ``chi2``/``chi2_p`` (year x group
    independence), ``trend_r``/``trend_p`` (monotonic trend), and the ``pre``/``post``
    high-risk shares around ``split_year`` with a two-proportion z-test.
    """
    from scipy import stats
    from statsmodels.stats.proportion import proportion_confint, proportions_ztest

    df = _prep(cases, group_col, high_label)
    df = df[df['_date'].dt.year >= start_year]
    df['_year'] = df['_date'].dt.year

    rows, tab = [], []
    for y, d in df.groupby('_year'):
        s = _group_stats(d)
        rows.append({'year': int(y), 'n': s['n'], 'high_pct': 100 * s['s_high'],
                     'recent': s['recent'], 'rec_high': s['rec_high'], 'rec_low': s['rec_low']})
        tab.append([s['n_high'], s['n_low']])
    by_year = pd.DataFrame(rows)
    tab = np.array(tab)

    chi2, chi2_p, _, _ = stats.chi2_contingency(tab)
    hp = tab[:, 0] / tab.sum(1)
    trend_r, trend_p = stats.pearsonr(np.arange(len(hp)), hp)

    pre = df[df['_year'] < split_year]
    post = df[df['_year'] >= split_year]
    cpre, npre = int(pre['_high'].sum()), len(pre)
    cpost, npost = int(post['_high'].sum()), len(post)
    z, pz = proportions_ztest([cpre, cpost], [npre, npost])
    ci_pre = proportion_confint(cpre, npre, method='wilson')
    ci_post = proportion_confint(cpost, npost, method='wilson')

    return {
        'by_year': by_year, 'chi2': float(chi2), 'chi2_p': float(chi2_p),
        'trend_r': float(trend_r), 'trend_p': float(trend_p),
        'pre_pct': 100 * cpre / npre, 'pre_ci': (100 * ci_pre[0], 100 * ci_pre[1]), 'pre_n': npre,
        'post_pct': 100 * cpost / npost, 'post_ci': (100 * ci_post[0], 100 * ci_post[1]), 'post_n': npost,
        'contrast_z': float(z), 'contrast_p': float(pz), 'split_year': split_year,
    }


def oblast_group_rates(cases_with_oblast: pd.DataFrame, oblast_col: str,
                       group_col: str = 'risk_group', high_label: str = 'high',
                       min_rec: int = 10) -> pd.DataFrame:
    """Per-oblast high-risk vs general recency rate, whole pool, with a
    two-proportion test. ``powered`` marks oblasts with >= ``min_rec`` recent
    events in BOTH groups (the only conclusive ones)."""
    from statsmodels.stats.proportion import proportions_ztest

    df = _prep(cases_with_oblast, group_col, high_label)
    rows = []
    for ob, d in df.groupby(oblast_col):
        s = _group_stats(d)
        if s['n_high'] < 5 or s['n_low'] < 5:
            continue
        try:
            z, p = proportions_ztest([s['rec_high'], s['rec_low']], [s['n_high'], s['n_low']])
        except Exception:  # noqa: BLE001 -- degenerate slice
            z, p = np.nan, np.nan
        rows.append({
            'oblast': ob, 'rec_high': s['rec_high'], 'n_high': s['n_high'],
            'r_high_pct': 100 * s['r_high'], 'rec_low': s['rec_low'], 'n_low': s['n_low'],
            'r_low_pct': 100 * s['r_low'],
            'ratio': (s['r_high'] / s['r_low']) if s['r_low'] else np.nan,
            'p_diff': float(p) if p == p else np.nan,
            'powered': bool(s['rec_high'] >= min_rec and s['rec_low'] >= min_rec),
        })
    return pd.DataFrame(rows).sort_values('rec_high', ascending=False).reset_index(drop=True)


def oblast_standardization(cases_with_oblast: pd.DataFrame, oblast_col: str,
                           cur_start: pd.Timestamp, cur_end: pd.Timestamp,
                           group_col: str = 'risk_group', high_label: str = 'high',
                           min_rec: int = 10) -> pd.DataFrame:
    """Indirect standardization per oblast for the current window.

    ``comp_ratio`` = expected-under-oblast-mix / expected-under-national-mix
    (reliable; the mix effect). ``smr`` = observed / expected-under-oblast-mix,
    with a Poisson-exact 95% CI on the observed count. Because national group
    rates are close, ``comp_ratio`` ~ 1 and ``smr`` tracks the detector's
    National reference ratio -- this is a cross-check, not a new axis.
    """
    from scipy import stats

    df = _prep(cases_with_oblast, group_col, high_label)
    cur = df[(df['_date'] >= cur_start) & (df['_date'] <= cur_end)]
    nat = _group_stats(cur)
    rh, rl, s_nat = nat['r_high'], nat['r_low'], nat['s_high']
    p_nat = nat['p_overall']

    def pois_ci(o: int, alpha: float = 0.05) -> Tuple[float, float]:
        lo = stats.chi2.ppf(alpha / 2, 2 * o) / 2 if o > 0 else 0.0
        hi = stats.chi2.ppf(1 - alpha / 2, 2 * o + 2) / 2
        return lo, hi

    rows = []
    for ob, d in cur.groupby(oblast_col):
        s = _group_stats(d)
        if s['n'] == 0:
            continue
        E = s['n_high'] * rh + s['n_low'] * rl        # expected recent count, oblast mix, national rates
        E_nat = s['n'] * p_nat                          # expected if the oblast had the national mix
        if E <= 0:
            continue
        o = s['recent']
        lo, hi = pois_ci(o)
        rows.append({
            'oblast': ob, 'N': s['n'], 'high_pct': 100 * s['s_high'],
            'observed': o, 'expected': round(E, 2),
            'comp_ratio': round(E / E_nat, 3) if E_nat > 0 else np.nan,
            'smr': round(o / E, 3), 'smr_lo': round(lo / E, 3), 'smr_hi': round(hi / E, 3),
            'powered': bool(o >= min_rec),
        })
    return pd.DataFrame(rows).sort_values('smr', ascending=False).reset_index(drop=True)


def escalating_group_rates(cases: pd.DataFrame, cur_start: pd.Timestamp,
                           group_col: str = 'risk_group', high_label: str = 'high',
                           min_rec: int = 10, start_year: int = 2021) -> pd.DataFrame:
    """National per-group recency rates at escalating temporal aggregation.

    Reports each yearly window's rates and whether it clears ``min_rec`` recent
    events in each group; the caller escalates (doubles) the window for segments
    that do not. Returns one row per (level, segment) with an ``adequate`` flag.
    """
    df = _prep(cases, group_col, high_label)
    end_year = int(df['_date'].dt.year.max())
    cur_end_str = pd.Timestamp(cur_start).strftime('%Y-%m')

    def seg(a: str, b: str, label: str, level: str) -> Dict[str, Any]:
        d = df[(df['_date'] >= a) & (df['_date'] <= b)]
        s = _group_stats(d)
        return {'level': level, 'segment': label, 'N': s['n'],
                'high_pct': round(100 * s['s_high'], 1) if s['n'] else np.nan,
                'rec_high': s['rec_high'], 'rec_low': s['rec_low'],
                'r_high_pct': round(100 * s['r_high'], 2) if s['n_high'] else np.nan,
                'r_low_pct': round(100 * s['r_low'], 2) if s['n_low'] else np.nan,
                'adequate': bool(s['rec_high'] >= min_rec and s['rec_low'] >= min_rec)}

    rows = [seg(f'{y}-01-01', f'{y}-12-31', str(y), 'yearly')
            for y in range(start_year, end_year + 1)]
    # 2-year pools
    y = start_year
    while y <= end_year:
        rows.append(seg(f'{y}-01-01', f'{min(y + 1, end_year)}-12-31', f'{y}-{min(y + 1, end_year)}', '2-year'))
        y += 2
    rows.append(seg(f'{start_year}-01-01', f'{end_year}-12-31', f'{start_year}-{end_year}', 'whole'))
    return pd.DataFrame(rows)
