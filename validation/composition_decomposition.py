# -*- coding: utf-8 -*-
"""
Kitagawa decomposition of the recent-infection trend (audit D1, supplementary).

The proportion of recent infections among newly-diagnosed declined over the
project. Part of that could be an artefact of the testing case-mix shifting
(the share of key-population clients fell), part is a genuine drop in the
recent fraction within risk groups. This script separates the two with a
Kitagawa (1955) decomposition:

    P = s_high * r_high + s_low * r_low          (overall recent fraction)
    dP = composition effect + rate effect
      composition = sum_g (s_g2 - s_g1) * (r_g1 + r_g2)/2
      rate        = sum_g (r_g2 - r_g1) * (s_g1 + s_g2)/2

where s_g is the share of tests in risk group g and r_g its recent fraction.
The composition effect is the change attributable purely to who was tested;
the rate effect is the change in the within-group recent fractions.

It is the evidence behind the audit conclusion that a baked-in risk-group
standardisation is unnecessary: the composition shift explains only a small
part of the decline, so the crude results are robust to it. Closed-form,
runs in a second.

Run:
    python validation/composition_decomposition.py
    python validation/composition_decomposition.py config.json --group-col risk_group
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Project root on sys.path so this script runs from the validation/ subfolder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.constants import DEFAULT_CONFIG


def period_stats(df: pd.DataFrame, group_col: str, high_label: str):
    """Return (share_high, rate_high, rate_low, P, n) for one slice of data."""
    n = len(df)
    if n == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), 0
    is_high = (df[group_col].astype(str).str.lower() == high_label)
    s_high = float(is_high.mean())
    rec = (df['type'].astype(str).str.lower() == 'recent')
    r_high = float(rec[is_high].mean()) if is_high.any() else 0.0
    r_low = float(rec[~is_high].mean()) if (~is_high).any() else 0.0
    P = s_high * r_high + (1 - s_high) * r_low
    return s_high, r_high, r_low, P, n


def decompose(s1, rh1, rl1, s2, rh2, rl2):
    """Kitagawa split of dP into composition and rate effects (2 groups)."""
    comp = (s2 - s1) * (rh1 + rh2) / 2 + ((1 - s2) - (1 - s1)) * (rl1 + rl2) / 2
    rate = (rh2 - rh1) * (s1 + s2) / 2 + (rl2 - rl1) * ((1 - s1) + (1 - s2)) / 2
    return comp, rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Kitagawa decomposition of the recency trend")
    parser.add_argument('config', nargs='?', default='config.json')
    parser.add_argument('--group-col', default='risk_group', help='Risk-group column (default risk_group)')
    parser.add_argument('--high-label', default='high', help='Value marking the high-risk group (default "high")')
    args = parser.parse_args()

    config = (json.load(open(args.config, encoding='utf-8'))
              if Path(args.config).exists() else dict(DEFAULT_CONFIG))
    df = pd.read_excel(config['excel_path'], sheet_name='hiv_cases')
    df['test_date'] = pd.to_datetime(df['test_date'])
    df['year'] = df['test_date'].dt.year

    years = sorted(y for y in df['year'].dropna().unique())

    print("=" * 92)
    print("Recent-infection trend: composition (case-mix) vs rate, by calendar year")
    print(f"high-risk group = '{args.high_label}' in column '{args.group_col}'")
    print("=" * 92)
    print(f"{'year':>6}{'n':>8}{'high share':>12}{'%recent':>10}"
          f"{'%rec high':>11}{'%rec low':>10}")
    print("-" * 92)
    stats = {}
    for y in years:
        s, rh, rl, P, n = period_stats(df[df['year'] == y], args.group_col, args.high_label)
        stats[y] = (s, rh, rl, P, n)
        print(f"{int(y):>6}{n:>8}{100*s:>11.1f}%{100*P:>9.2f}%{100*rh:>10.2f}%{100*rl:>9.2f}%")

    # Decompositions for informative period pairs (only where both years exist).
    pairs = [(2021, 2026), (2023, 2025), (2023, 2026), (2024, 2026)]
    pairs = [(a, b) for a, b in pairs if a in stats and b in stats]

    print("\n" + "=" * 92)
    print("Kitagawa decomposition of the change in %recent between year pairs")
    print("=" * 92)
    print(f"{'pair':>14}{'dP (pp)':>10}{'composition':>16}{'rate (within-group)':>22}")
    print("-" * 92)
    for a, b in pairs:
        s1, rh1, rl1, P1, _ = stats[a]
        s2, rh2, rl2, P2, _ = stats[b]
        comp, rate = decompose(s1, rh1, rl1, s2, rh2, rl2)
        dP = P2 - P1
        comp_pct = 100 * comp / dP if dP != 0 else float('nan')
        rate_pct = 100 * rate / dP if dP != 0 else float('nan')
        print(f"{f'{a}->{b}':>14}{100*dP:>+10.2f}"
              f"{100*comp:>+10.2f} ({comp_pct:>3.0f}%)"
              f"{100*rate:>+13.2f} ({rate_pct:>3.0f}%)")
    print("-" * 92)
    print("Interpretation: 'composition' is the part of the change explained purely by the\n"
          "shift in who was tested; 'rate' is the genuine change within risk groups. A small\n"
          "composition share means the decline is real, not a testing-mix artefact, so the\n"
          "crude (unstandardised) results are robust to the case-mix shift.")


if __name__ == '__main__':
    main()
