# -*- coding: utf-8 -*-
"""Threshold calibration by simulation (reviewer's requested study).

Uses per-hex testing volumes from the dataset (the synthetic input by default;
point SIM_DATA at the real file to reproduce the reported numbers). For each
replicate we generate recent counts under a KNOWN ground truth, fit the joint
two-period model, and record the posterior exceedance probabilities on a grid of
thresholds. Because the whole
threshold grid is read from the SAME posterior, one fit per replicate suffices;
the operating characteristics are computed in post-processing.

Ground-truth scenarios:
  A (mixed):  most hexes null (true rate = national in both periods); a few
              designated hexes are localized outbreaks (a current-period rise of
              2x / 3x / 5x national); one hex is a genuine decline; one hex is a
              site closure (current tests = 0).
  B (national shift): EVERY hex rises together (national trend shift, no local
              outbreak) -- tests whether the trend axis falsely flags local rises
              when the whole country moved.

Outputs, per confidence level c and per threshold:
  * RATE axis: false-positive rate (null hexes flagged elevated) and sensitivity
    (outbreak hex of each magnitude flagged) as the SMR ratio cutoff r varies.
  * TREND axis: false-positive rate (national-shift hexes flagged rising) and
    sensitivity (outbreak-rise hex flagged) as the delta cutoff t varies.
Presence gate (>=2 current recent events) is enforced for any hotspot flag.
"""
import os, sys, math
from pathlib import Path
import numpy as np, pandas as pd, h3
ROOT = str(Path(__file__).resolve().parent.parent)   # repo root
sys.path.insert(0, ROOT)
import pymc as pm, pytensor.tensor as pt
from pipeline.standardization import bayesian_fdr_threshold
CLIP_MIN, CLIP_MAX = 0.001, 0.99
# Output directory for the results file; defaults next to this script.
SCR = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent)
# Public-repo-safe default: the synthetic dataset (same structure as the real
# input). The reported 0.80 calibration was confirmed on the REAL site volumes
# too; set SIM_DATA to point at the real file to reproduce those exact numbers.
DATA = os.environ.get('SIM_DATA', f'{ROOT}/data/synthetic_input_data.xlsx')
REPS_A = int(os.environ.get('SIM_REPS_A', '50'))
REPS_B = int(os.environ.get('SIM_REPS_B', '30'))
DRAWS = int(os.environ.get('SIM_DRAWS', '500'))
P_NAT = float(os.environ.get('SIM_PNAT', '0.015'))
R_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]      # SMR (vs national) ratio cutoffs
T_GRID = [1.0, 1.5, 2.0]                # trend (delta) ratio cutoffs
CONF = [0.75, 0.80, 0.85, 0.90, 0.95]   # posterior-probability confidence levels

# ---- real per-hex testing volumes (sliding window Apr2025-May2026 vs prior 12mo)
xl = pd.ExcelFile(DATA); c = xl.parse('hiv_cases')
c['d'] = pd.to_datetime(c['test_date']); t = c['type'].astype(str).str.lower().str.strip()
c['den'] = t.isin(['recent', 'long-term']); c = c[c['den']].copy()
c['hex'] = [h3.latlng_to_cell(a, o, 3) for a, o in zip(c['latitude'], c['longitude'])]
cur = c[(c['d'] >= '2025-04-01') & (c['d'] <= '2026-05-31')]
his = c[(c['d'] >= '2024-04-01') & (c['d'] <= '2025-03-31')]
vol = pd.DataFrame({'n_cur': cur.groupby('hex').size(), 'n_his': his.groupby('hex').size()}).fillna(0).astype(int)
vol = vol[vol['n_cur'] > 0].sort_values('n_cur', ascending=False).reset_index(drop=True)
K = len(vol)
NCUR = vol['n_cur'].to_numpy(); NHIS = np.maximum(vol['n_his'].to_numpy(), 1)
print(f'{K} hexes; current volumes min/median/max = {NCUR.min()}/{int(np.median(NCUR))}/{NCUR.max()}', flush=True)

# assign ground-truth roles (vol is sorted largest-first). The three outbreak
# hexes sit at adjacent mid-size ranks so that only the outbreak MAGNITUDE varies
# between them, not the testing volume; decline and closure use separate ranks.
med = K // 2
OUT2, OUT3, OUT5 = med - 1, med, med + 1            # localized rises of 2x / 3x / 5x, same size tier
DECLINE = K // 4                                    # genuine decline (larger hex)
CLOSURE = min((3 * K) // 4, K - 1)                  # site closure (smaller hex, n_cur -> 0)
OUTBREAKS = {OUT2: 2.0, OUT3: 3.0, OUT5: 5.0}
print(f'outbreak hexes (idx:mag): {OUTBREAKS}; decline={DECLINE}; closure={CLOSURE}', flush=True)

def fit(rc, nc, rh, nh):
    y = rc.astype(int); n = nc.astype(int); y_h = rh.astype(int); n_h = nh.astype(int)
    has_h = n_h > 0; nat = float(y.sum() / max(n.sum(), 1))
    pmu = float(np.log(np.clip(nat, CLIP_MIN, CLIP_MAX) / (1 - np.clip(nat, CLIP_MIN, CLIP_MAX))))
    with pm.Model():
        mh = pm.Normal('mu_hist', pmu, 1.5); mc = pm.Normal('mu_curr', pmu, 1.5)
        sa = pm.HalfNormal('sigma_a', 2); md = pm.Normal('mu_delta', 0, 1.0); sd = pm.HalfNormal('sigma_delta', 1.0)
        ao = pm.Normal('a_offset', 0, 1, shape=K); a = pm.Deterministic('a', sa * ao)
        do = pm.Normal('d_offset', 0, 1, shape=K); delta = pm.Deterministic('delta', md + sd * do)
        ph = pm.math.invlogit(mh + a); pc_ = pm.Deterministic('p_curr', pm.math.invlogit(mc + a + delta))
        kap = pm.Gamma('kappa', 2, 0.01)
        pm.BetaBinomial('y', alpha=pc_ * kap, beta=(1 - pc_) * kap, n=np.maximum(n, 1), observed=y)
        hm = pt.as_tensor_variable(has_h.astype(float)); nhs = pt.as_tensor_variable(np.maximum(n_h, 1))
        bbh = pm.BetaBinomial.dist(alpha=ph * kap, beta=(1 - ph) * kap, n=nhs)
        pm.Potential('hist', (pm.logp(bbh, pt.as_tensor_variable(y_h)) * hm).sum())
        tr = pm.sample(DRAWS, tune=DRAWS, chains=2, cores=1, target_accept=0.9,
                       random_seed=1, progressbar=False)
    return tr.posterior['p_curr'].values.reshape(-1, K), tr.posterior['delta'].values.reshape(-1, K)

def gen(seed, scenario):
    rng = np.random.default_rng(seed)
    p_cur = np.full(K, P_NAT); p_his = np.full(K, P_NAT)
    nc = NCUR.copy()
    if scenario == 'A':
        for idx, m in OUTBREAKS.items():
            p_cur[idx] = m * P_NAT                      # localized rise
        p_his[DECLINE] = 4 * P_NAT; p_cur[DECLINE] = P_NAT
        nc = NCUR.copy(); nc[CLOSURE] = 0               # site closure
    else:  # B: national trend shift -- everyone rises 2x, no local outbreak
        p_cur = np.full(K, 2 * P_NAT)
    rc = rng.binomial(nc, p_cur); rh = rng.binomial(NHIS, p_his)
    return rc, nc, rh, NHIS

def exceedances(pc, dl, rc, nc):
    tot_r, tot_n = rc.sum(), nc.sum()
    loo = np.where((tot_n - nc) > 0, (tot_r - rc) / (tot_n - nc), tot_r / max(tot_n, 1))
    loo = np.maximum(loo, 1e-3)
    exc_smr = {r: np.array([(pc[:, i] > r * loo[i]).mean() for i in range(K)]) for r in R_GRID}
    exc_up = {tt: (dl > math.log(tt)).mean(axis=0) for tt in T_GRID}
    return exc_smr, exc_up

import pickle
PKL = f'{SCR}/sim_recs.pkl'
if os.environ.get('SIM_RELOAD') and os.path.exists(PKL):
    recs = pickle.load(open(PKL, 'rb'))     # re-sweep thresholds without refitting
    print(f'reloaded {len(recs)} replicates from cache', flush=True)
else:
    recs = []  # per (scenario, replicate): dict of exceedance arrays + counts
    for scen, reps in [('A', REPS_A), ('B', REPS_B)]:
        for rep in range(reps):
            rc, nc, rh, nh = gen(1000 * (1 if scen == 'A' else 2) + rep, scen)
            pc, dl = fit(rc, nc, rh, nh)
            exc_smr, exc_up = exceedances(pc, dl, rc, nc)
            recs.append(dict(scen=scen, rc=rc.copy(), nc=nc.copy(), exc_smr=exc_smr, exc_up=exc_up))
            if rep % 10 == 0:
                print(f'  {scen} replicate {rep+1}/{reps} done', flush=True)
    pickle.dump(recs, open(PKL, 'wb'))
print('FITS_DONE', flush=True)

# ---- post-processing: sweep thresholds, tally FP / sensitivity -------------
def flag_rate(scen, r, c_level):
    """rate-axis flags across replicates of a scenario; returns per-hex flag matrix."""
    flags = []
    for rr in [x for x in recs if x['scen'] == scen]:
        e = rr['exc_smr'][r]
        cut = bayesian_fdr_threshold(e, start=c_level, floor=c_level)[0]
        gate = rr['rc'] >= 2
        flags.append((e > cut) & gate)
    return np.array(flags)  # (reps, K)

def flag_trend(scen, tt, c_level):
    flags = []
    for rr in [x for x in recs if x['scen'] == scen]:
        e = rr['exc_up'][tt]
        cut = bayesian_fdr_threshold(e, start=c_level, floor=c_level)[0]
        gate = rr['rc'] >= 2
        flags.append((e > cut) & gate)
    return np.array(flags)

null_mask = np.ones(K, bool)
for idx in list(OUTBREAKS) + [DECLINE, CLOSURE]:
    null_mask[idx] = False

lines = []
lines.append('=== RATE AXIS (vs national) ===')
lines.append(f"{'conf':>5} {'ratio':>6} {'FP/hex/wk':>10} {'sens@2x':>8} {'sens@3x':>8} {'sens@5x':>8}")
for c_level in CONF:
    for r in R_GRID:
        F = flag_rate('A', r, c_level)
        fp = F[:, null_mask].mean()
        s2 = F[:, OUT2].mean(); s3 = F[:, OUT3].mean(); s5 = F[:, OUT5].mean()
        lines.append(f"{c_level:>5.2f} {r:>6.1f} {fp:>10.3f} {s2:>8.2f} {s3:>8.2f} {s5:>8.2f}")
lines.append('')
lines.append('=== TREND AXIS (vs own past) ===')
lines.append(f"{'conf':>5} {'ratio':>6} {'FP_natlshift':>13} {'sens_rise5x':>11}")
for c_level in CONF:
    for tt in T_GRID:
        Fb = flag_trend('B', tt, c_level)          # everyone rose -> any local 'rising' flag is a false positive
        fp = Fb.mean()
        Fa = flag_trend('A', tt, c_level)
        sens = Fa[:, OUT5].mean()
        lines.append(f"{c_level:>5.2f} {tt:>6.1f} {fp:>13.3f} {sens:>11.2f}")
out = '\n'.join(lines)
print(out, flush=True)
open(f'{SCR}/threshold_simulation_results.txt', 'w', encoding='utf-8').write(out)
print('SIM_DONE', flush=True)
