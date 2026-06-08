"""
Reliability scoring for hotspot analysis results.

The historical 40 / 30 / 30 weighted blend of data-adequacy, sample-size and
model-quality scores double-counted sample size (it appeared directly *and*
indirectly through the posterior width) and was insensitive to the
quantity that actually matters for an epidemiologist's decision: how tight
the posterior of the per-territory rate is. The current implementation
replaces the blend with:

* A **hard convergence gate** — if the sampler did not produce a usable
  posterior the score is ``NaN`` and the category is the explicit
  ``UNRELIABLE`` marker. Three failure modes trip the gate:

    - ``convergence_fatal`` (more than 5% divergences after all adaptive
      attempts);
    - ``R-hat >= 1.01`` anywhere (chains have not mixed);
    - minimum effective sample size below 400 (Vehtari et al. 2021
      stability threshold for credible-interval quantiles).

* A **CV-driven score** ``100 * exp(-CV)`` computed from the 95% credible
  interval of the SMR (with a fallback on the raw ``p`` posterior). The
  exponential map is smooth and monotone, so a 20% relative posterior
  width lands around 82, a doubling at 1.0 lands around 37, and very wide
  posteriors collapse smoothly to 0. Because ``SMR = p / national_rate``
  and the national rate is a constant per window, ``CV(SMR) = CV(p)``.

  The 80 / 60 HIGH/MODERATE cut-offs were checked by a calibration study
  (``validation/reliability_calibration.py``, audit M5): the score rises
  monotonically with sample size and HIGH estimates are decisive about the
  SMR > 2 hotspot threshold far more often than LOW ones (~0.60 vs ~0.10),
  so the ordering is sound and the cut-offs are kept. At the project's
  national rate (~2%) the recency counts are tiny, so reaching HIGH
  effectively needs hundreds of tests per territory; most territories are
  intrinsically LOW reliability -- a property of the data, not the cut-offs.

* Two **adjustments** kept from the previous design that capture
  identifiable failure modes a CV cannot see:

    - imputation penalty when the risk-group composition was filled in
      from the national mean rather than observed;
    - current/historical-tests ratio penalty when the analysis window is
      sparsely tested relative to baseline (the posterior is tight only
      because the prior dominates locally).

Component scores (``data_adequacy_score``, ``sample_size_score``,
``model_quality_score``) are still computed and emitted so the reliability
report can break the number down, but they no longer feed into the
overall.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class ReliabilityScoreCalculator:
    """Reliability calculator (hard gate + CV-based per-territory score)."""

    @staticmethod
    def calculate_data_adequacy_score(df: pd.DataFrame) -> Tuple[float, str]:
        """Aggregate data-adequacy score over active sites only.

        Active sites are detected via ``site_present`` when present, and
        fall back to ``all_tested_curr > 0``. Structural zeros are excluded
        because they would otherwise dominate the sparsity ratio.

        Thresholds follow CDC (2019) and Lawson (2018): < 30% zeros is
        excellent, 30-50% good, 50-70% moderate, > 70% poor.

        A graduated imputation penalty (5 / 10 / 15 / 20 points by
        percentile of imputed ``proportion_high_risk``) is applied last.
        """
        if 'site_present' in df.columns:
            df_active = df[df['site_present'] == True].copy()
        else:
            df_active = df[df['all_tested_curr'] > 0].copy()
        if len(df_active) == 0:
            return 0, "No active testing sites"

        pct_zeros = (df_active['recent_count_curr'] == 0).sum() / len(df_active) * 100

        if pct_zeros < 30:
            score, interpretation = 100, "Excellent data coverage"
        elif pct_zeros < 50:
            score, interpretation = 70, "Good data coverage"
        elif pct_zeros < 70:
            score, interpretation = 40, "Moderate data sparsity"
        else:
            score, interpretation = 10, "High data sparsity"

        n_active = len(df_active)
        n_total = len(df)
        interpretation += f" ({n_active}/{n_total} active sites)"

        if 'imputed_proportion_high_risk' in df_active.columns:
            pct_imputed = df_active['imputed_proportion_high_risk'].sum() / len(df_active) * 100
            if pct_imputed > 0:
                if pct_imputed < 25:
                    penalty = 5
                elif pct_imputed < 50:
                    penalty = 10
                elif pct_imputed < 75:
                    penalty = 15
                else:
                    penalty = 20
                score = max(0, score - penalty)
                interpretation += f" (imputed risk data: {pct_imputed:.1f}%, penalty: -{penalty})"

        return score, interpretation

    @staticmethod
    def calculate_sample_size_score(df: pd.DataFrame) -> Tuple[float, str]:
        """Aggregate sample-size score over active sites only.

        Thresholds follow Gelman & Hill (2007) and UNAIDS (2019):
        n >= 100 excellent, n >= 50 good, n >= 20 adequate, else small.
        """
        if 'site_present' in df.columns:
            df_active = df[df['site_present'] == True].copy()
        else:
            df_active = df[df['all_tested_curr'] > 0].copy()
        if len(df_active) == 0:
            return 0, "No active testing sites"

        avg_tests = df_active['all_tested_curr'].mean()
        if avg_tests >= 100:
            score, interpretation = 100, "Excellent sample size"
        elif avg_tests >= 50:
            score, interpretation = 80, "Good sample size"
        elif avg_tests >= 20:
            score, interpretation = 60, "Adequate sample size"
        else:
            score, interpretation = 30, "Small sample size"
        interpretation += f" (avg={avg_tests:.1f} tests/site)"
        return score, interpretation

    @staticmethod
    def calculate_territory_data_adequacy_score(row: pd.Series) -> Tuple[float, str]:
        """Per-territory data-adequacy score (kept as informational column)."""
        n_curr = row.get('all_tested_curr', 0)
        n_hist = row.get('all_tested_hist', 0)

        if n_curr == 0:
            return 0, "No current tests"

        if n_curr >= 50:
            score, interpretation = 100, "Excellent data coverage"
        elif n_curr >= 20:
            score, interpretation = 70, "Good data coverage"
        elif n_curr >= 10:
            score, interpretation = 40, "Moderate data sparsity"
        else:
            score, interpretation = 10, "High data sparsity"

        if n_hist > 0:
            ratio = n_curr / n_hist
            if ratio < 0.3:
                score = max(0, score - 20)
                interpretation += f" (current/historical ratio: {ratio:.2f})"
            elif ratio < 0.5:
                score = max(0, score - 10)
                interpretation += f" (current/historical ratio: {ratio:.2f})"

        if 'imputed_proportion_high_risk' in row.index and row['imputed_proportion_high_risk']:
            score = max(0, score - 10)
            interpretation += " (imputed risk data)"

        return score, interpretation

    @staticmethod
    def calculate_territory_sample_size_score(n_tests: int) -> Tuple[float, str]:
        """Per-territory sample-size score (kept as informational column)."""
        if n_tests >= 100:
            score, interpretation = 100, "Excellent sample size"
        elif n_tests >= 50:
            score, interpretation = 80, "Good sample size"
        elif n_tests >= 20:
            score, interpretation = 60, "Adequate sample size"
        else:
            score, interpretation = 30, "Small sample size"
        interpretation += f" (n={n_tests} tests)"
        return score, interpretation

    @staticmethod
    def calculate_model_quality_score(diagnostics: Dict[str, Any]) -> Tuple[float, str]:
        """Bayesian model-quality score from convergence / ESS / divergences / coverage / LOO.

        Two short-circuits override everything else: ``convergence_fatal``
        forces 0 (posterior geometry unhealthy), and the legacy
        ``convergence_ok == 'No'`` caps the score at 20.
        """
        if diagnostics.get('convergence_fatal', False):
            return (0, "FATAL: Posterior geometry unhealthy - results completely unreliable")
        if diagnostics.get('convergence_ok') == 'No':
            return (20, "Poor model convergence - results unreliable")

        if 'convergence_ok' in diagnostics:
            checks = [
                diagnostics.get('convergence_ok') == 'Yes',
                diagnostics.get('ess_adequate') == 'Yes',
                diagnostics.get('divergences_ok') == 'Yes',
                diagnostics.get('ci_coverage_ok') == 'Yes',
            ]
            if diagnostics.get('loo_ok') is not None and diagnostics.get('loo_ok') != 'Unknown':
                checks.append(diagnostics.get('loo_ok') == 'Yes')

            n_passed = sum(checks)
            n_total = len(checks)

            if n_total == 5:
                if n_passed >= 5:
                    score, interpretation = 100, "Excellent model convergence and predictive accuracy"
                elif n_passed >= 4:
                    score, interpretation = 80, "Good model convergence and predictive accuracy"
                elif n_passed >= 3:
                    score, interpretation = 60, "Acceptable model convergence"
                else:
                    score, interpretation = 30, "Poor model convergence"
            else:
                if n_passed >= 4:
                    score, interpretation = 100, "Excellent model convergence"
                elif n_passed >= 3:
                    score, interpretation = 70, "Good model convergence"
                elif n_passed >= 2:
                    score, interpretation = 40, "Acceptable model convergence"
                else:
                    score, interpretation = 20, "Poor model convergence"
        else:
            score, interpretation = 50, "Model quality unknown"

        return score, interpretation

    @staticmethod
    def calculate_overall_score(df: pd.DataFrame, diagnostics: Dict[str, Any],
                                cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Aggregate report-level reliability summary.

        Kept for the Excel / log summary blocks that emit a single number
        per run. Per-territory scoring lives in
        :meth:`calculate_territory_scores`.
        """
        data_score, data_interp = ReliabilityScoreCalculator.calculate_data_adequacy_score(df)
        sample_score, sample_interp = ReliabilityScoreCalculator.calculate_sample_size_score(df)
        model_score, model_interp = ReliabilityScoreCalculator.calculate_model_quality_score(diagnostics)

        if cfg and 'reliability_weights' in cfg:
            weights = cfg['reliability_weights']
            w_data = weights.get('data_adequacy', 40) / 100.0
            w_sample = weights.get('sample_size', 30) / 100.0
            w_model = weights.get('model_quality', 30) / 100.0
        else:
            w_data, w_sample, w_model = 0.40, 0.30, 0.30

        overall_score = w_data * data_score + w_sample * sample_score + w_model * model_score

        if overall_score >= 80:
            rating, flag = "HIGH", "[OK]"
            recommendation = "Results are reliable for decision-making"
        elif overall_score >= 60:
            rating, flag = "MODERATE", "[WARN]"
            recommendation = "Results are acceptable but interpret with caution"
        else:
            rating, flag = "LOW", "[WARN]"
            recommendation = "Results have high uncertainty - use with caution"

        return {
            'overall_score': round(overall_score, 1),
            'rating': rating,
            'flag': flag,
            'recommendation': recommendation,
            'components': {
                'data_adequacy': {'score': data_score, 'weight': int(w_data * 100), 'interpretation': data_interp},
                'sample_size':   {'score': sample_score, 'weight': int(w_sample * 100), 'interpretation': sample_interp},
                'model_quality': {'score': model_score, 'weight': int(w_model * 100), 'interpretation': model_interp},
            },
        }

    @staticmethod
    def calculate_territory_scores(df: pd.DataFrame, diagnostics: Dict[str, Any],
                                   cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Per-territory reliability (hard gate + CV-driven score).

        Writes ``reliability_score``, ``reliability_category``,
        ``data_adequacy_score``, ``sample_size_score``,
        ``model_quality_score`` and ``reliability_cv`` back to ``df``.
        """
        model_score, _ = ReliabilityScoreCalculator.calculate_model_quality_score(diagnostics)

        try:
            _rhat = float(diagnostics.get('rhat_max') or 0.0)
        except (TypeError, ValueError):
            _rhat = 0.0
        _ess_raw = diagnostics.get('ess_alpha_min', diagnostics.get('min_ess_bulk'))
        try:
            _ess = float(_ess_raw) if _ess_raw is not None else float('inf')
        except (TypeError, ValueError):
            _ess = float('inf')
        if (bool(diagnostics.get('convergence_fatal', False))
                or _rhat >= 1.01 or _ess < 400):
            df['reliability_score'] = np.nan
            df['reliability_category'] = "UNRELIABLE"
            df['data_adequacy_score'] = np.nan
            df['sample_size_score'] = np.nan
            df['model_quality_score'] = model_score
            df['reliability_cv'] = np.nan
            return df

        # Run-level divergence downgrade (audit M1). Divergences are a property
        # of the whole fit, not a single territory: > 5% already trips the hard
        # gate above (convergence_fatal -> UNRELIABLE). Here 1-5% caps every
        # territory at MODERATE and 0-1% applies a small penalty, so a fit with
        # any divergences is never reported as fully reliable.
        pct_div = float(diagnostics.get('pct_divergences') or 0.0)

        reliability_scores = []
        reliability_categories = []
        data_adequacy_scores = []
        sample_size_scores = []
        reliability_cvs = []

        for _, row in df.iterrows():
            if row['all_tested_curr'] == 0:
                reliability_scores.append(np.nan)
                reliability_categories.append("NO_DATA")
                data_adequacy_scores.append(np.nan)
                sample_size_scores.append(np.nan)
                reliability_cvs.append(np.nan)
                continue

            data_score, _ = ReliabilityScoreCalculator.calculate_territory_data_adequacy_score(row)
            sample_score, _ = ReliabilityScoreCalculator.calculate_territory_sample_size_score(
                row['all_tested_curr'])

            _smr = row.get('smr_mean')
            _lo, _hi = row.get('smr_lower'), row.get('smr_upper')
            if _smr is None or pd.isna(_smr) or _smr == 0:
                _smr = row.get('predicted_prob')
                _lo, _hi = row.get('prob_lower'), row.get('prob_upper')
            try:
                _denom = 3.92 * abs(float(_smr))
                _cv = abs(float(_hi) - float(_lo)) / _denom if _denom > 0 else float('nan')
            except (TypeError, ValueError):
                _cv = float('nan')

            if pd.isna(_cv) or _cv < 0:
                overall = float('nan')
            else:
                overall = 100.0 * float(np.exp(-_cv))

            if row.get('imputed_proportion_high_risk', False) and not pd.isna(overall):
                overall = max(0.0, overall - 10.0)

            _nh = row.get('all_tested_hist') or 0
            try:
                _nh = float(_nh)
            except (TypeError, ValueError):
                _nh = 0.0
            if _nh > 0 and not pd.isna(overall):
                _r = row['all_tested_curr'] / _nh
                if _r < 0.3:
                    overall = max(0.0, overall - 20.0)
                elif _r < 0.5:
                    overall = max(0.0, overall - 10.0)

            # Run-level divergence cap (M1): 1-5% divergences -> at most MODERATE
            # (score < 80); 0-1% -> small penalty.
            if not pd.isna(overall):
                if pct_div > 1.0:
                    overall = min(overall, 79.9)
                elif pct_div > 0.0:
                    overall = max(0.0, overall - 5.0)

            if not pd.isna(overall):
                overall = max(0.0, min(100.0, overall))

            if pd.isna(overall):
                category = "UNKNOWN"
            elif overall >= 80:
                category = "HIGH"
            elif overall >= 60:
                category = "MODERATE"
            else:
                category = "LOW"

            reliability_scores.append(overall)
            reliability_categories.append(category)
            data_adequacy_scores.append(data_score)
            sample_size_scores.append(sample_score)
            reliability_cvs.append(_cv)

        df['reliability_score'] = reliability_scores
        df['reliability_category'] = reliability_categories
        df['data_adequacy_score'] = data_adequacy_scores
        df['sample_size_score'] = sample_size_scores
        df['model_quality_score'] = model_score
        df['reliability_cv'] = reliability_cvs

        return df
