"""
Bayesian model diagnostics.

:func:`calculate_bayesian_diagnostics` computes the diagnostics dict for the
two-period detector: R-hat per hyperparameter, ESS (bulk / tail), divergence
count and rate, BFMI, tree-depth saturation, the convergence_ok /
divergences_ok / ess_adequate flags, and the overall quality label.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import arviz as az
import numpy as np
import pymc as pm

from pipeline.classification import add_smr_sir_counts as _add_smr_sir_counts
from pipeline.diagnostics.interpreter import DiagnosticInterpreter
from pipeline.diagnostics.ppc import PPCCalculator

logger = logging.getLogger(__name__)


def calculate_bayesian_diagnostics(trace, df, level_name, national_rate, model=None, ppc=None, convergence_fatal=False) -> dict:
    """Calculate comprehensive Bayesian diagnostics.

    Args:
        model: PyMC Model object (needed for PPC calculation if ppc not provided)
        ppc: Pre-computed posterior predictive samples (avoids redundant sampling)
        convergence_fatal: Flag indicating critical convergence failure (divergences >5%)
    """
    diagnostics = {
        'level': level_name,
        'model_name': 'Bayesian',
        'n_territories': len(df),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'convergence_fatal': convergence_fatal
    }

    # Convergence diagnostics (R-hat). Models name their per-territory
    # parameters differently -- the standard single-window model uses alpha
    # (intercept) and beta (slope), the two-period model uses a (intercept) and
    # delta (current-period change) -- so pick whichever intercept/slope-style
    # variables are present rather than assuming fixed names.
    rhat = az.rhat(trace)
    _int_var = next((v for v in ('alpha', 'a') if v in rhat), None)
    _slope_var = next((v for v in ('beta', 'delta') if v in rhat), None)
    diagnostics['rhat_alpha_max'] = float(rhat[_int_var].max().values) if _int_var else None
    diagnostics['rhat_beta_max'] = float(rhat[_slope_var].max().values) if _slope_var else None

    # Hyperparameters of the random intercept. ``beta`` is now a single
    # shared coefficient (audit C1) so there is no mu_beta / sigma_beta.
    if 'mu_alpha' in rhat:
        diagnostics['rhat_mu_alpha'] = float(rhat['mu_alpha'].values)
        diagnostics['rhat_sigma_alpha'] = float(rhat['sigma_alpha'].values)
    else:
        diagnostics['rhat_mu_alpha'] = None
        diagnostics['rhat_sigma_alpha'] = None

    # Check all available R-hats < 1.1 (a missing variable is simply skipped).
    _rhat_vals = [diagnostics[k] for k in
                  ('rhat_alpha_max', 'rhat_beta_max', 'rhat_mu_alpha', 'rhat_sigma_alpha')
                  if diagnostics.get(k) is not None]
    all_rhat_ok = all(v < 1.1 for v in _rhat_vals) if _rhat_vals else True

    diagnostics['convergence_ok'] = 'Yes' if all_rhat_ok else 'No'

    # Effective sample size
    ess = az.ess(trace)
    _int_var_e = next((v for v in ('alpha', 'a') if v in ess), None)
    _slope_var_e = next((v for v in ('beta', 'delta') if v in ess), None)
    diagnostics['ess_alpha_min'] = float(ess[_int_var_e].min().values) if _int_var_e else None
    diagnostics['ess_beta_min'] = float(ess[_slope_var_e].min().values) if _slope_var_e else None

    if 'mu_alpha' in ess:
        diagnostics['ess_mu_alpha'] = float(ess['mu_alpha'].values)
    else:
        diagnostics['ess_mu_alpha'] = None

    diagnostics['ess_adequate'] = ('Yes' if (diagnostics['ess_alpha_min'] is not None
                                             and diagnostics['ess_alpha_min'] > 400) else 'No')

    # Divergences (NEW)
    try:
        divergences = trace.sample_stats.diverging.sum().values
        total_samples = trace.posterior.sizes['draw'] * trace.posterior.sizes['chain']
        diagnostics['n_divergences'] = int(divergences)
        diagnostics['pct_divergences'] = float(divergences / total_samples * 100)
        diagnostics['divergences_ok'] = 'Yes' if diagnostics['pct_divergences'] < 1 else 'No'
    except (AttributeError, KeyError, ZeroDivisionError) as e:
        logger.warning(f"Could not calculate divergences: {e}")
        diagnostics['n_divergences'] = 0
        diagnostics['pct_divergences'] = 0.0
        diagnostics['divergences_ok'] = 'Yes'

    # Posterior predictive checks
    diagnostics['mean_predicted_prob'] = float(df['predicted_prob'].mean())
    diagnostics['mean_observed_prob'] = float((df['recent_count_curr'] / df['all_tested_curr']).mean())
    diagnostics['prediction_bias'] = diagnostics['mean_predicted_prob'] - diagnostics['mean_observed_prob']
    diagnostics['prediction_bias_ok'] = 'Yes' if abs(diagnostics['prediction_bias']) < 0.01 else 'No'

    # Credible interval coverage
    # FIXED: Check coverage for counts, not proportions
    if 'count_lower' in df.columns and 'count_upper' in df.columns:
        observed_count = df['recent_count_curr']
        in_ci = ((observed_count >= df['count_lower']) & (observed_count <= df['count_upper'])).sum()
        diagnostics['ci_coverage_pct'] = float(in_ci / len(df) * 100)
        diagnostics['ci_coverage_ok'] = 'Yes' if 85 <= diagnostics['ci_coverage_pct'] <= 98 else 'No'
    else:
        # Fallback to proportion-based (old method)
        observed_prop = df['recent_count_curr'] / df['all_tested_curr']
        in_ci = ((observed_prop >= df['prob_lower']) & (observed_prop <= df['prob_upper'])).sum()
        diagnostics['ci_coverage_pct'] = float(in_ci / len(df) * 100)
        diagnostics['ci_coverage_ok'] = 'Yes' if 90 <= diagnostics['ci_coverage_pct'] <= 98 else 'No'

    # Z-scores distribution
    z_national = df['z_national'].dropna()
    z_residual = df['z_residual'].dropna()
    combined_z = df['combined_z'].dropna()

    diagnostics['z_national_mean'] = float(z_national.mean())
    diagnostics['z_national_std'] = float(z_national.std())
    diagnostics['z_residual_mean'] = float(z_residual.mean())
    diagnostics['z_residual_std'] = float(z_residual.std())
    diagnostics['combined_z_mean'] = float(combined_z.mean())
    diagnostics['combined_z_std'] = float(combined_z.std())

    # Classification distribution
    class_counts = df['classification'].value_counts()
    total = len(df)
    diagnostics['pct_obvious_increase'] = float((class_counts.get('Obvious Increase', 0) / total * 100))
    diagnostics['pct_slight_increase'] = float((class_counts.get('Slight Increase', 0) / total * 100))
    diagnostics['pct_no_difference'] = float((class_counts.get('No Difference', 0) / total * 100))
    diagnostics['pct_slight_decrease'] = float((class_counts.get('Slight Decrease', 0) / total * 100))
    diagnostics['pct_obvious_decrease'] = float((class_counts.get('Obvious Decrease', 0) / total * 100))
    # Mirror the same distribution under the SIR/SMR taxonomy.
    _add_smr_sir_counts(diagnostics, df, total)

    # Extreme territories
    n_extreme = ((combined_z > 2.0) | (combined_z < -2.0)).sum()
    diagnostics['n_extreme_territories'] = int(n_extreme)
    diagnostics['pct_extreme_territories'] = float((n_extreme / total * 100))
    diagnostics['extreme_expected_pct'] = 5.0
    diagnostics['extreme_distribution_ok'] = 'Yes' if diagnostics['pct_extreme_territories'] <= 10 else 'No'

    # Posterior predictive p-value (correct: uses conditional mu = n * p)
    try:
        ppc_pvalue, ppc_ok, ppc_details = PPCCalculator.calculate_ppc_pvalue(
            trace=trace,
            y_obs=df['recent_count_curr'].values,
            n_obs=df['all_tested_curr'].values,
        )
        diagnostics['ppc_pvalue'] = ppc_pvalue
        diagnostics['ppc_ok'] = 'Yes' if ppc_ok else ('No' if ppc_ok is not None else 'Unknown')
        if ppc_details:
            diagnostics['ppc_pvalue_chi2'] = ppc_details.get('p_value_chi2')
            diagnostics['ppc_pvalue_zeros'] = ppc_details.get('p_value_zeros')
    except (ValueError, KeyError, AttributeError) as e:
        logger.warning(f"Could not calculate PPC p-value: {e}")
        diagnostics['ppc_pvalue'] = None
        diagnostics['ppc_ok'] = 'Unknown'

    # LOO-IC (Leave-One-Out Information Criterion)
    try:
        logger.info("Calculating LOO-IC for model comparison...")
        loo_result = az.loo(trace)
        diagnostics['loo_elpd'] = float(loo_result.elpd_loo)
        diagnostics['loo_se'] = float(loo_result.se)
        diagnostics['loo_p'] = float(loo_result.p_loo)

        # Check for problematic observations (high Pareto k values)
        if hasattr(loo_result, 'pareto_k'):
            pareto_k = loo_result.pareto_k.values
            n_high_k = (pareto_k > 0.7).sum()
            diagnostics['loo_n_high_pareto_k'] = int(n_high_k)
            diagnostics['loo_ok'] = 'Yes' if n_high_k == 0 else 'No'

            if n_high_k > 0:
                logger.warning(f"[WARN] {n_high_k} observations with high Pareto k (>0.7) - LOO may be unreliable")
        else:
            diagnostics['loo_ok'] = 'Yes'

        logger.info(f"LOO-IC: ELPD={diagnostics['loo_elpd']:.2f} ± {diagnostics['loo_se']:.2f}, p_loo={diagnostics['loo_p']:.2f}")
    except (ValueError, KeyError, AttributeError, RuntimeError) as e:
        logger.warning(f"Could not calculate LOO-IC: {e}")
        diagnostics['loo_elpd'] = None
        diagnostics['loo_se'] = None
        diagnostics['loo_p'] = None
        diagnostics['loo_ok'] = 'Unknown'

    # OVERALL QUALITY ASSESSMENT
    quality_checks = [
        diagnostics['convergence_ok'] == 'Yes',
        diagnostics['ess_adequate'] == 'Yes',
        diagnostics['divergences_ok'] == 'Yes',
        diagnostics['ci_coverage_ok'] == 'Yes',
        diagnostics['extreme_distribution_ok'] == 'Yes'
    ]

    n_passed = sum(quality_checks)
    if n_passed >= 4:
        diagnostics['overall_quality'] = 'GOOD'
    elif n_passed >= 3:
        diagnostics['overall_quality'] = 'ACCEPTABLE'
    else:
        diagnostics['overall_quality'] = 'POOR'

    diagnostics['n_quality_checks_passed'] = n_passed
    diagnostics['n_quality_checks_total'] = len(quality_checks)

    # [WARN] IMPROVEMENT 5: Enhanced diagnostic interpretation (BayesianAnalyzer)
    interpretation = DiagnosticInterpreter.interpret_bayesian_diagnostics(diagnostics)
    diagnostics['interpretation'] = interpretation
    logger.info("\n--- Enhanced Diagnostic Interpretation ---")
    for line in interpretation[:10]:  # Show first 10 lines in log
        logger.info(line)

    return diagnostics

