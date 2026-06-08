"""
Comprehensive Bayesian model diagnostics.

Three flavours, one per analyzer class:

* :func:`calculate_bayesian_diagnostics` -- crude Bayesian model
  (alpha + beta hyperparameters).
* :func:`calculate_covariates_diagnostics` -- Bayesian model with the
  ``proportion_high_risk`` covariate (alpha + beta + beta_risk).
* :func:`calculate_covariates_diagnostics_stratified` -- stratified
  covariates fit on the HARD frame.

All three compute the same fundamental diagnostics dict: R-hat per
hyperparameter, ESS (bulk / tail), divergence count and rate, BFMI,
tree-depth saturation, the convergence_ok / divergences_ok / ess_adequate
flags, and the overall quality label. They differ only in which model
parameters are summarised (Bayesian has no beta_risk; the stratified
variant has different signature for the trace / df pair).
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

    # Convergence diagnostics (R-hat)
    rhat = az.rhat(trace)
    diagnostics['rhat_alpha_max'] = float(rhat['alpha'].max().values)
    diagnostics['rhat_beta_max'] = float(rhat['beta'].max().values)

    # Hyperparameters of the random intercept. ``beta`` is now a single
    # shared coefficient (audit C1) so there is no mu_beta / sigma_beta.
    if 'mu_alpha' in rhat:
        diagnostics['rhat_mu_alpha'] = float(rhat['mu_alpha'].values)
        diagnostics['rhat_sigma_alpha'] = float(rhat['sigma_alpha'].values)

        # Check all R-hat < 1.1
        all_rhat_ok = all([
            diagnostics['rhat_alpha_max'] < 1.1,
            diagnostics['rhat_beta_max'] < 1.1,
            diagnostics['rhat_mu_alpha'] < 1.1,
            diagnostics['rhat_sigma_alpha'] < 1.1
        ])
    else:
        # Fixed effects model - no hyperparameters
        diagnostics['rhat_mu_alpha'] = None
        diagnostics['rhat_sigma_alpha'] = None

        all_rhat_ok = all([
            diagnostics['rhat_alpha_max'] < 1.1,
            diagnostics['rhat_beta_max'] < 1.1
        ])

    diagnostics['convergence_ok'] = 'Yes' if all_rhat_ok else 'No'

    # Effective sample size
    ess = az.ess(trace)
    diagnostics['ess_alpha_min'] = float(ess['alpha'].min().values)
    diagnostics['ess_beta_min'] = float(ess['beta'].min().values)

    if 'mu_alpha' in ess:
        diagnostics['ess_mu_alpha'] = float(ess['mu_alpha'].values)
    else:
        diagnostics['ess_mu_alpha'] = None

    diagnostics['ess_adequate'] = 'Yes' if diagnostics['ess_alpha_min'] > 400 else 'No'

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



def calculate_covariates_diagnostics_stratified(trace, df_territory, df_stratified, level_name, national_rate,
                                     beta_risk_mean, beta_risk_lower, beta_risk_upper,
                                     OR_risk, OR_risk_lower, OR_risk_upper,
                                     territory_analysis, ppc=None, convergence_fatal=False) -> dict:
    """Calculate diagnostics for stratified model with component analysis.

    Args:
        ppc: Pre-computed posterior predictive samples (avoids redundant sampling)
        convergence_fatal: Flag indicating critical convergence failure (divergences >5%)
    """
    diagnostics = {
        'level': level_name,
        'model_type': 'Bayesian Hierarchical (Stratified by Risk Group)',
        'n_territories': len(df_territory),
        'n_observations': len(df_stratified),
        'national_rate': float(national_rate),
        'convergence_fatal': convergence_fatal
    }

    # Risk group effect (KEY RESULT!)
    diagnostics['effect_high_vs_low_mean'] = beta_risk_mean
    diagnostics['effect_high_vs_low_lower'] = beta_risk_lower
    diagnostics['effect_high_vs_low_upper'] = beta_risk_upper
    diagnostics['odds_ratio_high_vs_low'] = OR_risk
    diagnostics['odds_ratio_high_vs_low_lower'] = OR_risk_lower
    diagnostics['odds_ratio_high_vs_low_upper'] = OR_risk_upper

    # 1b. Testing intensity effect (NEW!)
    beta_intensity_mean = float(trace.posterior['beta_intensity'].mean().values)
    beta_intensity_hdi = az.hdi(trace, var_names=['beta_intensity'])['beta_intensity'].values
    beta_intensity_lower = float(beta_intensity_hdi[0])
    beta_intensity_upper = float(beta_intensity_hdi[1])

    OR_intensity = np.exp(beta_intensity_mean)
    OR_intensity_lower = np.exp(beta_intensity_lower)
    OR_intensity_upper = np.exp(beta_intensity_upper)

    diagnostics['effect_testing_intensity_mean'] = beta_intensity_mean
    diagnostics['effect_testing_intensity_lower'] = beta_intensity_lower
    diagnostics['effect_testing_intensity_upper'] = beta_intensity_upper
    diagnostics['odds_ratio_testing_intensity'] = OR_intensity
    diagnostics['odds_ratio_testing_intensity_lower'] = OR_intensity_lower
    diagnostics['odds_ratio_testing_intensity_upper'] = OR_intensity_upper

    # Component analysis results
    diagnostics['territory_analysis'] = territory_analysis

    # Count territories by type
    n_high_outbreak = sum(1 for t in territory_analysis if t['high_outbreak'] and not t['low_outbreak'] and not t['testing_artifact'])
    n_low_outbreak = sum(1 for t in territory_analysis if t['low_outbreak'] and not t['high_outbreak'] and not t['testing_artifact'])
    n_both_outbreak = sum(1 for t in territory_analysis if t['high_outbreak'] and t['low_outbreak'])
    n_testing_artifact = sum(1 for t in territory_analysis if t['testing_artifact'])
    n_stable = sum(1 for t in territory_analysis if not t['high_outbreak'] and not t['low_outbreak'] and not t['testing_artifact'])

    diagnostics['n_high_outbreak'] = n_high_outbreak
    diagnostics['n_low_outbreak'] = n_low_outbreak
    diagnostics['n_both_outbreak'] = n_both_outbreak
    diagnostics['n_testing_artifact'] = n_testing_artifact
    diagnostics['n_stable'] = n_stable

    if beta_risk_lower > 0:
        diagnostics['high_vs_low_conclusion'] = 'High-risk has SIGNIFICANTLY higher level (real outbreak)'
    elif beta_risk_upper < 0:
        diagnostics['high_vs_low_conclusion'] = 'High-risk has SIGNIFICANTLY lower level'
    else:
        diagnostics['high_vs_low_conclusion'] = 'NO significant difference between high and low'

    if beta_intensity_lower > 0:
        diagnostics['testing_intensity_conclusion'] = 'Higher testing intensity → SIGNIFICANTLY higher observed rate'
    elif beta_intensity_upper < 0:
        diagnostics['testing_intensity_conclusion'] = 'Higher testing intensity → SIGNIFICANTLY lower observed rate'
    else:
        diagnostics['testing_intensity_conclusion'] = 'NO significant effect of testing intensity'

    # Model convergence diagnostics
    summary = az.summary(trace, var_names=['beta_risk', 'beta_intensity', 'beta_hist', 'mu_alpha'])
    diagnostics['max_rhat'] = float(summary['r_hat'].max())
    diagnostics['convergence_ok'] = 'Yes' if diagnostics['max_rhat'] < 1.01 else 'No'

    # ESS
    diagnostics['min_ess_bulk'] = float(summary['ess_bulk'].min())
    diagnostics['min_ess_tail'] = float(summary['ess_tail'].min())
    diagnostics['ess_adequate'] = 'Yes' if diagnostics['min_ess_bulk'] > 400 else 'No'

    # Divergences
    divergences = trace.sample_stats['diverging'].sum().values
    diagnostics['n_divergences'] = int(divergences)
    diagnostics['pct_divergences'] = float(divergences / (trace.posterior.sizes['draw'] * trace.posterior.sizes['chain']) * 100)
    diagnostics['divergences_ok'] = 'Yes' if diagnostics['pct_divergences'] < 1.0 else 'No'

    # CI coverage (check on stratified data)
    if 'count_lower' in df_stratified.columns and 'count_upper' in df_stratified.columns:
        observed_count = df_stratified['recent_count_curr']
        in_ci = ((observed_count >= df_stratified['count_lower']) &
                 (observed_count <= df_stratified['count_upper'])).sum()
        diagnostics['ci_coverage_pct'] = float(in_ci / len(df_stratified) * 100)
        diagnostics['ci_coverage_ok'] = 'Yes' if 85 <= diagnostics['ci_coverage_pct'] <= 98 else 'No'
    else:
        diagnostics['ci_coverage_pct'] = None
        diagnostics['ci_coverage_ok'] = 'Unknown'

    # Z-score calibration (on territory level)
    z_national = df_territory['z_national'].values
    z_residual = df_territory['z_residual'].values
    combined_z = df_territory['combined_z'].values

    diagnostics['mean_z_national'] = float(np.mean(z_national))
    diagnostics['std_z_national'] = float(np.std(z_national))
    diagnostics['mean_z_residual'] = float(np.mean(z_residual))
    diagnostics['std_z_residual'] = float(np.std(z_residual))
    diagnostics['mean_combined_z'] = float(np.mean(combined_z))
    diagnostics['std_combined_z'] = float(np.std(combined_z))

    # Classification distribution
    class_counts = df_territory['classification'].value_counts()
    total = len(df_territory)
    diagnostics['n_no_difference'] = int(class_counts.get('No Difference', 0))
    diagnostics['n_slight_increase'] = int(class_counts.get('Slight Increase', 0))
    diagnostics['n_obvious_increase'] = int(class_counts.get('Obvious Increase', 0))
    diagnostics['n_slight_decrease'] = int(class_counts.get('Slight Decrease', 0))
    diagnostics['n_obvious_decrease'] = int(class_counts.get('Obvious Decrease', 0))

    diagnostics['pct_no_difference'] = float((class_counts.get('No Difference', 0) / total * 100))
    diagnostics['pct_slight_increase'] = float((class_counts.get('Slight Increase', 0) / total * 100))
    diagnostics['pct_obvious_increase'] = float((class_counts.get('Obvious Increase', 0) / total * 100))
    diagnostics['pct_slight_decrease'] = float((class_counts.get('Slight Decrease', 0) / total * 100))
    diagnostics['pct_obvious_decrease'] = float((class_counts.get('Obvious Decrease', 0) / total * 100))
    # Mirror the same distribution under the SIR/SMR taxonomy (territory level).
    _add_smr_sir_counts(diagnostics, df_territory, total, include_n=True)

    # Extreme territories
    n_extreme = ((combined_z > 2.0) | (combined_z < -2.0)).sum()
    diagnostics['n_extreme_territories'] = int(n_extreme)
    diagnostics['pct_extreme_territories'] = float((n_extreme / total * 100))
    diagnostics['extreme_expected_pct'] = 5.0
    diagnostics['extreme_distribution_ok'] = 'Yes' if diagnostics['pct_extreme_territories'] <= 10 else 'No'

    # PPC p-value (correct: uses conditional mu = n * p)
    try:
        ppc_pvalue, ppc_ok, ppc_details = PPCCalculator.calculate_ppc_pvalue(
            trace=trace,
            y_obs=df_stratified['recent_count_curr'].values,
            n_obs=df_stratified['all_tested_curr'].values,
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

    # Overall quality
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

    return diagnostics



def calculate_covariates_diagnostics(trace, df, level_name, national_rate) -> dict:
    """Calculate comprehensive Bayesian diagnostics."""
    diagnostics = {
        'level': level_name,
        'model_name': 'Bayesian with Covariates',
        'n_territories': len(df),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Convergence diagnostics (R-hat)
    rhat = az.rhat(trace)
    diagnostics['rhat_alpha_max'] = float(rhat['alpha'].max().values)
    diagnostics['rhat_beta_max'] = float(rhat['beta'].max().values)
    diagnostics['rhat_beta_risk'] = float(rhat['beta_risk'].values)
    diagnostics['rhat_mu_alpha'] = float(rhat['mu_alpha'].values)
    diagnostics['rhat_mu_beta'] = float(rhat['mu_beta'].values)
    diagnostics['rhat_sigma_alpha'] = float(rhat['sigma_alpha'].values)
    diagnostics['rhat_sigma_beta'] = float(rhat['sigma_beta'].values)

    # Check all R-hat < 1.1
    all_rhat_ok = all([
        diagnostics['rhat_alpha_max'] < 1.1,
        diagnostics['rhat_beta_max'] < 1.1,
        diagnostics['rhat_beta_risk'] < 1.1,
        diagnostics['rhat_mu_alpha'] < 1.1,
        diagnostics['rhat_mu_beta'] < 1.1,
        diagnostics['rhat_sigma_alpha'] < 1.1,
        diagnostics['rhat_sigma_beta'] < 1.1
    ])
    diagnostics['convergence_ok'] = 'Yes' if all_rhat_ok else 'No'

    # Effective sample size
    ess = az.ess(trace)
    diagnostics['ess_alpha_min'] = float(ess['alpha'].min().values)
    diagnostics['ess_beta_min'] = float(ess['beta'].min().values)
    diagnostics['ess_beta_risk'] = float(ess['beta_risk'].values)
    diagnostics['ess_adequate'] = 'Yes' if diagnostics['ess_alpha_min'] > 400 else 'No'

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

    # NEW: Z-score calibration check (CRITICAL for covariate model)
    diagnostics['z_calibration_ok'] = 'Yes' if 0.8 <= diagnostics['combined_z_std'] <= 1.5 else 'No'

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

    # [WARN] IMPROVEMENT 5: Enhanced diagnostic interpretation (BayesianCovariatesAnalyzer)
    interpretation = DiagnosticInterpreter.interpret_bayesian_diagnostics(diagnostics)
    diagnostics['interpretation'] = interpretation
    logger.info("\n--- Enhanced Diagnostic Interpretation ---")
    for line in interpretation[:10]:  # Show first 10 lines in log
        logger.info(line)

    return diagnostics

