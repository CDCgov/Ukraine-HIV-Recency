"""
Per-level diagnostic plot rendering for the Bayesian Covariates fit.

Mirrors :func:`create_bayesian_plots` but writes into the
``bayesian_covariates`` output subtree and uses the covariates-model
variable names (``mu_alpha``, ``sigma_alpha``, ``mu_beta``,
``sigma_beta``, ``beta_risk``) for the trace plot fallback. The
posterior-predictive plot for the covariates model reads
``ppc.posterior_predictive['obs']`` (vs ``['y_obs']`` for the
non-covariates model).

The model-context path delegates to
:meth:`BayesianDiagnosticsFixed.create_diagnostic_plots_with_context`;
the fallback path renders plots individually via the passed-in
``DiagnosticPlotter`` and is the same shape as the Bayesian variant
to keep the two side by side comparable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pipeline.diagnostics import BayesianDiagnosticsFixed
from pipeline.reporting import generate_diagnostic_plots_guide

logger = logging.getLogger(__name__)


def create_bayesian_covariates_plots(orchestrator: Any,
                                     diag_bayes_cov: Optional[Dict[str, Any]],
                                     level_name: str,
                                     period_str: str,
                                     plotter: Any) -> None:
    """Render the Bayesian Covariates diagnostic plot pack for one level."""
    if not diag_bayes_cov:
        return

    try:
        logger.info(f"\n--- Creating Bayesian Covariates Diagnostic Plots ---")

        model = diag_bayes_cov.get('model')
        trace = diag_bayes_cov.get('trace')
        y_obs = diag_bayes_cov.get('y_obs')
        ppc = diag_bayes_cov.get('ppc')

        if trace is None:
            return

        is_hex = level_name.startswith("Hex_Res")

        if model is not None:
            dummy_file = orchestrator.get_output_path(
                "bayesian_covariates", level_name, "dummy.txt", is_hex=is_hex,
            )
            output_dir = dummy_file.parent

            BayesianDiagnosticsFixed.create_diagnostic_plots_with_context(
                model=model, trace=trace, output_dir=output_dir,
                prefix=f"Bayesian_Covariates_{level_name}_{period_str}", ppc=ppc
            )
            generate_diagnostic_plots_guide(output_dir)

            diag_bayes_cov['trace'] = None
            del trace
            logger.debug("Cleaned up trace from memory after diagnostic plots (Bayesian Covariates)")
            return

        # Fallback when only trace is available (no model context)
        trace_file = orchestrator.get_output_path("bayesian_covariates", level_name, f"Trace_Bayesian_Covariates_{level_name}_{period_str}.png", is_hex=is_hex)
        var_names = ['mu_alpha', 'sigma_alpha', 'mu_beta', 'sigma_beta', 'beta_risk']
        plotter.plot_trace(trace, var_names=var_names, output_file=trace_file)

        diag_file = orchestrator.get_output_path("bayesian_covariates", level_name, f"Diagnostics_Bayesian_Covariates_{level_name}_{period_str}.png", is_hex=is_hex)
        plotter.plot_diagnostics_summary(diag_bayes_cov, model_type='Bayesian', output_file=diag_file)

        energy_file = orchestrator.get_output_path("bayesian_covariates", level_name, f"Energy_Bayesian_Covariates_{level_name}_{period_str}.png", is_hex=is_hex)
        plotter.plot_energy(trace, output_file=energy_file)

        if diag_bayes_cov.get('n_divergences', 0) > 0:
            div_file = orchestrator.get_output_path("bayesian_covariates", level_name, f"Divergences_Bayesian_Covariates_{level_name}_{period_str}.png", is_hex=is_hex)
            plotter.plot_divergences_scatter(trace, output_file=div_file)

        if y_obs is not None and ppc is not None:
            ppc_samples = ppc.posterior_predictive['obs'].values.reshape(-1, len(y_obs))
            ppc_file = orchestrator.get_output_path("bayesian_covariates", level_name, f"PPC_Bayesian_Covariates_{level_name}_{period_str}.png", is_hex=is_hex)
            plotter.plot_posterior_predictive_check(y_obs, ppc_samples, output_file=ppc_file)

        diag_bayes_cov['trace'] = None
        del trace
        logger.debug("Cleaned up trace from memory after diagnostic plots (Bayesian Covariates fallback)")
    except (IOError, AttributeError, ValueError) as e:
        logger.error(f"Failed to create Bayesian Covariates diagnostic plots: {e}")
