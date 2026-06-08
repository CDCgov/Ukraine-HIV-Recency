"""
Per-level Bayesian diagnostic plot rendering.

Two paths:

* If the diagnostics dict carries the fitted ``model``, delegate
  to :meth:`BayesianDiagnosticsFixed.create_diagnostic_plots_with_context`
  which knows how to enter the model context and ship the standard
  pack (trace, energy, PPC, divergences) into the level's output
  directory, then drop the human-readable guide alongside.
* Otherwise (fallback when only the trace is available) render
  trace / diagnostics / energy / divergences individually via the
  passed-in ``DiagnosticPlotter``, and run PPC by re-sampling from
  the posterior. PyMC normally requires a model context for PPC --
  this fallback rarely survives in practice and is kept for parity
  with the original behaviour.

After rendering, the trace is cleared from ``diag_bayes`` to free
memory before the comparison/covariates work in the same level.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pymc as pm

from pipeline.diagnostics import BayesianDiagnosticsFixed
from pipeline.reporting import generate_diagnostic_plots_guide

logger = logging.getLogger(__name__)


def create_bayesian_plots(orchestrator: Any,
                          diag_bayes: Optional[Dict[str, Any]],
                          level_name: str,
                          period_str: str,
                          plotter: Any) -> None:
    """Render the Bayesian diagnostic plot pack for one level."""
    if not diag_bayes:
        return

    try:
        logger.info(f"\n--- Creating Bayesian Diagnostic Plots ---")

        model = diag_bayes.get('model')
        trace = diag_bayes.get('trace')
        y_obs = diag_bayes.get('y_obs')
        ppc = diag_bayes.get('ppc')

        if trace is None:
            return

        is_hex = level_name.startswith("Hex_Res")

        if model is not None:
            dummy_file = orchestrator.get_output_path(
                "bayesian", level_name, "dummy.txt", is_hex=is_hex,
            )
            output_dir = dummy_file.parent

            BayesianDiagnosticsFixed.create_diagnostic_plots_with_context(
                model=model, trace=trace, output_dir=output_dir,
                prefix=f"Bayesian_{level_name}_{period_str}", ppc=ppc
            )
            generate_diagnostic_plots_guide(output_dir)

            diag_bayes['trace'] = None
            del trace
            logger.debug("Cleaned up trace from memory after diagnostic plots")
            return

        # Fallback when only trace is available (no model context)
        trace_file = orchestrator.get_output_path("bayesian", level_name, f"Trace_Bayesian_{level_name}_{period_str}.png", is_hex=is_hex)
        plotter.plot_trace(trace, output_file=trace_file)

        diag_file = orchestrator.get_output_path("bayesian", level_name, f"Diagnostics_Bayesian_{level_name}_{period_str}.png", is_hex=is_hex)
        plotter.plot_diagnostics_summary(diag_bayes, model_type='Bayesian', output_file=diag_file)

        energy_file = orchestrator.get_output_path("bayesian", level_name, f"Energy_Bayesian_{level_name}_{period_str}.png", is_hex=is_hex)
        plotter.plot_energy(trace, output_file=energy_file)

        if diag_bayes.get('n_divergences', 0) > 0:
            div_file = orchestrator.get_output_path("bayesian", level_name, f"Divergences_Bayesian_{level_name}_{period_str}.png", is_hex=is_hex)
            plotter.plot_divergences_scatter(trace, output_file=div_file)

        if y_obs is not None:
            ppc = pm.sample_posterior_predictive(
                trace, progressbar=False,
                random_seed=orchestrator.config.get('random_seed', 42),
            )
            ppc_samples = ppc.posterior_predictive['y_obs'].values.reshape(-1, len(y_obs))

            ppc_file = orchestrator.get_output_path("bayesian", level_name, f"PPC_Bayesian_{level_name}_{period_str}.png", is_hex=is_hex)
            plotter.plot_posterior_predictive_check(y_obs, ppc_samples, output_file=ppc_file)

        diag_bayes['trace'] = None
        del trace
        logger.debug("Cleaned up trace from memory after diagnostic plots (fallback method)")
    except (IOError, AttributeError, ValueError) as e:
        logger.error(f"Failed to create Bayesian diagnostic plots: {e}")
