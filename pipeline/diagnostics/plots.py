"""
Diagnostic plot generation.

Two classes live here:

* :class:`BayesianDiagnosticsFixed` -- PyMC-context-aware plotter that
  produces the PPC / pair / forest plots for a fitted model. The "Fixed"
  in the name is historical: an earlier version regenerated the trace
  inside the wrong PyMC context; this version takes the model and trace
  explicitly so the plotting block can be reused safely across analyzers.
* :class:`DiagnosticPlotter` -- the lighter static-methods plotter that
  emits the summary panel, trace, energy, divergences scatter and PPC
  histogram from the InferenceData alone.

Both rely on matplotlib + arviz; ``arviz`` cooperates with the active
PyMC context for the BayesianDiagnosticsFixed flows because the model is
re-entered before each ``az.plot_*`` call.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

logger = logging.getLogger(__name__)


class BayesianDiagnosticsFixed:
    """Fixed Bayesian diagnostics with proper PyMC context management."""

    @staticmethod
    def create_diagnostic_plots_with_context(model: pm.Model, trace: az.InferenceData,
                                             output_dir: Path, prefix: str,
                                             ppc: Optional[Any] = None) -> None:
        """Create PPC + pair + forest plots inside the model's PyMC context.

        ``ppc`` is reused when supplied; otherwise it is sampled from the
        trace once and used for the PPC panel.
        """
        try:
            with model:
                try:
                    if ppc is None:
                        ppc = pm.sample_posterior_predictive(trace, random_seed=42)
                    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                    try:
                        az.plot_ppc(ppc, ax=ax)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            plt.tight_layout()
                            plt.savefig(output_dir / f'{prefix}_PPC.png', dpi=150, bbox_inches='tight')
                        logger.info(f"[OK] PPC plot saved: {prefix}_PPC.png")
                    finally:
                        plt.close(fig)
                except (ValueError, KeyError, RuntimeError, IOError) as e:
                    logger.warning(f"Could not create PPC plot: {e}")

                try:
                    available_vars = list(trace.posterior.data_vars.keys())
                    hyper_vars = [v for v in available_vars
                                  if v.startswith(('mu_', 'sigma_', 'beta_'))
                                  and not v.endswith('_offset')]

                    if len(hyper_vars) >= 2:
                        try:
                            az.plot_pair(
                                trace,
                                var_names=hyper_vars[:6],
                                kind='scatter',
                                figsize=(12, 12),
                                divergences=True,
                            )
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Scatter plot failed: {e}, trying kde")
                            az.plot_pair(
                                trace,
                                var_names=hyper_vars[:6],
                                kind='kde',
                                figsize=(12, 12),
                            )
                        try:
                            plt.tight_layout()
                            plt.savefig(output_dir / f'{prefix}_Pairs.png', dpi=150, bbox_inches='tight')
                            logger.info(f"[OK] Pair plot saved: {prefix}_Pairs.png")
                        finally:
                            plt.close('all')
                    else:
                        logger.info(f"Skipping pair plot: only {len(hyper_vars)} hyperparameters found")
                except (ValueError, KeyError, RuntimeError, IOError, TypeError) as e:
                    logger.warning(f"Could not create pair plot: {e}")

                try:
                    available_vars = list(trace.posterior.data_vars.keys())
                    hyper_vars = [v for v in available_vars
                                  if v.startswith(('mu_', 'sigma_', 'beta_'))
                                  and not v.endswith('_offset')]

                    if len(hyper_vars) >= 1:
                        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                        try:
                            az.plot_forest(
                                trace,
                                var_names=hyper_vars[:6],
                                combined=True,
                                ax=ax,
                            )
                            plt.tight_layout()
                            plt.savefig(output_dir / f'{prefix}_Forest.png', dpi=150, bbox_inches='tight')
                            logger.info(f"[OK] Forest plot saved: {prefix}_Forest.png")
                        finally:
                            plt.close(fig)
                    else:
                        logger.info(f"Skipping forest plot: no hyperparameters found")
                except (ValueError, KeyError, RuntimeError, IOError) as e:
                    logger.warning(f"Could not create forest plot: {e}")

        except (ValueError, KeyError, RuntimeError, IOError) as e:
            logger.error(f"Failed to create diagnostic plots with context: {e}")


class DiagnosticPlotter:
    """Static plotting helpers for fitted Bayesian models."""

    @staticmethod
    def plot_diagnostics_summary(diagnostics: Dict, model_type: str, output_file: Path) -> None:
        """Four-panel summary card: convergence / fit / diagnostics / count."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{model_type} Diagnostics Summary', fontsize=16)

        ax = axes[0, 0]
        ax.axis('off')
        converged = diagnostics.get('converged', diagnostics.get('convergence_ok', 'Unknown'))
        ax.text(0.5, 0.5, f"Converged: {converged}", ha='center', va='center', fontsize=14)
        ax.set_title('Convergence')

        ax = axes[0, 1]
        ax.axis('off')
        if 'pseudo_r2' in diagnostics:
            r2 = diagnostics['pseudo_r2']
            ax.text(0.5, 0.5, f"Pseudo R²: {r2:.3f}", ha='center', va='center', fontsize=14)
        elif 'overall_quality' in diagnostics:
            quality = diagnostics['overall_quality']
            ax.text(0.5, 0.5, f"Quality: {quality}", ha='center', va='center', fontsize=14)
        ax.set_title('Model Fit')

        ax = axes[1, 0]
        ax.axis('off')
        diag_text: List[str] = []
        if 'ess_adequate' in diagnostics:
            diag_text.append(f"ESS: {diagnostics['ess_adequate']}")
        if 'divergences_ok' in diagnostics:
            diag_text.append(f"Divergences: {diagnostics['divergences_ok']}")
        if 'residuals_normal' in diagnostics:
            diag_text.append(f"Residuals Normal: {diagnostics['residuals_normal']}")
        ax.text(0.5, 0.5, '\n'.join(diag_text), ha='center', va='center', fontsize=12)
        ax.set_title('Diagnostics')

        ax = axes[1, 1]
        ax.axis('off')
        n_terr = diagnostics.get('n_territories', 'N/A')
        ax.text(0.5, 0.5, f"Territories: {n_terr}", ha='center', va='center', fontsize=14)
        ax.set_title('Summary')

        try:
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"[OK] Diagnostic summary plot saved: {output_file}")
        finally:
            plt.close(fig)

    @staticmethod
    def plot_trace(trace: az.InferenceData, var_names: Optional[List[str]] = None,
                   output_file: Optional[Path] = None) -> None:
        """Standard arviz trace plot (defaults to first six variables)."""
        if var_names is None:
            var_names = list(trace.posterior.data_vars.keys())[:6]

        fig = az.plot_trace(trace, var_names=var_names, figsize=(12, len(var_names) * 2))
        try:
            plt.tight_layout()
            if output_file:
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                logger.info(f"[OK] Trace plot saved: {output_file}")
        finally:
            plt.close(fig)

    @staticmethod
    def plot_energy(trace: az.InferenceData, output_file: Path) -> None:
        """Standard arviz energy plot (HMC sampler health)."""
        fig = az.plot_energy(trace)
        try:
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"[OK] Energy plot saved: {output_file}")
        finally:
            plt.close(fig)

    @staticmethod
    def plot_divergences_scatter(trace: az.InferenceData, output_file: Path) -> None:
        """Two-parameter scatter highlighting divergent transitions in red."""
        divergent = trace.sample_stats.diverging.values.flatten()

        if divergent.sum() == 0:
            logger.info("No divergences to plot")
            return

        var_names = list(trace.posterior.data_vars.keys())[:2]
        if len(var_names) < 2:
            logger.warning("Not enough variables for divergence scatter plot")
            return

        x = trace.posterior[var_names[0]].values.flatten()
        y = trace.posterior[var_names[1]].values.flatten()

        fig, ax = plt.subplots(figsize=(10, 8))
        try:
            ax.scatter(x[~divergent], y[~divergent], alpha=0.5, label='Normal', s=10)
            ax.scatter(x[divergent], y[divergent], color='red', alpha=0.8, label='Divergent', s=20)
            ax.set_xlabel(var_names[0])
            ax.set_ylabel(var_names[1])
            ax.set_title('Divergences Scatter Plot')
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"[OK] Divergences scatter plot saved: {output_file}")
        finally:
            plt.close(fig)

    @staticmethod
    def plot_posterior_predictive_check(y_obs: np.ndarray, ppc_samples: np.ndarray,
                                        output_file: Path) -> None:
        """Histogram of observed data with up to 100 posterior-predictive draws overlaid."""
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            ax.hist(y_obs, bins=30, alpha=0.5, label='Observed', density=True)
            for i in range(min(100, ppc_samples.shape[0])):
                ax.hist(ppc_samples[i], bins=30, alpha=0.01, color='blue', density=True)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.set_title('Posterior Predictive Check')
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"[OK] PPC plot saved: {output_file}")
        finally:
            plt.close(fig)
