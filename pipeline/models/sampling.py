"""
MCMC sampling configuration and adaptive sampler.

Two pieces live here:

* :func:`get_sampling_config` picks ``chains / tune / draws / cores /
  target_accept`` from the number of territories in the analysis (or
  returns a fast preset when ``--test`` runs). The choice is conservative:
  more chains and a higher ``target_accept`` for small problems where the
  posterior geometry is harder to sample reliably, fewer chains for the
  largest problems where wall-clock cost dominates.

* :func:`adaptive_sample` is the wrapper that PyMC ``pm.sample`` is called
  through. It retries up to three times when diagnostics fail
  (divergences, low E-BFMI, tree-depth saturation) and progressively
  tightens ``target_accept`` / extends ``max_treedepth``. Diagnostics are
  recorded on the returned ``sampling_info`` dict so the rest of the
  pipeline (reliability score, audit trail) can act on them.

The :class:`ParallelSamplingConfig` and :class:`SamplingProgressBar`
classes are thin static-method wrappers around these functions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import arviz as az
import numpy as np
import pymc as pm

logger = logging.getLogger(__name__)


def get_sampling_config(n_territories: int, fast_mode: bool = False,
                        cores_override: Optional[int] = None) -> Dict[str, Any]:
    """Pick MCMC parameters from problem size.

    ``fast_mode=True`` is the ``--test`` preset: 2 chains, 500 tune /
    500 draws, lower ``target_accept``. Otherwise the choice is driven by
    territory count: fewer than 20 territories gets the most thorough
    configuration (4 chains, 2000 tune, 2000 draws), 20--49 gets a slightly
    leaner version, and 50+ falls back to 2 chains / 1000 draws to keep
    the wall-clock manageable.

    ``cores_override`` (from ``cfg['sampling']['cores']``) only changes how
    many CPU cores run the chains in parallel; it does not affect the chain
    count, seed or draws, so results are unchanged by it.
    """
    if fast_mode:
        config = {
            'chains': 2,
            'tune': 500,
            'draws': 500,
            'cores': 2,
            'target_accept': 0.9,
        }
    else:
        if n_territories < 20:
            config = {
                'chains': 4,
                'tune': 2000,
                'draws': 2000,
                'cores': 4,
                'target_accept': 0.95,
            }
        elif n_territories < 50:
            config = {
                'chains': 4,
                'tune': 2000,
                'draws': 1500,
                'cores': 4,
                'target_accept': 0.95,
            }
        else:
            config = {
                'chains': 2,
                'tune': 2000,
                'draws': 1000,
                'cores': 2,
                'target_accept': 0.95,
            }

    if cores_override:
        config['cores'] = int(cores_override)

    logger.info(f"Sampling config: {config['chains']} chains × {config['draws']} draws "
                f"({config['cores']} cores)")
    return config


def adaptive_sample(model, initial_target_accept: float = 0.95,
                    draws: int = 1000, tune: int = 2000,
                    chains: int = 4, cores: int = 4,
                    random_seed: int = 42, progressbar: bool = False,
                    callback=None) -> Tuple[az.InferenceData, Dict[str, Any]]:
    """Run ``pm.sample`` with automatic retry on diagnostics issues.

    Up to three attempts. After each sample the function checks:

    * **divergences** -- > 10% bumps ``target_accept`` to 0.99,
      > 5% to 0.97;
    * **E-BFMI** -- min below 0.2 bumps ``target_accept`` to 0.99 to give
      the sampler a chance to explore the funnel-like geometry;
    * **tree depth saturation** -- > 10% of transitions hitting
      ``max_treedepth`` raises the limit by 5.

    On success returns the trace plus a ``sampling_info`` dict carrying
    the final settings and the observed diagnostics; on the third attempt
    the function returns whatever it has so the caller can record the
    ``convergence_fatal`` state and downgrade reliability accordingly.
    """
    sampling_info = {
        'n_attempts': 0,
        'adapted': False,
        'adapted_reasons': [],
        'final_target_accept': initial_target_accept,
        'final_max_treedepth': 10,
        'divergence_pct': 0.0,
        'ebfmi': None,
        'treedepth_exceeded_pct': 0.0,
    }

    target_accept = initial_target_accept
    max_treedepth = 10
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        sampling_info['n_attempts'] = attempt

        logger.info(f"Sampling attempt {attempt}/{max_attempts} "
                    f"(target_accept={target_accept:.2f}, max_treedepth={max_treedepth})")

        # Fresh initvals matching current model dimensions. Stops PyMC
        # from reusing cached initial points from a previous model with a
        # different n_territories (which produced ``y_obs: -inf`` in
        # iterative mode).
        try:
            initvals = model.initial_point()
        except Exception:
            initvals = None

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            max_treedepth=max_treedepth,
            initvals=initvals,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
            random_seed=random_seed,
            progressbar=progressbar,
            callback=callback,
        )

        issues = []

        divergences = trace.sample_stats.diverging.values
        n_divergences = int(divergences.sum())
        n_total = divergences.size
        divergence_pct = (n_divergences / n_total * 100) if n_total > 0 else 0.0
        sampling_info['divergence_pct'] = divergence_pct
        logger.info(f"  Divergences: {n_divergences}/{n_total} ({divergence_pct:.1f}%)")

        if divergence_pct > 10:
            issues.append(('divergences_high', target_accept, 0.99))
        elif divergence_pct > 5:
            issues.append(('divergences_moderate', target_accept, 0.97))

        try:
            ebfmi_values = az.bfmi(trace)
            ebfmi_min = float(np.min(ebfmi_values))
            sampling_info['ebfmi'] = ebfmi_min
            logger.info(f"  E-BFMI: {ebfmi_min:.3f} (chains: {[f'{v:.3f}' for v in ebfmi_values]})")

            if ebfmi_min < 0.2:
                issues.append(('ebfmi_low', target_accept, 0.99))
                logger.warning(f"  [WARN] Low E-BFMI ({ebfmi_min:.3f} < 0.2) — poor exploration")
            elif ebfmi_min < 0.5:
                logger.info(f"  [WARN] E-BFMI ({ebfmi_min:.3f}) — acceptable but not ideal")
        except Exception as e:
            logger.debug(f"  Could not compute E-BFMI: {e}")
            sampling_info['ebfmi'] = None

        try:
            tree_depth = trace.sample_stats['tree_depth'].values
            max_td_used = int(np.max(tree_depth))
            n_exceeded = int(np.sum(tree_depth >= max_treedepth))
            treedepth_exceeded_pct = (n_exceeded / tree_depth.size * 100) if tree_depth.size > 0 else 0.0
            sampling_info['treedepth_exceeded_pct'] = treedepth_exceeded_pct
            logger.info(f"  Tree depth: max used={max_td_used}, "
                        f"exceeded limit ({max_treedepth}) in {treedepth_exceeded_pct:.1f}% transitions")

            if treedepth_exceeded_pct > 10:
                issues.append(('treedepth_exceeded', max_treedepth, max_treedepth + 5))
                logger.warning(f"  [WARN] Tree depth exceeded in {treedepth_exceeded_pct:.1f}% — increasing to {max_treedepth + 5}")
        except Exception as e:
            logger.debug(f"  Could not check tree depth: {e}")
            sampling_info['treedepth_exceeded_pct'] = 0.0

        if issues and attempt < max_attempts:
            sampling_info['adapted'] = True
            for reason, _old_val, new_val in issues:
                sampling_info['adapted_reasons'].append(reason)
                if 'accept' in reason:
                    target_accept = new_val
                elif 'treedepth' in reason:
                    max_treedepth = new_val
            continue
        else:
            if issues:
                logger.warning(f"[WARN] Issues present but max attempts reached: {[i[0] for i in issues]}")
            else:
                logger.info(f"[OK] All diagnostics acceptable")

            sampling_info['final_target_accept'] = target_accept
            sampling_info['final_max_treedepth'] = max_treedepth
            return trace, sampling_info

    sampling_info['final_target_accept'] = target_accept
    sampling_info['final_max_treedepth'] = max_treedepth
    return trace, sampling_info


class ParallelSamplingConfig:
    """Static-method wrapper kept for ``ParallelSamplingConfig.<m>(...)`` callers."""

    @staticmethod
    def get_sampling_config(n_territories: int, fast_mode: bool = False,
                            cores_override: Optional[int] = None) -> Dict[str, Any]:
        """Delegate to the module-level :func:`get_sampling_config`."""
        return get_sampling_config(n_territories, fast_mode=fast_mode,
                                   cores_override=cores_override)

    @staticmethod
    def adaptive_sample(model, initial_target_accept: float = 0.95,
                        draws: int = 1000, tune: int = 2000,
                        chains: int = 4, cores: int = 4,
                        random_seed: int = 42, progressbar: bool = False,
                        callback=None) -> Tuple[az.InferenceData, Dict[str, Any]]:
        """Delegate to the module-level :func:`adaptive_sample`."""
        return adaptive_sample(
            model,
            initial_target_accept=initial_target_accept,
            draws=draws, tune=tune, chains=chains, cores=cores,
            random_seed=random_seed, progressbar=progressbar,
            callback=callback,
        )


class SamplingProgressBar:
    """tqdm-backed progress callback for PyMC sampling.

    Falls back to ``None`` (no progress bar) when tqdm is not installed.
    """

    @staticmethod
    def create_progress_callback():
        """Return a per-draw tqdm callback, or ``None`` if tqdm is missing."""
        try:
            from tqdm.auto import tqdm

            pbar = None
            total_draws = None

            def callback(trace, draw):
                """Advance the bar one draw, creating it on the first call."""
                nonlocal pbar, total_draws
                if pbar is None:
                    if hasattr(trace, 'nchains'):
                        total_draws = 2000 * trace.nchains
                    else:
                        total_draws = 2000
                    pbar = tqdm(total=total_draws, desc="Sampling")
                pbar.update(1)

            return callback
        except ImportError:
            logger.warning("tqdm not installed - no progress bar available")
            return None
