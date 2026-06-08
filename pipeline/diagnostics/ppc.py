"""
Posterior predictive check for the HIV hotspot pipeline.

The check uses three targeted test statistics instead of a single
chi-square: zero proportion (matters because facility-based surveillance
data are heavily zero-loaded), 95th-percentile count (the upper tail is
what the SMR / SIR ranking reads) and overall mean rate (calibration of
the central tendency). Replicated counts are drawn from the same
Beta-Binomial likelihood the model uses, so the check honours the model's
own dispersion.

Each statistic returns a Bayesian p-value ``P(T(y_rep) >= T(y_obs) | data)``.
Values near 0 or 1 flag a direction in which the model misfits the data;
values near 0.5 indicate that the replicated data look like the observed.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import arviz as az
import numpy as np

logger = logging.getLogger(__name__)


class PPCCalculator:
    """Posterior predictive check via three targeted test statistics."""

    @staticmethod
    def calculate_ppc_pvalue(trace: az.InferenceData, y_obs: np.ndarray,
                             n_obs: np.ndarray,
                             ppc=None) -> Tuple[Optional[float], Optional[bool], dict]:
        """
        Run the three-statistic posterior predictive check.

        Args:
            trace: ArviZ InferenceData with posterior ``p`` and ``kappa``
                (both are produced by every analyzer migrated to the
                always-Beta-Binomial likelihood).
            y_obs: Observed recent counts per territory (1-D).
            n_obs: Tests per territory (1-D, same length as ``y_obs``).
            ppc: Kept for backward-compatible signature; unused.

        Returns:
            ``(primary, is_ok, all_pvalues)``. ``primary`` is the p95
            p-value because tail behaviour matters most for hotspot
            detection. ``is_ok`` is True when every p-value sits in
            ``[0.05, 0.95]``. ``all_pvalues`` carries each individual
            p-value under keys ``p_value_zeros``, ``p_value_p95`` and
            ``p_value_rate``.
        """
        try:
            post = trace.posterior
            if 'p' not in post:
                logger.info("PPC: posterior has no Deterministic 'p'; skipping checks.")
                return None, None, {}

            p_samples = post['p'].values  # (chains, draws, n_territories)
            _, _, n_terr = p_samples.shape
            p_flat = p_samples.reshape(-1, n_terr)
            n_obs_i = np.asarray(n_obs, dtype=int)
            y_obs_f = np.asarray(y_obs, dtype=float)

            # Beta-Binomial generative draw matches the model's likelihood.
            # If kappa is absent (older trace) fall back to a plain
            # Binomial, noting it understates dispersion.
            rng = np.random.default_rng(42)
            if 'kappa' in post:
                kappa_flat = post['kappa'].values.reshape(-1)
                a = p_flat * kappa_flat[:, np.newaxis]
                b = (1.0 - p_flat) * kappa_flat[:, np.newaxis]
                theta = rng.beta(a, b)
                y_rep = rng.binomial(n_obs_i[np.newaxis, :], theta)
            else:
                logger.info("PPC: no kappa in trace; using Binomial y_rep (legacy).")
                y_rep = rng.binomial(n_obs_i[np.newaxis, :], p_flat)

            # T1: proportion of zero-count territories.
            t_obs_zeros = float(np.mean(y_obs_f == 0))
            t_rep_zeros = np.mean(y_rep == 0, axis=1)
            p_zeros = float(np.mean(t_rep_zeros >= t_obs_zeros))

            # T2: 95th percentile of counts (tail / extremes).
            t_obs_p95 = float(np.percentile(y_obs_f, 95))
            t_rep_p95 = np.percentile(y_rep, 95, axis=1)
            p_p95 = float(np.mean(t_rep_p95 >= t_obs_p95))

            # T3: overall mean rate (calibration of the central tendency).
            n_sum = float(np.sum(n_obs_i)) if np.sum(n_obs_i) > 0 else 1.0
            t_obs_rate = float(np.sum(y_obs_f) / n_sum)
            t_rep_rate = y_rep.sum(axis=1) / n_sum
            p_rate = float(np.mean(t_rep_rate >= t_obs_rate))

            results = {
                'p_value_zeros': p_zeros,
                'p_value_p95':   p_p95,
                'p_value_rate':  p_rate,
            }

            is_ok = all(0.05 < p < 0.95 for p in (p_zeros, p_p95, p_rate))
            primary = p_p95  # tail behaviour matters most for hotspot detection

            logger.info(
                f"PPC p-values: zeros={p_zeros:.3f}, p95={p_p95:.3f}, "
                f"mean_rate={p_rate:.3f} ({'OK' if is_ok else 'CHECK'})"
            )

            return float(primary), is_ok, results

        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"Could not calculate PPC p-values: {e}")
            return None, None, {}
