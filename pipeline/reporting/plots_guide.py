"""
Diagnostic-plots interpretation guide.

A long, static template that explains each diagnostic plot (PPC, pair,
forest, trace, energy, divergence-scatter) to a reader who is not
otherwise familiar with Bayesian model diagnostics. Written into
DIAGNOSTIC_PLOTS_GUIDE.txt at the end of every run so the contents
of the diagnostics directory are interpretable on their own.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_diagnostic_plots_guide(output_dir: Path) -> None:
    """
    Generate a text guide explaining diagnostic plots.

    Args:
        output_dir: Directory where to save the guide
    """
    guide_content = """
================================================================================
DIAGNOSTIC PLOTS INTERPRETATION GUIDE
================================================================================
Generated: {timestamp}

This guide explains the diagnostic plots generated during Bayesian model analysis.
Each plot helps assess different aspects of model quality and reliability.

================================================================================
1. POSTERIOR PREDICTIVE CHECK (PPC) PLOT
================================================================================

WHAT IT SHOWS:
- Compares observed data (blue histogram) with model predictions (gray lines)
- Each gray line represents one sample from the posterior predictive distribution
- Shows how well the model can reproduce the observed data patterns

HOW TO INTERPRET:
[OK] GOOD: Observed data (blue) falls within the range of predicted distributions (gray)
[OK] GOOD: Blue histogram overlaps well with the gray prediction envelope
[FAIL] BAD: Blue histogram is far outside the gray prediction range
[FAIL] BAD: Systematic mismatch between observed and predicted patterns

WHAT IT MEANS:
- Good fit: Model captures the data-generating process well
- Poor fit: Model is missing important features or patterns in the data

EXAMPLES OF PROBLEMS:
1. Observed data consistently higher than predictions
   → Model underestimates infection rates
   → SOLUTION: Check if covariates are missing or model specification is wrong

2. Observed data has different shape than predictions
   → Model doesn't capture the distribution correctly
   → SOLUTION: Consider zero-inflated model or different likelihood

3. Very wide prediction envelope
   → High uncertainty in model predictions
   → SOLUTION: May need more data or stronger priors

================================================================================
2. PAIR PLOT (PARAMETER CORRELATIONS)
================================================================================

WHAT IT SHOWS:
- Relationships between model parameters (mu_alpha, sigma_alpha, mu_beta, etc.)
- Diagonal: Distribution of each parameter (should be smooth, bell-shaped)
- Off-diagonal: Scatter plots showing correlations between parameters

HOW TO INTERPRET:
[OK] GOOD: Smooth, unimodal distributions on diagonal
[OK] GOOD: Elliptical or circular scatter patterns (weak correlations)
[FAIL] BAD: Multimodal distributions (multiple peaks)
[FAIL] BAD: Strong linear correlations (elongated ellipses)
[FAIL] BAD: Banana-shaped or curved patterns

WHAT IT MEANS:
- Smooth distributions: Parameters are well-identified
- Strong correlations: Parameters are confounded (hard to estimate separately)
- Multimodal: Multiple solutions exist (convergence issues)

EXAMPLES OF PROBLEMS:
1. Banana-shaped correlation between mu_alpha and sigma_alpha
   → Centered parametrization issue
   → SOLUTION: Use non-centered parametrization

2. Very wide parameter distributions
   → Weak information in data
   → SOLUTION: Use informative priors or collect more data

3. Multiple peaks in parameter distribution
   → Model hasn't converged or has multiple modes
   → SOLUTION: Increase sampling iterations or check model specification

================================================================================
3. FOREST PLOT (PARAMETER ESTIMATES)
================================================================================

WHAT IT SHOWS:
- Point estimates (dots) and credible intervals (horizontal lines) for each parameter
- Shows uncertainty in parameter estimates
- Vertical line at zero helps identify significant effects

HOW TO INTERPRET:
[OK] GOOD: Narrow credible intervals (precise estimates)
[OK] GOOD: Intervals don't cross zero for parameters that should be significant
[FAIL] BAD: Very wide intervals (high uncertainty)
[FAIL] BAD: Intervals include zero when effect should be present

WHAT IT MEANS:
- Narrow intervals: Parameter is well-estimated from the data
- Wide intervals: High uncertainty, need more data
- Interval excludes zero: Parameter has a significant effect

EXAMPLES OF PROBLEMS:
1. All intervals very wide
   → Insufficient data or weak priors
   → SOLUTION: Collect more data or use informative priors

2. beta_hist interval includes zero
   → Historical baseline doesn't predict current infections
   → SOLUTION: Check data quality or consider different predictors

3. Asymmetric intervals
   → Non-normal posterior (common and often OK)
   → ACTION: Check if parameter is constrained (e.g., sigma must be positive)

================================================================================
4. TRACE PLOT (MCMC CONVERGENCE)
================================================================================

WHAT IT SHOWS:
- How parameter values evolved during MCMC sampling
- Each colored line represents one chain
- Shows mixing and convergence of the sampler

HOW TO INTERPRET:
[OK] GOOD: All chains overlap and look like "fuzzy caterpillars"
[OK] GOOD: No trends or drifts over time
[OK] GOOD: Chains explore the same region of parameter space
[FAIL] BAD: Chains don't overlap (different regions)
[FAIL] BAD: Trends or drifts (not stationary)
[FAIL] BAD: Chains stuck in one region (poor mixing)

WHAT IT MEANS:
- Good mixing: Sampler efficiently explores the posterior
- Poor mixing: Sampler is stuck or hasn't converged
- Separated chains: Convergence failure

EXAMPLES OF PROBLEMS:
1. One chain in different region than others
   → Convergence failure
   → SOLUTION: Increase tune iterations or check initialization

2. Slow drift upward or downward
   → Chain hasn't reached stationary distribution
   → SOLUTION: Increase tune iterations

3. Chains look like "steps" instead of fuzzy
   → Poor mixing, high autocorrelation
   → SOLUTION: Reparametrize model or increase target_accept

================================================================================
5. ENERGY PLOT (HMC DIAGNOSTICS)
================================================================================

WHAT IT SHOWS:
- Distribution of energy levels during Hamiltonian Monte Carlo sampling
- Compares energy at different stages of sampling
- Helps detect problems with the sampler geometry

HOW TO INTERPRET:
[OK] GOOD: Energy and energy_transition distributions overlap well
[OK] GOOD: Similar shapes and ranges
[FAIL] BAD: Large gap between distributions
[FAIL] BAD: Very different shapes

WHAT IT MEANS:
- Good overlap: HMC sampler is working efficiently
- Poor overlap: Sampler has difficulty exploring the posterior
- Often related to divergences

EXAMPLES OF PROBLEMS:
1. Large gap between energy distributions
   → Sampler struggling with posterior geometry
   → SOLUTION: Increase target_accept to 0.95 or 0.99

2. Energy distribution has long tail
   → Some regions of posterior are hard to sample
   → SOLUTION: Reparametrize model (try non-centered)

================================================================================
6. DIVERGENCES SCATTER PLOT
================================================================================

WHAT IT SHOWS:
- Locations in parameter space where divergences occurred
- Divergences are red dots, normal samples are blue
- Shows which parameter combinations cause sampling problems

HOW TO INTERPRET:
[OK] GOOD: No red dots (no divergences)
[OK] GOOD: Few scattered red dots (<1% of samples)
[FAIL] BAD: Many red dots (>5% of samples)
[FAIL] BAD: Red dots clustered in specific region

WHAT IT MEANS:
- No divergences: Sampler working well
- Few divergences: Minor issues, results likely OK
- Many divergences: Serious problems, results may be biased

EXAMPLES OF PROBLEMS:
1. Divergences clustered near boundary (e.g., sigma near 0)
   → Posterior has difficult geometry near constraints
   → SOLUTION: Use non-centered parametrization

2. Divergences scattered throughout
   → General sampling difficulties
   → SOLUTION: Increase target_accept or add more tune iterations

3. >10% divergences
   → Results may be unreliable
   → SOLUTION: Must fix before trusting results (reparametrize or increase target_accept)

================================================================================
GENERAL TROUBLESHOOTING WORKFLOW
================================================================================

1. CHECK CONVERGENCE FIRST (Trace Plot, R-hat values)
   - If chains haven't converged, other diagnostics are meaningless
   - SOLUTION: Increase tune/draw iterations

2. CHECK FOR DIVERGENCES (Divergences Plot, Energy Plot)
   - Divergences indicate biased sampling
   - SOLUTION: Increase target_accept or reparametrize

3. CHECK MODEL FIT (PPC Plot)
   - Does model reproduce observed data?
   - SOLUTION: Adjust model specification if poor fit

4. CHECK PARAMETER ESTIMATES (Forest Plot, Pair Plot)
   - Are parameters well-identified?
   - SOLUTION: Add informative priors or collect more data

================================================================================
QUICK REFERENCE: DIAGNOSTIC QUALITY
================================================================================

EXCELLENT MODEL:
- R-hat < 1.01 for all parameters
- ESS > 400 for all parameters
- 0 divergences
- PPC p-value between 0.05 and 0.95
- Trace plots show good mixing
- Energy distributions overlap well

ACCEPTABLE MODEL:
- R-hat < 1.05 for all parameters
- ESS > 100 for all parameters
- <1% divergences
- PPC p-value between 0.01 and 0.99
- Trace plots converged but may have some autocorrelation

PROBLEMATIC MODEL (DO NOT USE):
- R-hat > 1.05 for any parameter
- ESS < 100 for any parameter
- >5% divergences
- PPC p-value < 0.01 or > 0.99
- Trace plots show non-convergence

================================================================================
ADDITIONAL RESOURCES
================================================================================

For more information on Bayesian diagnostics:
- PyMC documentation: https://www.pymc.io/
- Gelman et al. "Bayesian Data Analysis" (2013)
- Betancourt "A Conceptual Introduction to Hamiltonian Monte Carlo" (2017)
- McElreath "Statistical Rethinking" (2020)

================================================================================
END OF GUIDE
================================================================================
""".format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    guide_path = output_dir / "DIAGNOSTIC_PLOTS_GUIDE.txt"
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)

    logger.info(f"[OK] Diagnostic plots guide saved: {guide_path}")

