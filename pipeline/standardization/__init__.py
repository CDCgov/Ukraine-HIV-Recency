"""
Standardisation helpers for the HIV hotspot pipeline.

Hosts the SMR / SIR computation (``smr_sir``), the Empirical-Bayes
shrinkage helper for the historical baseline rate, the pooled-SE Z-scores
(``z_scores``) and the FDR-controlled threshold picker (``thresholds``).
"""
from pipeline.standardization.thresholds import bayesian_fdr_threshold
from pipeline.standardization.smr_sir import eb_baseline_rate, compute_smr_sir
from pipeline.standardization.z_scores import calculate_z_scores
from pipeline.standardization.group_attribution import (
    period_composition,
    oblast_group_rates,
    oblast_standardization,
    escalating_group_rates,
)

__all__ = [
    "bayesian_fdr_threshold",
    "eb_baseline_rate",
    "compute_smr_sir",
    "calculate_z_scores",
    "period_composition",
    "oblast_group_rates",
    "oblast_standardization",
    "escalating_group_rates",
]
