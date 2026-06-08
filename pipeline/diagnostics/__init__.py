"""
Diagnostics helpers for the HIV hotspot pipeline.

Posterior predictive checks, convergence summaries, reliability scoring,
multicollinearity checks and the diagnostic plotters that turn a fitted
trace into the numbers and figures the reports consume.
"""
from pipeline.diagnostics.ppc import PPCCalculator
from pipeline.diagnostics.reliability import ReliabilityScoreCalculator
from pipeline.diagnostics.interpreter import DiagnosticInterpreter
from pipeline.diagnostics.data_quality import DataQualityChecker
from pipeline.diagnostics.plots import BayesianDiagnosticsFixed, DiagnosticPlotter
from pipeline.diagnostics.bayesian import (
    calculate_bayesian_diagnostics,
    calculate_covariates_diagnostics,
    calculate_covariates_diagnostics_stratified,
)
from pipeline.diagnostics.multicollinearity import check_multicollinearity

__all__ = [
    "PPCCalculator",
    "ReliabilityScoreCalculator",
    "DiagnosticInterpreter",
    "DataQualityChecker",
    "BayesianDiagnosticsFixed",
    "DiagnosticPlotter",
    "calculate_bayesian_diagnostics",
    "calculate_covariates_diagnostics",
    "calculate_covariates_diagnostics_stratified",
    "check_multicollinearity",
]
