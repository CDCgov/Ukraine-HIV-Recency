"""
``pipeline`` package -- modular home for the HIV hotspot detection codebase.

The analysis is driven from the ``run_hotspots.py`` CLI entrypoint, which
is now a thin shell over this package: parses arguments, configures
logging, constructs :class:`pipeline.orchestrator.PipelineOrchestrator`,
and runs it. Almost every line of real logic lives in this package --
:mod:`pipeline.analyzers` for the Bayesian fits, :mod:`pipeline.orchestration`
for the per-stage helpers, :mod:`pipeline.standardization`, :mod:`pipeline.diagnostics`,
:mod:`pipeline.reporting`, and so on. The key modules:

    exceptions      -- the pipeline error hierarchy
    validators      -- input DataFrame / GeoDataFrame validation
    constants       -- ANALYSIS_CONSTANTS and DEFAULT_CONFIG
    logging_setup   -- console / file logger, compiler check
    standardization -- bayesian_fdr_threshold, eb_baseline_rate, compute_smr_sir
    diagnostics     -- PPCCalculator (targeted-statistic posterior predictive)
    classification  -- SIR/SMR taxonomy + single-axis exceedance classifier
    analyzers       -- the Bayesian model fits
    orchestration   -- the per-stage run helpers
"""
from pipeline.exceptions import (
    PipelineError,
    RecoverableError,
    FatalError,
    DataValidationError,
    ModelConvergenceError,
    GeometryError,
    InsufficientDataError,
)
from pipeline.validators import validate_dataframe, validate_geodataframe
from pipeline.constants import ANALYSIS_CONSTANTS, DEFAULT_CONFIG
from pipeline.logging_setup import setup_logging, check_compiler_availability
from pipeline.standardization import bayesian_fdr_threshold
from pipeline.diagnostics import PPCCalculator
from pipeline.classification import (
    HOTSPOT_LABELS,
    SMR_SIR_LABELS,
    classify_with_exceedance,
    classify_with_smr_sir,
    is_hotspot,
    add_smr_sir_counts,
)

__all__ = [
    "PipelineError",
    "RecoverableError",
    "FatalError",
    "DataValidationError",
    "ModelConvergenceError",
    "GeometryError",
    "InsufficientDataError",
    "validate_dataframe",
    "validate_geodataframe",
    "ANALYSIS_CONSTANTS",
    "DEFAULT_CONFIG",
    "setup_logging",
    "check_compiler_availability",
    "bayesian_fdr_threshold",
    "PPCCalculator",
    "HOTSPOT_LABELS",
    "SMR_SIR_LABELS",
    "classify_with_exceedance",
    "classify_with_smr_sir",
    "is_hotspot",
    "add_smr_sir_counts",
]
