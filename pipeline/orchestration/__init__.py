"""Orchestration helpers (pure functions over orchestrator state)."""

from pipeline.orchestration.diagnostics_check import check_diagnostics
from pipeline.orchestration.dashboard_collect import collect_dashboard_data
from pipeline.orchestration.current_results import collect_current_results
from pipeline.orchestration.model_comparison import generate_model_comparison
from pipeline.orchestration.model_comparison_entry import build_model_comparison_entry
from pipeline.orchestration.config_paths import resolve_paths
from pipeline.orchestration.config_validation import validate_config
from pipeline.orchestration.recommendations import generate_recommendations
from pipeline.orchestration.interactive_setup import run_interactive_setup
from pipeline.orchestration.calibrate_min_tests import calibrate_min_tests
from pipeline.orchestration.calibrate_sigma_multipliers import calibrate_sigma_multipliers
from pipeline.orchestration.iterative_hotspots_report import generate_iterative_hotspots_report
from pipeline.orchestration.iterative_windows import prepare_iterative_windows
from pipeline.orchestration.first_window_calibration import run_first_window_calibration
from pipeline.orchestration.post_pipeline import create_summary_dashboard, run_historical_comparison
from pipeline.orchestration.audit_trail_reports import generate_audit_trail_reports
from pipeline.orchestration.skipped_windows_table import write_skipped_windows_table
from pipeline.orchestration.data_quality_audit import assess_data_quality
from pipeline.orchestration.wizard_and_audit import run_wizard_and_record_decisions
from pipeline.orchestration.specification_analysis import analyze_specification
from pipeline.orchestration.iterative_loop import run_iterative_loop
from pipeline.orchestration.bayesian_dispatch import run_bayesian_dispatch
from pipeline.orchestration.interpret_bayesian import interpret_bayesian_diagnostics
from pipeline.orchestration.bayesian_plots import create_bayesian_plots
from pipeline.orchestration.interpret_bayesian_covariates import interpret_bayesian_covariates_diagnostics
from pipeline.orchestration.bayesian_covariates_plots import create_bayesian_covariates_plots
from pipeline.orchestration.bayesian_covariates_dispatch import run_bayesian_covariates_dispatch
from pipeline.orchestration.finalize_model_choice import finalize_model_choice
from pipeline.orchestration.comparison_report import generate_comparison_report

__all__ = [
    "check_diagnostics",
    "collect_dashboard_data",
    "collect_current_results",
    "generate_model_comparison",
    "build_model_comparison_entry",
    "resolve_paths",
    "validate_config",
    "generate_recommendations",
    "run_interactive_setup",
    "calibrate_min_tests",
    "calibrate_sigma_multipliers",
    "generate_iterative_hotspots_report",
    "prepare_iterative_windows",
    "run_first_window_calibration",
    "create_summary_dashboard",
    "run_historical_comparison",
    "generate_audit_trail_reports",
    "write_skipped_windows_table",
    "assess_data_quality",
    "run_wizard_and_record_decisions",
    "analyze_specification",
    "run_iterative_loop",
    "run_bayesian_dispatch",
    "interpret_bayesian_diagnostics",
    "create_bayesian_plots",
    "interpret_bayesian_covariates_diagnostics",
    "create_bayesian_covariates_plots",
    "run_bayesian_covariates_dispatch",
    "finalize_model_choice",
    "generate_comparison_report",
]
