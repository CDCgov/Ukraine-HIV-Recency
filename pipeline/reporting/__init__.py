"""Reporting helpers: dashboards, plot-interpretation guides."""

from pipeline.reporting.dashboard import SummaryDashboard
from pipeline.reporting.plots_guide import generate_diagnostic_plots_guide
from pipeline.reporting.disclaimer import (
    get_facility_based_disclaimer,
    add_disclaimer_to_metadata,
)
from pipeline.reporting.summary import print_summary
from pipeline.reporting.oblast_labels import add_oblast_labels
from pipeline.reporting.diagnostics_excel import save_diagnostics
from pipeline.reporting.territory_info import (
    add_admin_territory_info,
    add_hex_territory_info,
)
from pipeline.reporting.boundary_only_map import render_boundary_only_map
from pipeline.reporting.reliability_map import render_reliability_map
from pipeline.reporting.anomaly_map import render_anomaly_map
from pipeline.reporting.watchlist_map import render_watchlist_map
from pipeline.reporting.excel_report import write_report

__all__ = [
    "SummaryDashboard",
    "generate_diagnostic_plots_guide",
    "get_facility_based_disclaimer",
    "add_disclaimer_to_metadata",
    "print_summary",
    "add_oblast_labels",
    "save_diagnostics",
    "add_admin_territory_info",
    "add_hex_territory_info",
    "render_boundary_only_map",
    "render_reliability_map",
    "render_anomaly_map",
    "render_watchlist_map",
    "write_report",
]
