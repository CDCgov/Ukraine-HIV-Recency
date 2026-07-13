"""
Interactive configuration prompts.

The :class:`InteractiveConfig` collects the wider analysis parameters
that the wizard does not -- analysis type (standard vs iterative),
hex resolutions, time periods, parametrization, model-selection mode --
through a sequence of stdin prompts. Each prompt has a typed default so
non-interactive callers (or the orchestrator's CI path) get a sensible
configuration without raising.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dateutil.relativedelta import relativedelta

from pipeline.aggregation.periods import (
    BASELINE_DATA_FLOOR,
    baseline_months_for,
    months_between,
)


class InteractiveConfig:
    """Handles interactive configuration of analysis parameters."""

    @staticmethod
    def choose_analysis_type() -> str:
        """Choose between standard and iterative hotspot search."""
        print("\n" + "=" * 60)
        print("ANALYSIS TYPE")
        print("=" * 60)
        print("1 - Standard analysis (single time period, all models)")
        print("2 - Iterative hotspot search (sliding window, Bayesian only, Res4)")
        print()
        print("Iterative mode:")
        print("  - Runs Bayesian non-centered model on Res4 hexagons")
        print("  - 3-month analysis period, 12-month baseline")
        print("  - Steps backward 1 month at a time through entire dataset")
        print("  - Generates aggregated hotspots report")
        print("  - [WARN] May take 2-4 hours depending on dataset size")

        while True:
            choice = input("\nEnter your choice (1/2): ").strip()
            if choice == '1':
                return 'standard'
            elif choice == '2':
                return 'iterative'
            print("Invalid choice. Please try again.")

    @staticmethod
    def choose_hex_resolutions() -> List[int]:
        """Return the H3 resolution to analyse.

        The pipeline standardised on res3 (larger hexagons, more testing
        facilities per unit). Finer resolutions are not offered because most
        finer hexes hold zero or one active site, so the hexagon is then just a
        relabelled facility rather than a meaningful analytical unit.
        """
        print("\n" + "=" * 60)
        print("H3 HEXAGON RESOLUTION")
        print("=" * 60)
        print("Using res3 (larger hexagons, more facilities per unit).")
        return [3]

    @staticmethod
    def choose_levels() -> List:
        """Choose one or more analysis levels: res3 hexagons and/or adm1 oblasts.

        Returns a list mixing the H3 resolution integer 3 and the string
        'Oblast' for ADM1, in the canonical order res3, adm1. The pipeline
        standardised on res3 for the hexagon unit; finer resolutions are not
        offered because most finer hexes hold zero or one active site.
        """
        print("\n" + "=" * 60)
        print("ANALYSIS LEVELS")
        print("=" * 60)
        print("1 - res3       H3 hexagons (larger unit, more facilities per hexagon)")
        print("2 - adm1       administrative oblasts (coarsest; most events per unit)")
        print("3 - res3,adm1  both")
        print("Enter an index (1/2/3), or the names comma-separated (e.g. res3,adm1).")

        mapping = {'res3': 3, 'adm1': 'Oblast'}
        order = ['res3', 'adm1']
        index_map = {'1': ['res3'], '2': ['adm1'], '3': ['res3', 'adm1']}
        while True:
            raw = input("\nLevels [default: 3 = res3,adm1]: ").strip().lower()
            if raw == '':
                return [3, 'Oblast']
            tokens = index_map[raw] if raw in index_map else \
                [t.strip() for t in raw.split(',') if t.strip()]
            if tokens and all(t in mapping for t in tokens):
                return [mapping[t] for t in order if t in tokens]
            print("Invalid choice. Enter 1, 2 or 3, or names like res3,adm1.")

    @staticmethod
    def choose_iterative_analysis_window() -> int:
        """Prompt for the iterative analysis-window length in months (3 or 6).

        A 6-month window accumulates more recent infections per cell (higher
        reliability on sparse recency data) and aligns better with the assay's
        recency-detection window; a 3-month window gives finer temporal
        resolution. The baseline is the (non-overlapping) 12 months before.
        """
        print("\n" + "=" * 60)
        print("ITERATIVE ANALYSIS WINDOW")
        print("=" * 60)
        print("3  - 3-month window  (baseline 12m)")
        print("6  - 6-month window  (baseline 12m)")
        print("9  - 9-month window  (baseline 18m)")
        print("12 - 12-month window (baseline 24m)")
        print("Longer windows accumulate more recent infections per cell "
              "(higher reliability) but fewer iterations fit.")

        while True:
            choice = input("\nEnter your choice (3/6/9/12) [default: 3]: ").strip()
            if choice == '' or choice == '3':
                return 3
            if choice in ('6', '9', '12'):
                return int(choice)
            print("Invalid choice. Please enter 3, 6, 9 or 12.")

    @staticmethod
    def choose_confidence_level() -> float:
        """Prompt for the posterior-probability confidence level for hotspot calls.

        A hotspot is called when the posterior probability that a cell exceeds
        the national rate clears this level (FDR-controlled). The default 0.80 is
        the Richardson et al. (2004) disease-mapping decision rule D(0.8, 1) and
        was confirmed as the sensitivity/false-positive knee by a simulation on
        the real site volumes. 0.85-0.95 are progressively more conservative
        (fewer false alerts, lower sensitivity on sparse counts). The presence
        gate (>=2 recent events) holds at every level, so no single-event cell is
        ever flagged.
        """
        print("\n" + "=" * 60)
        print("CONFIDENCE LEVEL (hotspot calls)")
        print("=" * 60)
        print("80 - Richardson D(0.8,1) rule; simulation-calibrated (recommended)")
        print("85 - near-zero false alerts, slightly lower sensitivity")
        print("90 / 95 - conservative confirmatory bars")
        while True:
            choice = input("\nEnter your choice (80/85/90/95) [default: 80]: ").strip()
            mapping = {'': 0.80, '80': 0.80, '85': 0.85, '90': 0.90, '95': 0.95}
            if choice in mapping:
                return mapping[choice]
            print("Invalid choice. Please enter 80, 85, 90 or 95.")

    @staticmethod
    def choose_analysis_period() -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Prompt for a single-mode analysis period (window <= 12 months).

        Warns up front that the window cannot exceed 12 months, re-asks until
        the entered period is <= 12 months, and rejects periods whose derived
        baseline would start before the 2023 floor (asking for a later start).
        The baseline length is derived automatically (1-6m->12, 7-9m->18,
        10-12m->24).
        """
        print("\n" + "=" * 60)
        print("ANALYSIS PERIOD")
        print("=" * 60)
        print("The analysis window must NOT exceed 12 months.")
        print("The baseline is set automatically from the window length")
        print("(1-6m -> 12m, 7-9m -> 18m, 10-12m -> 24m) and may not start")
        print(f"before {BASELINE_DATA_FLOOR.date()}.")

        while True:
            try:
                start = pd.to_datetime(input("\nStart date (YYYY-MM-DD): ").strip())
                end = pd.to_datetime(input("End date (YYYY-MM-DD): ").strip())
            except (ValueError, TypeError) as e:
                print(f"Error parsing dates: {e}. Use YYYY-MM-DD.")
                continue

            if start >= end:
                print("Error: start date must be before end date.")
                continue

            months = months_between(start, end)
            if months > 12:
                print(f"Error: the analysis window is {months} months — the maximum is 12. "
                      "Choose a shorter period.")
                continue

            baseline_months = baseline_months_for(months)
            b_start = start - relativedelta(months=baseline_months)
            if b_start < BASELINE_DATA_FLOOR:
                earliest = BASELINE_DATA_FLOOR + relativedelta(months=baseline_months)
                print(f"Error: a {months}-month window needs a {baseline_months}-month baseline, "
                      f"which would start {b_start.date()} — before the {BASELINE_DATA_FLOOR.date()} "
                      f"floor. Choose a start date on or after {earliest.date()}.")
                continue

            print(f"  -> {months}-month window, {baseline_months}-month baseline "
                  f"(from {b_start.date()}).")
            return start, end

    @staticmethod
    def choose_iterative_resolution() -> int:
        """Return the single H3 resolution used in iterative mode.

        The pipeline standardised on res3 (larger hexagons, more tests per cell,
        higher reliability). Finer resolutions are not offered because most finer
        hexes hold zero or one active site.
        """
        print("\n" + "=" * 60)
        print("ITERATIVE H3 RESOLUTION")
        print("=" * 60)
        print("Using res3 (larger hexagons, more tests per hexagon).")
        return 3

    @staticmethod
    def choose_period() -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Prompt for the analysis window; re-asks until end is after start."""
        print("\n" + "=" * 60)
        print("ANALYSIS PERIOD")
        print("=" * 60)

        while True:
            try:
                start_str = input("Start date (YYYY-MM-DD): ").strip()
                end_str = input("End date (YYYY-MM-DD): ").strip()

                start = pd.to_datetime(start_str)
                end = pd.to_datetime(end_str)

                if start >= end:
                    print("Error: Start date must be before end date. Please try again.")
                    continue

                return start, end
            except ValueError as e:
                print(f"Error parsing dates: {e}. Please use YYYY-MM-DD format.")

    @staticmethod
    def choose_iterative_date_range() -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Choose date range for iterative analysis."""
        print("\n" + "=" * 60)
        print("ITERATIVE ANALYSIS DATE RANGE")
        print("=" * 60)
        print("Select the date range for sliding window analysis.")
        print("The iterative analysis will only use data within this range.")
        print()

        while True:
            try:
                start_str = input("Start date (YYYY-MM-DD): ").strip()
                end_str = input("End date (YYYY-MM-DD): ").strip()

                start = pd.to_datetime(start_str)
                end = pd.to_datetime(end_str)

                if start >= end:
                    print("Error: Start date must be before end date. Please try again.")
                    continue

                # Check minimum range for iterative analysis (need at least 15 months for 3-month window + 12-month baseline)
                months_diff = (end.year - start.year) * 12 + (end.month - start.month)
                if months_diff < 15:
                    print(f"Warning: Date range is only {months_diff} months.")
                    print("Iterative analysis requires at least 15 months (3-month analysis + 12-month baseline).")
                    confirm = input("Continue anyway? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue

                return start, end
            except ValueError as e:
                print(f"Error parsing dates: {e}. Please use YYYY-MM-DD format.")

    @staticmethod
    def choose_parametrization() -> str:
        """Prompt for non-centered (default) vs centered parametrization."""
        print("\n" + "=" * 60)
        print("BAYESIAN PARAMETRIZATION")
        print("=" * 60)
        print("1 - Non-centered (default, recommended)")
        print("2 - Centered (for large samples only)")
        print("\nInfo:")
        print("  Non-centered: Better for small samples, reduces divergences (RECOMMENDED)")
        print("  Centered: Standard parametrization, only for large samples (>50 territories)")
        print("  Note: System will auto-select based on data if configured")

        while True:
            choice = input("\nEnter your choice (1/2) [default: 1]: ").strip()
            if choice == '' or choice == '1':
                return 'non_centered'
            elif choice == '2':
                return 'centered'
            print("Invalid choice. Please try again.")

    @staticmethod
    def choose_model_selection() -> str:
        """Choose which model(s) to run: bayesian, bayesian_covariates, or both.

        The old 'Auto' option (which used to pick between the retired GLM and the
        Bayesian model) is gone: the Bayesian model is always the detector, so the
        only choice is whether to also run the explanatory covariate layer.
        """
        print("\n" + "=" * 60)
        print("MODEL SELECTION")
        print("=" * 60)
        print("\n1 - Bayesian only (hierarchical, no covariates)")
        print("2 - Bayesian with Covariates (stratified risk groups)")
        print("3 - Both (Bayesian and Bayesian with Covariates)")
        print("\nInfo:")
        print("  Bayesian only:    the primary detector; best for sparse data, many zeros, small N")
        print("  Bayesian Cov:     explanatory layer that accounts for risk-group composition")
        print("  Both:             run the detector and the explanatory layer side by side")

        while True:
            try:
                choice = input("\nEnter your choice (1/2/3) [default: 3 = both]: ").strip()
                if choice == '1':
                    return 'bayesian'
                elif choice == '2':
                    return 'bayesian_covariates'
                elif choice == '' or choice == '3':
                    return 'both'
                print("Invalid choice. Please enter 1, 2, or 3.")
            except (EOFError, KeyboardInterrupt):
                return 'both'

    @staticmethod
    def ask_overwrite_config(config_path: str) -> bool:
        """Ask if user wants to save configuration."""
        print("\n" + "=" * 60)
        response = input(f"Save configuration to {config_path}? (y/n): ").strip().lower()
        return response == 'y'
