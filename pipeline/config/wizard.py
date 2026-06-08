"""
Interactive model-configuration wizard.

The :class:`ModelConfigurationWizard` is the single entry-point the
orchestrator calls before fitting any model. With a TTY attached it asks
two questions (data-source type, analysis mode) and shows the proposed
configuration before sampling starts; without a TTY (or if anything in
the interactive path raises) it falls back to a deterministic automatic
configuration so headless / CI runs do not stall.

The Truncated-Binomial threshold lives in
:data:`pipeline.constants.DEFAULT_TRUNCATED_BINOMIAL_STRUCTURAL_ZEROS_PCT`
so the rule here, the audit-trail message and the report agree on the
same number.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional

from pipeline.constants import DEFAULT_TRUNCATED_BINOMIAL_STRUCTURAL_ZEROS_PCT

logger = logging.getLogger(__name__)


class ModelConfigurationWizard:
    """Interactive wizard to determine optimal model configuration based on data characteristics."""

    @staticmethod
    def run_wizard(n_active_sites: int, pct_structural_zeros: float,
                   level_name: str, cli_args: Optional[Dict] = None,
                   config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run interactive wizard to determine model configuration.

        Args:
            n_active_sites: Number of active testing sites
            pct_structural_zeros: Percentage of territories without testing sites
            level_name: Administrative level name (e.g., "Community", "District")
            cli_args: CLI arguments if provided (to skip wizard)
            config: Configuration dict (to check for data_type setting)

        Returns:
            Dict with configuration: {
                'use_hurdle': bool,
                'hurdle_threshold': float,
                'use_loo_ic': bool
            }
        """
        # If CLI arguments provided, skip wizard
        if cli_args:
            logger.info("Using CLI-provided configuration, skipping wizard")
            return {
                'use_hurdle': cli_args.get('use_hurdle', False),
                'hurdle_threshold': cli_args.get('hurdle_threshold', 70.0),
                'use_loo_ic': cli_args.get('use_loo_ic', False)
            }

        # Try interactive mode first
        try:
            # Check if we can read from stdin
            if sys.stdin.isatty():
                # Display header
                print("\n" + "="*70)
                print("MODEL CONFIGURATION WIZARD")
                print("="*70)
                print(f"\nAnalysis level: {level_name}")
                print(f"Active testing sites: {n_active_sites}")
                print(f"Territories without sites: {pct_structural_zeros:.1f}%")
                print("\n" + "-"*70)

                # Question 1: Data type
                print("\n1. Your data represents:")
                print("   a) Testing site location (GPS coordinates of facilities)")
                print("   b) Patient residence (registered address)")
                print("\n   In most cases this is (a) - testing site location")

                data_type = ModelConfigurationWizard._get_choice(['a', 'b'], default='a')

                # Question 2: Analysis mode
                print("\n2. Analysis mode:")
                print("   a) Standard (recommended for practical use)")
                print("   b) Research (detailed model comparison, for scientific publications)")
                print("\n   Choose (b) only if preparing a scientific publication")

                analysis_mode = ModelConfigurationWizard._get_choice(['a', 'b'], default='a')

                # Determine configuration
                config = ModelConfigurationWizard._determine_config(
                    data_type, analysis_mode, n_active_sites, pct_structural_zeros
                )

                # Display recommendation
                ModelConfigurationWizard._display_recommendation(config, pct_structural_zeros)

                # Confirm
                print("\nProceed with this configuration? [Y/n]: ", end='')
                response = input().strip().lower()
                if response and response not in ['y', 'yes', '']:
                    print("\nAborted by user")
                    raise KeyboardInterrupt("User aborted configuration")

                return config
            else:
                # Non-interactive: use automatic configuration
                raise EOFError("Non-interactive mode")

        except (EOFError, KeyboardInterrupt):
            # Automatic mode: determine configuration based on data characteristics
            logger.info(f"Automatic configuration for {level_name}:")
            result_config = ModelConfigurationWizard._get_default_config(
                n_active_sites, pct_structural_zeros, config
            )
            logger.info(f"  - spatial_structure: exchangeable")
            logger.info(f"  - use_hurdle: {result_config['use_hurdle']}")
            logger.info(f"  - use_loo_ic: {result_config['use_loo_ic']}")
            return result_config
        except Exception as e:
            logger.warning(f"Interactive wizard failed: {e}")
            logger.info("Falling back to automatic configuration")
            result_config = ModelConfigurationWizard._get_default_config(
                n_active_sites, pct_structural_zeros, config
            )
            return result_config

    @staticmethod
    def _get_choice(options: list, default: str = None) -> str:
        """Get user choice from options."""
        while True:
            print(f"\nYour choice [{'/'.join(options)}]: ", end='')
            try:
                choice = input().strip().lower()
                if not choice and default:
                    return default
                if choice in options:
                    return choice
                print(f"Invalid choice. Please enter one of: {', '.join(options)}")
            except (EOFError, KeyboardInterrupt):
                print("\nAborted by user")
                raise KeyboardInterrupt("User aborted input")

    @staticmethod
    def _determine_config(data_type: str, analysis_mode: str,
                         n_active_sites: int, pct_structural_zeros: float) -> Dict:
        """Determine configuration based on answers."""
        config = {}

        # Spatial structure
        # Exchangeable model (facility-based data)
        config['spatial_structure'] = 'exchangeable'

        # Truncated-Binomial recommendation (automatic, based on structural
        # zeros). The single threshold constant is at module scope so the
        # rule here, the audit-trail message and the report all agree.
        threshold = DEFAULT_TRUNCATED_BINOMIAL_STRUCTURAL_ZEROS_PCT
        config['use_hurdle'] = pct_structural_zeros >= threshold
        config['hurdle_threshold'] = threshold

        # LOO-IC
        if analysis_mode == 'b':
            # Research mode → use LOO-IC
            config['use_loo_ic'] = True
        else:
            # Standard mode → fast heuristic
            config['use_loo_ic'] = False

        return config

    @staticmethod
    def _display_recommendation(config: Dict, pct_structural_zeros: float):
        """Display recommended configuration."""
        print("\n" + "="*70)
        print("RECOMMENDED CONFIGURATION")
        print("="*70)

        # Spatial structure
        print(f"\n[OK] Spatial structure: EXCHANGEABLE (facility-based data)")

        # Truncated Binomial (active sites only) -- historical CLI flag
        # ``--use-hurdle`` is preserved for backwards compatibility.
        if config['use_hurdle']:
            print(f"\n[OK] Truncated Binomial (active sites): ENABLED")
            print(f"   (Automatic: {pct_structural_zeros:.1f}% territories without sites)")
        else:
            print(f"\n[OK] Truncated Binomial (active sites): DISABLED")
            print(f"   ({pct_structural_zeros:.1f}% structural zeros < 70% threshold)")

        # LOO-IC
        if config['use_loo_ic']:
            print(f"\n[OK] Model selection: RESEARCH MODE (LOO-IC comparison)")
            print(f"   Estimated time: ~20-30 minutes")
        else:
            print(f"\n[OK] Model selection: STANDARD (fast heuristic)")
            print(f"   Estimated time: ~5-10 minutes")

        print("\n" + "="*70)

    @staticmethod
    def _get_default_config(n_active_sites: int, pct_structural_zeros: float,
                           config: Optional[Dict] = None) -> Dict:
        """Get default configuration for non-interactive mode.

        Exchangeable model — correct for facility-based surveillance.
        """
        result = {
            'use_hurdle': pct_structural_zeros >= 70.0,
            'hurdle_threshold': 70.0,
            'use_loo_ic': False  # Fast mode by default
        }

        logger.info(f"Automatic configuration:")
        logger.info(f"  - spatial_structure=exchangeable")
        logger.info(f"  - use_hurdle={result['use_hurdle']}")
        logger.info(f"  - use_loo_ic={result['use_loo_ic']}")

        return result
