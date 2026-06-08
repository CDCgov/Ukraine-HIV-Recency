"""
Centralised constants for the HIV hotspot detection pipeline.

The two dictionaries below keep the pipeline's tunable values in one
place, so they can be adjusted without touching the analysis code.

``ANALYSIS_CONSTANTS`` collects the magic numbers used by the diagnostics
and the priors (MCMC thresholds, ESS target, clip values for logit, etc.),
each annotated with its literature source. ``DEFAULT_CONFIG`` is the fall-
back configuration used by ``--test`` mode and when no JSON config is
passed on the command line.
"""


# Semantic pipeline version. Bump the minor on a methodological change
# (likelihood swap, prior reshape), the patch on a fix or polish. Written
# into the metadata of every output so results trace back to the exact
# code that produced them.
__version__ = "0.2.0"
PIPELINE_VERSION = __version__


# =============================================================================
# ANALYSIS CONSTANTS
# All magic numbers centralized with literature references.
# Adjustable for different datasets via config overrides.
# =============================================================================
ANALYSIS_CONSTANTS = {
    # --- MCMC Diagnostics (universal thresholds from literature) ---
    'ess_min_per_chain': {
        'value': 400,
        'source': 'Vehtari et al. (2021), "Rank-normalization, folding, and localization", Bayesian Analysis',
        'adjustable': False,
        'note': 'Minimum effective sample size for stable quantile estimates; '
                'matches the >400 adequacy gate used in the diagnostics'
    },
    'rhat_max': {
        'value': 1.01,
        'source': 'Vehtari et al. (2021) — tightened from traditional 1.1',
        'adjustable': False,
        'note': 'Gelman-Rhat threshold for convergence diagnosis'
    },
    'divergence_pct_strong': {
        'value': 10.0,
        'source': 'Betancourt (2017), "A Conceptual Introduction to Hamiltonian Monte Carlo"',
        'adjustable': False,
        'note': 'Above this — model must be reparametrized'
    },
    'divergence_pct_moderate': {
        'value': 5.0,
        'source': 'Betancourt (2017), Stan best practices',
        'adjustable': False,
        'note': 'Above this — increase target_accept'
    },
    'ebfmi_low_threshold': {
        'value': 0.2,
        'source': 'Betancourt (2017) — E-BFMI below 0.2 indicates poor exploration',
        'adjustable': False,
        'note': 'Below this — increase target_accept'
    },
    'max_treedepth_default': {
        'value': 10,
        'source': 'PyMC/Stan default — prevents infinite trajectories',
        'adjustable': True,
        'note': 'Can increase if treedepth exceeded frequently'
    },
    'treedepth_exceeded_pct': {
        'value': 10.0,
        'source': 'Stan best practices — if >10% transitions hit max, increase it',
        'adjustable': False,
        'note': 'Threshold for triggering treedepth adaptation'
    },

    # --- Sampling Parameters ---
    'target_accept_default': {
        'value': 0.95,
        'source': 'Stan/PyMC best practice for hierarchical models',
        'adjustable': True,
        'note': 'Lower (0.9) for simple models, higher (0.99) for complex'
    },
    'target_accept_adapt_moderate': {
        'value': 0.97,
        'source': 'PyMC adaptive sampling strategy',
        'adjustable': False,
        'note': 'Used when divergences > 5%'
    },
    'target_accept_adapt_high': {
        'value': 0.99,
        'source': 'PyMC adaptive sampling strategy',
        'adjustable': False,
        'note': 'Used when divergences > 10% or E-BFMI < 0.2'
    },

    # --- Prior Specification ---
    'sigma_hyperprior_small_sample_mult': {
        'value': 0.7,
        'source': 'Empirical — tighter priors when < 10 territories to prevent overfitting',
        'adjustable': True,
        'note': 'Can be calibrated via cross-validation on historical data'
    },
    'sigma_hyperprior_local_density_mult': {
        'value': 0.85,
        'source': 'Empirical — moderate shrinkage when avg 15-30 tests/territory',
        'adjustable': True,
        'note': 'Can be calibrated via cross-validation'
    },
    'sigma_hyperprior_se_high_mult': {
        'value': 1.3,
        'source': 'Empirical — widen prior when national SE > 50% of rate (noisy baseline)',
        'adjustable': True,
        'note': 'relative SE threshold for widening'
    },
    'sigma_hyperprior_se_moderate_mult': {
        'value': 1.15,
        'source': 'Empirical — slight widening when national SE > 30% of rate',
        'adjustable': True,
        'note': 'relative SE threshold for widening'
    },

    # --- FDR and Classification ---
    'fdr_threshold': {
        'value': 0.05,
        'source': 'Benjamini-Hochberg (1995) — standard FDR control level',
        'adjustable': True,
        'note': 'Lower = fewer false positives, higher = more sensitivity'
    },

    # --- Overdispersion ---
    'overdispersion_mild': {
        'value': 1.2,
        'source': 'Cameron & Trivedi (2013), "Regression Analysis of Count Data" — φ > 1.2 indicates mild overdispersion',
        'adjustable': False,
        'note': 'Dispersion ratio threshold for switching to Beta-Binomial'
    },
    'overdispersion_moderate': {
        'value': 1.5,
        'source': 'Cameron & Trivedi (2013) — φ > 1.5 indicates moderate overdispersion',
        'adjustable': False,
        'note': 'Used in model selection diagnostics'
    },
    'overdispersion_strong': {
        'value': 2.0,
        'source': 'Cameron & Trivedi (2013) — φ > 2.0 indicates strong overdispersion',
        'adjustable': False,
        'note': 'Used in model selection diagnostics'
    },

    # --- Data Quality ---
    'min_tests_territory': {
        'value': 20,
        'source': 'CDC HIV Surveillance Guidelines (2019) — minimum for stable rate estimation',
        'adjustable': True,
        'note': 'Below this — territory flagged as small sample'
    },
    'small_sample_threshold': {
        'value': 20,
        'source': 'Gelman & Hill (2007), "Data Analysis Using Regression" — threshold for informative priors',
        'adjustable': True,
        'note': 'Below this — use tighter prior sigma'
    },
    'zero_events_pct_warning': {
        'value': 30,
        'source': 'Empirical — >30% zeros may indicate sparse data requiring Bayesian approach',
        'adjustable': False,
        'note': 'Threshold for warning about excess zeros'
    },
    'extreme_outlier_z': {
        'value': 3.0,
        'source': 'Standard 3-sigma rule for outlier detection',
        'adjustable': False,
        'note': 'Z-score threshold for extreme outlier flagging'
    },

    # --- site_present ---
    'default_target_crs': {
        'value': 'EPSG:3857',
        'source': 'Web Mercator — standard for web maps and spatial operations',
        'adjustable': True,
        'note': 'Can use EPSG:4326 (WGS84) or local Ukrainian CRS'
    },

    # --- Data Quality Precheck Thresholds ---
    'dq_low_test_pct_warning': {
        'value': 30,
        'source': 'Empirical — >30% territories with few tests suggests sparse surveillance',
        'adjustable': True,
        'note': 'Percentage of territories with < min_tests to trigger warning'
    },
    'dq_low_test_pct_info': {
        'value': 15,
        'source': 'Empirical',
        'adjustable': True,
        'note': 'Percentage threshold for info-level message'
    },
    'dq_zero_events_pct_warning': {
        'value': 50,
        'source': 'Empirical — >50% zeros strongly suggests Bayesian approach needed',
        'adjustable': False,
        'note': 'Percentage of territories with zero events for warning'
    },
    'dq_outlier_pct_warning': {
        'value': 10,
        'source': 'Empirical — >10% outliers may indicate data quality issues',
        'adjustable': False,
        'note': 'Percentage of territories flagged as outliers'
    },
    'dq_total_tests_min': {
        'value': 100,
        'source': 'Empirical — fewer than 100 total tests makes national estimate unreliable',
        'adjustable': True,
        'note': 'Minimum total tests for reliable national baseline'
    },
    'dq_total_recent_min': {
        'value': 5,
        'source': 'Empirical — fewer than 5 recent events gives unstable rate',
        'adjustable': True,
        'note': 'Minimum recent events for stable rate estimation'
    },

    # --- Reliability Score Thresholds ---
    'rel_data_adequacy_test_tiers': {
        'value': [(100, 100), (50, 80), (20, 60), (10, 40), (0, 10)],
        'source': 'CDC HIV Surveillance Guidelines (2019) — data quality tiers',
        'adjustable': True,
        'note': 'List of (min_tests, score) tuples — territory gets score of first matching tier'
    },
    'rel_sample_size_tiers': {
        'value': [(100, 100), (50, 80), (20, 60), (0, 30)],
        'source': 'Gelman & Hill (2007) — sample size adequacy tiers',
        'adjustable': True,
        'note': 'List of (min_tests, score) tuples for sample size component'
    },
    'rel_zeros_tiers': {
        'value': [(30, 100), (50, 70), (70, 40), (100, 10)],
        'source': 'Empirical — proportion of zero-event territories',
        'adjustable': True,
        'note': 'List of (max_zeros_pct, score) tuples'
    },
    'rel_curr_hist_ratio_tiers': {
        'value': [(0.5, 0), (0.3, -10), (0, -20)],
        'source': 'Empirical — if current data < 30% of historical, flag as sparse',
        'adjustable': True,
        'note': 'List of (max_ratio, penalty) tuples for temporal density check'
    },
    'rel_imputed_pct_penalty': {
        'value': [(25, 0), (50, -5), (75, -10), (100, -15)],
        'source': 'Empirical — penalty for imputed covariate values',
        'adjustable': True,
        'note': 'List of (max_imputed_pct, penalty) tuples'
    },

    # --- Sampling Configuration Thresholds ---
    'sampling_territory_threshold_small': {
        'value': 20,
        'source': 'Empirical — < 20 territories needs fewer chains to avoid overfitting',
        'adjustable': True,
        'note': 'Below this — use 2 chains, 500 draws'
    },
    'sampling_territory_threshold_large': {
        'value': 50,
        'source': 'Empirical — > 50 territories can use more cores',
        'adjustable': True,
        'note': 'Above this — use 4 cores'
    },
    'hurdle_threshold': {
        'value': 70.0,
        'source': 'Empirical — if >70% territories have zero events, hurdle model recommended',
        'adjustable': True,
        'note': 'Percentage of zero-event territories to trigger hurdle model suggestion'
    },

    # --- Prior Specification (logit-scale) ---
    'prior_mu_logit_clip_min': {
        'value': 0.001,
        'source': 'Numerical safety — prevents logit(-inf)',
        'adjustable': False,
        'note': 'Minimum value for clipping national_rate before logit transform'
    },
    'prior_mu_logit_clip_max': {
        'value': 0.99,
        'source': 'Numerical safety — prevents logit(+inf)',
        'adjustable': False,
        'note': 'Maximum value for clipping national_rate before logit transform'
    },
}



# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "excel_path": "data/input_data.xlsx",
    "output_dir": "output",
    "target_crs": "EPSG:3857",
    "analysis_period": {
        "start": "2026-01-01",
        "end": "2026-03-31"
    },
    "residual_thresholds": {
        "obvious": 2.0,
        "slight": 1.0
    },
    "color_map": {
        # Legacy single-axis labels (kept for any output produced before the
        # SIR/SMR taxonomy was introduced; the map currently reads the new
        # column when present and falls back to these otherwise).
        "Obvious Increase": "#ef2bc1",
        "Slight Increase":  "#ffab4d",
        "No Difference":    "#d3d3d3",
        "Slight Decrease":  "#38d430",
        "Obvious Decrease": "#00b9e5",
        # Two-dimensional SIR/SMR taxonomy.
        "Established hotspot":        "#d62728",
        "Emerging hotspot":           "#fb8500",
        "Stable high-burden":         "#ffd60a",
        "Declining from high-burden": "#48cae4",
        "Emerging decrease":          "#a0e0a8",
        "Significant decrease":       "#06d6a0",
        "Normal":                     "#d3d3d3",
        "No Data":                    "#ffffff"
    },
    # Ukrainian translations for map labels (both legacy and SIR/SMR taxonomy).
    "map_strings_ua": {
        # Legacy single-axis labels.
        "Obvious Increase": "Значне зростання",
        "Slight Increase":  "Незначне зростання",
        "No Difference":    "Без змін",
        "Slight Decrease":  "Незначне зниження",
        "Obvious Decrease": "Значне зниження",
        # Two-dimensional SIR/SMR taxonomy.
        "Established hotspot":        "Підтверджена гаряча точка",
        "Emerging hotspot":           "Ранній сигнал",
        "Stable high-burden":         "Стабільно високий рівень",
        "Declining from high-burden": "Спад із високого рівня",
        "Emerging decrease":          "Ранній спад",
        "Significant decrease":       "Значне зниження (рівень)",
        "Normal":                     "Без сигналу",
        "No Data": "Немає даних",
        "Low Reliability (weak data)": "Низька надійність (мало даних)",
        "Classification": "Класифікація",
        "High - Reliable for decisions": "Висока — надійно для рішень",
        "Moderate - Use with caution": "Середня — використовуйте обережно",
        "Low - High uncertainty": "Низька — висока невизначеність",
        "Reliability Rating": "Рівень надійності",
        "Reliability": "Надійність",
        "Model": "Модель",
        "Analysis period": "Період аналізу",
        "Baseline period": "Період базової лінії",
        "National baseline": "Національна базова лінія",
        "Territories by reliability": "Території за рівнем надійності",
        "Based on": "На основі"
    },
    # Oblast name translations: EN -> UA
    "oblast_names_ua": {
        "Vinnytska": "Вінницька",
        "Volynska": "Волинська",
        "Dnipropetrovska": "Дніпропетровська",
        "Dnipro": "Дніпро",
        "Donetska": "Донецька",
        "Zhytomyrska": "Житомирська",
        "Zakarpatska": "Закарпатська",
        "Zakarpattia": "Закарпаття",
        "Zaporizka": "Запорізька",
        "Zaporizhzhia": "Запоріжжя",
        "Ivano-Frankivska": "Івано-Франківська",
        "Iv.-Frank.": "Ів.-Фр.",
        "Kyiv": "Київ",
        "Kyivska": "Київська",
        "Kirovohradska": "Кіровоградська",
        "Kirovohr.": "Кіровогр.",
        "Kropyvnytska": "Кропивницька",
        "Kropyvn.": "Кропивн.",
        "Luhanska": "Луганська",
        "Lvivska": "Львівська",
        "Mykolaivska": "Миколаївська",
        "Odesa": "Одеса",
        "Odeska": "Одеська",
        "Poltavska": "Полтавська",
        "Rivnenska": "Рівненська",
        "Sumska": "Сумська",
        "Ternopilska": "Тернопільська",
        "Kharkivska": "Харківська",
        "Khersonska": "Херсонська",
        "Khmelnytska": "Хмельницька",
        "Khmeln.": "Хмельн.",
        "Cherkaska": "Черкаська",
        "Chernivetska": "Чернівецька",
        "Chernihivska": "Чернігівська"
    },
    "analysis_mode": "h3_hexagons",
    "admin_levels": [],
    "hex_resolutions": [4],
    "administrative_units": {
        "adm3_path": "data/Ukraine_Adm3_OTG.geojson",
        "adm2_path": "data/Ukraine_Adm2_Rayon.geojson",
        "adm1_path": "data/Ukraine_Adm1_Oblast.geojson",
        "otg_col": "ADM3_EN",
        "rayon_col": "ADM2_EN",
        "oblast_col": "ADM1_EN"
    },
    # Epidemiological cut-offs for the SMR/SIR exceedance taxonomy. SMR is
    # "current rate vs national" and SIR is "current vs trend-adjusted own
    # history"; a territory is flagged on an axis when P(ratio > threshold)
    # clears the FDR cut-off. RR >= 2 ("doubling") and 1.5 are the conventional
    # elevated / moderately-elevated levels; expose them so they can be tuned.
    "detection": {
        "smr_threshold": 2.0,
        "sir_threshold": 1.5
    },
    # Combined burden + rate watch-list (pipeline.classification.add_watchlist).
    # Triage knobs, NOT significance thresholds: the rigorous FDR hotspot call
    # in `classification` is unaffected.
    "watchlist": {
        "burden_top_frac": 0.80,   # flag territories carrying the top 80% of recent cases
        "rate_percentile": 0.80    # SMR >= 80th percentile => relatively elevated (top 20%)
    },
    "h3_hexagons": {
        "res3_path": "data/h3_hexagons_res3.geojson",
        "res4_path": "data/h3_hexagons_res4.geojson",
        # Finer hexagons (resolution 5) -- populate when the layer is added
        # for the hepatitis / STI / TB rollouts.
        "res5_path": None,
        "h3_id_col": "h3_id"
    }
}


# Default share of structural zeros above which the wizard recommends the
# truncated-Binomial (a.k.a. "hurdle") branch. Defined once so the same
# threshold appears in the rule, in the audit-trail message and in any
# downstream report.
DEFAULT_TRUNCATED_BINOMIAL_STRUCTURAL_ZEROS_PCT = 70.0


# =============================================================================
# STATISTICAL THRESHOLDS - data-specification analyzer (AutoSpecificationSystem)
# =============================================================================
# Zero-inflation thresholds (Lambert 1992; Hilbe 2011): > 50% zeros is severe,
# > 30% moderate.
ZERO_INFLATION_THRESHOLD_HIGH = 50.0
ZERO_INFLATION_THRESHOLD_MODERATE = 30.0

# Overdispersion thresholds (Cameron & Trivedi 2013).
OVERDISPERSION_THRESHOLD_HIGH = 2.0
OVERDISPERSION_THRESHOLD_MODERATE = 1.5

# Composite Bayesian-recommendation score (3+ = high confidence, 2+ = medium).
BAYESIAN_SCORE_HIGH_CONFIDENCE = 3
BAYESIAN_SCORE_MEDIUM_CONFIDENCE = 2

# Sample-size thresholds (CLT / small-sample literature).
SMALL_SAMPLE_SIZE = 30
VERY_SMALL_SAMPLE_SIZE = 10

# Outlier detection (Tukey's fences for extreme outliers).
OUTLIER_THRESHOLD_PCT = 10.0
IQR_MULTIPLIER_EXTREME = 3.0

# Small-sample proportion thresholds (territories with n < SMALL_SAMPLE_SIZE).
SMALL_SAMPLE_PCT_HIGH = 50.0
SMALL_SAMPLE_PCT_MODERATE = 30.0
