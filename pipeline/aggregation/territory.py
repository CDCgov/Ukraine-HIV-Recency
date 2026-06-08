"""
Territory-level aggregation and outbreak/artifact detection.

Three routines live here:

* :func:`aggregate_covariates` -- per-territory covariate composition
  (risk_high_pct, age_15_24_pct, age_25_44_pct). Falls back to proxy
  values with a loud warning if ``risk_group`` is absent.
* :func:`aggregate_stats_stratified` -- SOFT stratification: one row per
  territory with ``proportion_high_risk`` as a covariate. Imputes the
  national average for territories without current-period data via
  empirical Bayes. Computes testing-intensity (testo-months) and
  network-stability z-scores vectorised.
* :func:`detect_outbreak_and_artifact` -- per-territory diagnosis of
  whether an apparent rise is real, group-specific or a testing-
  composition artefact. Uses the HARD stratification frame; falls back
  to :func:`pipeline.aggregation.soft_fallback_result` when the strata
  do not carry enough tests.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from pipeline.aggregation.outbreak_defaults import soft_fallback_result
from pipeline.aggregation.testing_network import (
    calculate_testing_intensity,
    classify_network_stability,
)

logger = logging.getLogger(__name__)


def aggregate_covariates(gdf_admin: gpd.GeoDataFrame, gdf_cases: gpd.GeoDataFrame,
                        start: pd.Timestamp, end: pd.Timestamp) -> gpd.GeoDataFrame:
    """Aggregate covariates (risk_group, age_group) for each territory."""
    gdf_admin = gdf_admin.copy()

    # Initialize covariate columns
    gdf_admin['risk_high_pct'] = 0.0
    gdf_admin['age_15_24_pct'] = 0.0
    gdf_admin['age_25_44_pct'] = 0.0

    # Check if covariates exist in data
    required_cols = ['risk_group']
    if not all(col in gdf_cases.columns for col in required_cols):
        logger.warning("="*80)
        logger.warning("[WARN]  MISSING COVARIATE COLUMNS - USING PROXY DATA")
        logger.warning("="*80)
        logger.warning(f"Required columns: {required_cols}")
        logger.warning("Current approach: Proxy stratification (high-risk = top 50% by historical rate)")
        logger.warning("")
        logger.warning("[WARN]  IMPORTANT: Results for stratified analysis are based on PROXY data,")
        logger.warning("   not real risk group classifications. Conclusions about high-risk vs")
        logger.warning("   low-risk groups should be interpreted with caution.")
        logger.warning("")
        logger.warning("RECOMMENDATION: Add 'risk_group' and 'age_group' columns to input data")
        logger.warning("   for accurate stratified analysis.")
        logger.warning("="*80)
        return gdf_admin

    # Current period cases
    curr_all = gdf_cases[(gdf_cases['test_date'] >= start) & (gdf_cases['test_date'] <= end)]

    # Spatial join - ONE TIME
    joined = gpd.sjoin(curr_all, gdf_admin[['geometry']], how='left', predicate='within')

    # Aggregate covariates using groupby (FAST - no loop!)
    if len(joined) > 0:
        # Risk group percentages
        risk_counts = joined.groupby(['index_right', 'risk_group']).size().unstack(fill_value=0)
        if 'high' in risk_counts.columns:
            total_counts = joined.groupby('index_right').size()
            risk_high_pct = (risk_counts['high'] / total_counts * 100)
            gdf_admin['risk_high_pct'] = gdf_admin.index.map(risk_high_pct).fillna(0.0)

    # Age groups - set to 0 if not available
    gdf_admin['age_15_24_pct'] = 0.0
    gdf_admin['age_25_44_pct'] = 0.0

    return gdf_admin



def aggregate_stats_stratified(gdf_admin: gpd.GeoDataFrame, gdf_cases: gpd.GeoDataFrame,
                               start: pd.Timestamp, end: pd.Timestamp,
                               b_start: pd.Timestamp, b_end: pd.Timestamp) -> pd.DataFrame:
    """
    Aggregate stats with SOFT STRATIFICATION (covariate approach).

    Instead of creating 2 rows per territory (high/low), creates 1 row per territory
    with proportion_high_risk as a covariate. This preserves ALL territories, even those
    with incomplete risk group data.

    Model: y ~ alpha + beta_hist * hist + beta_risk * proportion_high_risk

    Args:
        gdf_admin: GeoDataFrame with administrative boundaries
        gdf_cases: GeoDataFrame with HIV testing cases
        start: Start date of current analysis period
        end: End date of current analysis period
        b_start: Start date of baseline period
        b_end: End date of baseline period

    Returns:
        DataFrame with one row per territory, including proportion_high_risk covariate
    """

    # Check if risk_group exists
    if 'risk_group' not in gdf_cases.columns:
        logger.warning("risk_group column not found - using proportion_high_risk=0.5 for all territories")
        has_risk_group = False
    else:
        has_risk_group = True

    # Current period
    curr_all = gdf_cases[(gdf_cases['test_date'] >= start) & (gdf_cases['test_date'] <= end)]
    curr_recent = curr_all[curr_all['type'] == 'recent']

    # Historical period
    hist_all = gdf_cases[(gdf_cases['test_date'] >= b_start) & (gdf_cases['test_date'] <= b_end)]
    hist_recent = hist_all[hist_all['type'] == 'recent']

    # Spatial joins
    curr_all_joined = gpd.sjoin(curr_all, gdf_admin[['geometry']], how='left', predicate='within')
    curr_recent_joined = gpd.sjoin(curr_recent, gdf_admin[['geometry']], how='left', predicate='within')
    hist_all_joined = gpd.sjoin(hist_all, gdf_admin[['geometry']], how='left', predicate='within')
    hist_recent_joined = gpd.sjoin(hist_recent, gdf_admin[['geometry']], how='left', predicate='within')

    # Get territory name column
    name_col = None
    for col in ['ADM3_EN', 'ADM2_EN', 'ADM1_EN', 'name']:
        if col in gdf_admin.columns:
            name_col = col
            break

    # Aggregate TOTAL counts per territory (all risk groups combined)
    curr_all_total = curr_all_joined.groupby('index_right').size()
    curr_recent_total = curr_recent_joined.groupby('index_right').size()
    hist_all_total = hist_all_joined.groupby('index_right').size()
    hist_recent_total = hist_recent_joined.groupby('index_right').size()

    # Reindex to include all territories
    all_indices = gdf_admin.index
    curr_all_total = curr_all_total.reindex(all_indices, fill_value=0)
    curr_recent_total = curr_recent_total.reindex(all_indices, fill_value=0)
    hist_all_total = hist_all_total.reindex(all_indices, fill_value=0)
    hist_recent_total = hist_recent_total.reindex(all_indices, fill_value=0)

    # Calculate proportion_high_risk per territory
    if has_risk_group:
        # Count high-risk tests per territory
        curr_high_counts = curr_all_joined[curr_all_joined['risk_group'] == 'high'].groupby('index_right').size()
        curr_high_counts = curr_high_counts.reindex(all_indices, fill_value=0)

        # Calculate national average proportion of high-risk tests (Empirical Bayes prior)
        # Only from territories with data to avoid circular logic
        territories_with_data = curr_all_total > 0
        if territories_with_data.sum() > 0:
            national_avg_high_risk = curr_high_counts[territories_with_data].sum() / curr_all_total[territories_with_data].sum()
        else:
            # Fallback if no territories have data (unlikely)
            national_avg_high_risk = 0.5

        # Proportion of tests from high-risk group
        # Use national average for territories without data (Empirical Bayes imputation)
        proportion_high_risk = np.where(
            curr_all_total > 0,
            curr_high_counts / curr_all_total,
            national_avg_high_risk  # Empirical Bayes: use national average instead of arbitrary 0.5
        )

        # Track which territories have imputed proportion_high_risk
        imputed_proportion = (curr_all_total == 0)

        # Log imputation statistics
        n_imputed = imputed_proportion.sum()
        if n_imputed > 0:
            logger.info(f"Imputed proportion_high_risk for {n_imputed} territories without data")
            logger.info(f"  National average used: {national_avg_high_risk:.3f}")
            logger.info(f"  (Previous approach used 0.5, which overestimated risk)")
        else:
            logger.debug(f"All territories have data, no imputation needed")
    else:
        # No risk group data - cannot calculate proportion_high_risk
        # Use 0.5 as truly neutral assumption (no information available)
        logger.warning("[WARN] No risk_group column found in data")
        logger.warning("   Using proportion_high_risk = 0.5 for all territories (no empirical data)")
        proportion_high_risk = np.full(len(all_indices), 0.5)
        # All territories have imputed values when no risk_group column
        imputed_proportion = np.full(len(all_indices), True)

    # Get territory names
    territory_names = gdf_admin[name_col] if name_col else gdf_admin.index.to_series().apply(lambda x: f"Territory_{x}")

    # Create DataFrame with ONE row per territory
    df_stratified = pd.DataFrame({
        'territory_idx': all_indices,
        'territory_name': territory_names.values,
        'all_tested_curr': curr_all_total.values,
        'recent_count_curr': curr_recent_total.values,
        'all_tested_hist': hist_all_total.values,
        'recent_count_hist': hist_recent_total.values,
        'proportion_high_risk': proportion_high_risk,
        'imputed_proportion_high_risk': imputed_proportion
    })

    # Calculate proportions
    df_stratified['recent_proportion_curr'] = np.where(
        df_stratified['all_tested_curr'] > 0,
        df_stratified['recent_count_curr'] / df_stratified['all_tested_curr'],
        0
    )
    df_stratified['recent_proportion_hist'] = np.where(
        df_stratified['all_tested_hist'] > 0,
        df_stratified['recent_count_hist'] / df_stratified['all_tested_hist'],
        np.nan  # Use NaN instead of 0 for territories without historical data
    )

    # Filter: keep only territories with current period data
    df_stratified = df_stratified[df_stratified['all_tested_curr'] > 0]

    logger.info(f"Soft stratification: {len(df_stratified)} territories with proportion_high_risk covariate")
    logger.info(f"  Mean proportion_high_risk: {df_stratified['proportion_high_risk'].mean():.2f}")
    logger.info(f"  Range: [{df_stratified['proportion_high_risk'].min():.2f}, {df_stratified['proportion_high_risk'].max():.2f}]")

    # Calculate testing intensity for each territory
    logger.info("Calculating testing intensity (testo-months)...")

    df_stratified['testing_intensity_curr'] = 0.0
    df_stratified['testing_intensity_hist'] = 0.0
    df_stratified['n_active_months_curr'] = 0
    df_stratified['n_active_months_hist'] = 0

    # Vectorized testing intensity — one spatial join + groupby instead of N loops
    gdf_admin_with_idx = gdf_admin.reset_index().rename(columns={'index': 'territory_idx_join'})

    for period_label, all_cases, period_start, period_end, col_intensity, col_months in [
        ('current', curr_all, start, end, 'testing_intensity_curr', 'n_active_months_curr'),
        ('baseline', hist_all, b_start, b_end, 'testing_intensity_hist', 'n_active_months_hist')
    ]:
        if len(all_cases) == 0:
            continue
        joined = gpd.sjoin(all_cases, gdf_admin_with_idx[['territory_idx_join', 'geometry']],
                           how='inner', predicate='within')
        joined['year_month'] = joined['test_date'].dt.to_period('M')
        monthly = joined.groupby(['territory_idx_join', 'year_month']).size().reset_index(name='n_tests')
        agg = monthly.groupby('territory_idx_join').agg(
            n_active=('year_month', 'nunique'),
            total=('n_tests', 'sum')
        )
        agg['intensity'] = agg['total'] / agg['n_active'].clip(lower=1)
        # Merge back to df_stratified
        merged = agg[['intensity', 'n_active']].rename(columns={
            'intensity': col_intensity, 'n_active': col_months
        })
        if len(merged) > 0:
            df_stratified = df_stratified.merge(
                merged, left_on='territory_idx', right_index=True, how='left'
            )
        # Ensure columns exist (may be missing if no data matched)
        if col_intensity not in df_stratified.columns:
            df_stratified[col_intensity] = 0.0
        if col_months not in df_stratified.columns:
            df_stratified[col_months] = 0
        df_stratified[col_intensity] = df_stratified[col_intensity].fillna(0)
        df_stratified[col_months] = df_stratified[col_months].fillna(0).astype(int)

    logger.info(f"[OK] Testing intensity calculated (vectorized)")
    logger.info(f"  Mean intensity (current): {df_stratified['testing_intensity_curr'].mean():.1f} test-months")
    logger.info(f"  Mean intensity (baseline): {df_stratified['testing_intensity_hist'].mean():.1f} test-months")

    # Vectorized network stability classification
    logger.info("Classifying network stability...")

    curr_vals = df_stratified['testing_intensity_curr'].values
    hist_vals = df_stratified['testing_intensity_hist'].values

    # Relative change per territory (suppress divide-by-zero for territories with 0 history)
    with np.errstate(divide='ignore', invalid='ignore'):
        safe_hist = np.where(hist_vals > 0, hist_vals, np.nan)
        rel_change = (curr_vals - hist_vals) / safe_hist

    # Z-score: compare each territory's change to population distribution
    valid_mask = ~np.isnan(rel_change)
    if valid_mask.sum() >= 3:
        mean_change = np.nanmean(rel_change)
        std_change = np.nanstd(rel_change)
        if std_change > 0:
            z_scores = (rel_change - mean_change) / std_change
        else:
            z_scores = np.zeros(len(rel_change))
    else:
        z_scores = np.full(len(rel_change), np.nan)

    # Classify based on z-score or simple thresholds
    df_stratified['intensity_change_pct'] = np.where(valid_mask, rel_change * 100, 0)
    df_stratified['network_stability_z'] = np.where(np.isnan(z_scores), 0.0, z_scores)

    # Classification logic (vectorized)
    has_baseline = hist_vals > 0
    has_enough_data = valid_mask.sum() >= 3

    if has_enough_data:
        # Z-score based classification
        df_stratified['network_stability'] = np.select(
            [~has_baseline,
             np.abs(z_scores) < 1.0,
             np.abs(z_scores) < 2.0,
             np.abs(z_scores) >= 2.0],
            ['NO_BASELINE', 'STABLE', 'MODERATE_CHANGE', 'MAJOR_CHANGE'],
            default='STABLE'
        )
    else:
        # Simple threshold classification
        df_stratified['network_stability'] = np.select(
            [~has_baseline,
             np.abs(rel_change) < 0.25,
             np.abs(rel_change) < 0.50,
             np.abs(rel_change) >= 0.50],
            ['NO_BASELINE', 'STABLE', 'MODERATE_CHANGE', 'MAJOR_CHANGE'],
            default='STABLE'
        )

    # Log stability distribution
    stability_counts = df_stratified['network_stability'].value_counts()
    logger.info(f"[OK] Network stability classified:")
    for category, count in stability_counts.items():
        pct = count / len(df_stratified) * 100
        logger.info(f"  {category}: {count} ({pct:.1f}%)")

    return df_stratified



def detect_outbreak_and_artifact(territory_idx: int, df_hard: pd.DataFrame,
                                 national_rate: float) -> Dict[str, Any]:
    """
    Detect outbreak in specific risk groups and testing artifact for one territory.

    Uses HARD stratification data (2 rows: high + low) to determine:
    1. Is there an outbreak in high-risk group?
    2. Is there an outbreak in low-risk group?
    3. Is the overall increase due to testing artifact (change in testing composition)?
    4. What % of increase is explained by testing artifact vs real outbreak?

    Args:
        territory_idx: Territory index
        df_hard: DataFrame from aggregate_stats_hard_stratified (2 rows per territory)
        national_rate: National baseline rate

    Returns:
        Dict with:
        - high_outbreak: bool
        - low_outbreak: bool
        - testing_artifact: bool
        - artifact_contribution: float (0-100, % of increase due to testing change)
        - outbreak_type: str (e.g., "REAL OUTBREAK IN HIGH-RISK GROUP")
        - explanation: str (detailed explanation in Ukrainian)
        - stratification_method: "HARD" or "SOFT"
    """
    # Get data for this territory (2 rows: high + low)
    terr_data = df_hard[df_hard['territory_idx'] == territory_idx]

    if len(terr_data) != 2:
        logger.error(f"Expected 2 rows for territory {territory_idx}, got {len(terr_data)}")
        return soft_fallback_result()

    # Extract high and low data
    high_data = terr_data[terr_data['risk_group'] == 'high'].iloc[0]
    low_data = terr_data[terr_data['risk_group'] == 'low'].iloc[0]

    # Check if can use HARD
    can_use_hard = high_data['can_use_hard']

    if not can_use_hard:
        return soft_fallback_result()

    # Extract values
    high_curr_prop = high_data['recent_proportion_curr']
    high_hist_prop = high_data['recent_proportion_hist']
    high_n_curr = high_data['all_tested_curr']
    high_n_hist = high_data['all_tested_hist']

    low_curr_prop = low_data['recent_proportion_curr']
    low_hist_prop = low_data['recent_proportion_hist']
    low_n_curr = low_data['all_tested_curr']
    low_n_hist = low_data['all_tested_hist']

    # Calculate testing composition change
    total_curr = high_n_curr + low_n_curr
    total_hist = high_n_hist + low_n_hist

    prop_high_curr = high_n_curr / total_curr if total_curr > 0 else 0.5
    prop_high_hist = high_n_hist / total_hist if total_hist > 0 else 0.5

    testing_shift = prop_high_curr - prop_high_hist

    # Detect outbreak in each group using binomial test
    # High-risk group: test if current > historical
    if high_n_curr >= 3 and high_n_hist >= 3:
        high_pvalue = stats.binomtest(
            int(high_curr_prop * high_n_curr),
            high_n_curr,
            high_hist_prop,
            alternative='greater'
        ).pvalue
        high_outbreak = (high_pvalue < 0.05)
    else:
        high_outbreak = False
        high_pvalue = 1.0

    # Low-risk group: test if current > historical
    if low_n_curr >= 3 and low_n_hist >= 3:
        low_pvalue = stats.binomtest(
            int(low_curr_prop * low_n_curr),
            low_n_curr,
            low_hist_prop,
            alternative='greater'
        ).pvalue
        low_outbreak = (low_pvalue < 0.05)
    else:
        low_outbreak = False
        low_pvalue = 1.0

    # Detect testing artifact
    # Use a statistical test instead of an ad-hoc 10% threshold
    # Test if proportion of high-risk tests changed significantly between periods
    # Use two-proportion z-test (chi-square equivalent)

    # Aliases for clarity (total tests per group per period)
    high_total_curr = int(high_n_curr)
    low_total_curr  = int(low_n_curr)
    high_total_hist = int(high_n_hist)
    low_total_hist  = int(low_n_hist)

    if high_total_curr > 0 and high_total_hist > 0:
        # Two-proportion z-test
        # H0: prop_high_curr == prop_high_hist
        # H1: prop_high_curr != prop_high_hist
        contingency_table = np.array([
            [high_total_curr, low_total_curr],
            [high_total_hist, low_total_hist]
        ])
        chi2, testing_pvalue, _, _ = stats.chi2_contingency(contingency_table)

        # IMPROVED: Don't use simple boolean - testing artifact can coexist with outbreak
        # We'll calculate artifact_contribution below and use that for classification
        testing_composition_changed = (testing_pvalue < 0.05 and abs(testing_shift) > 0.05)

        logger.debug(f"Testing composition check: shift={testing_shift:.3f}, p={testing_pvalue:.3f}, "
                    f"changed={testing_composition_changed}")
    else:
        # Cannot test if no data in one period
        testing_composition_changed = False
        testing_pvalue = 1.0

    # Calculate artifact contribution
    # Decompose overall change into: real outbreak + testing composition change
    overall_curr = (high_curr_prop * prop_high_curr + low_curr_prop * (1 - prop_high_curr))
    overall_hist = (high_hist_prop * prop_high_hist + low_hist_prop * (1 - prop_high_hist))
    overall_change = overall_curr - overall_hist

    if overall_change > 0.001:  # Avoid division by zero
        # Change due to testing shift (keeping rates constant)
        change_from_testing = (high_hist_prop * testing_shift - low_hist_prop * testing_shift)

        # Change due to real outbreak (keeping composition constant)
        change_from_outbreak = ((high_curr_prop - high_hist_prop) * prop_high_hist +
                               (low_curr_prop - low_hist_prop) * (1 - prop_high_hist))

        artifact_contribution = (change_from_testing / overall_change * 100) if overall_change > 0 else 0.0
        artifact_contribution = max(0, min(100, artifact_contribution))  # Clamp to [0, 100]
    else:
        artifact_contribution = 0.0

    # IMPROVED: Classify based on artifact_contribution instead of simple boolean
    # This allows for mixed cases (both outbreak AND testing artifact)
    if artifact_contribution > 50 and testing_composition_changed:
        # Predominantly testing artifact
        testing_artifact = True
        artifact_severity = "PREDOMINANTLY_ARTIFACT"
    elif artifact_contribution > 20 and testing_composition_changed:
        # Mixed: both outbreak and testing artifact
        testing_artifact = True  # Flag as artifact present
        artifact_severity = "MIXED"
    elif testing_composition_changed:
        # Testing composition changed but contributes little to overall change
        testing_artifact = False
        artifact_severity = "MINOR_ARTIFACT"
    else:
        # No significant testing composition change
        testing_artifact = False
        artifact_severity = "NO_ARTIFACT"

    # Determine outbreak type and explanation
    territory_name = high_data['territory_name']

    if high_outbreak and low_outbreak:
        outbreak_type = "OUTBREAK IN BOTH GROUPS"
        explanation = (
            f"Outbreak detected in both risk groups. "
            f"High-risk: {high_curr_prop*100:.1f}% (was {high_hist_prop*100:.1f}%, p={high_pvalue:.3f}). "
            f"Low-risk: {low_curr_prop*100:.1f}% (was {low_hist_prop*100:.1f}%, p={low_pvalue:.3f}). "
            f"This is a critical situation requiring immediate intervention in both groups."
        )
    elif high_outbreak and not low_outbreak:
        outbreak_type = "REAL OUTBREAK IN HIGH-RISK GROUP"
        explanation = (
            f"Real outbreak detected only in high-risk group. "
            f"High-risk: {high_curr_prop*100:.1f}% (was {high_hist_prop*100:.1f}%, p={high_pvalue:.3f}). "
            f"Low-risk group stable: {low_curr_prop*100:.1f}% (was {low_hist_prop*100:.1f}%). "
        )
        if artifact_contribution > 10:
            explanation += (
                f"Additionally, change in testing composition (high-risk proportion increased from {prop_high_hist*100:.0f}% to {prop_high_curr*100:.0f}%) "
                f"explains {artifact_contribution:.0f}% of overall increase. "
                f"Remaining {100-artifact_contribution:.0f}% is real epidemic worsening in high-risk group."
            )
    elif low_outbreak and not high_outbreak:
        outbreak_type = "REAL OUTBREAK IN LOW-RISK GROUP"
        explanation = (
            f"Real outbreak detected only in low-risk group. "
            f"Low-risk: {low_curr_prop*100:.1f}% (was {low_hist_prop*100:.1f}%, p={low_pvalue:.3f}). "
            f"High-risk group stable: {high_curr_prop*100:.1f}% (was {high_hist_prop*100:.1f}%). "
            f"This is an unusual situation requiring detailed epidemiological investigation, "
            f"as outbreaks typically start in high-risk groups."
        )
    elif testing_artifact:
        # Use artifact_severity for more nuanced explanation
        if artifact_severity == "PREDOMINANTLY_ARTIFACT":
            outbreak_type = "TESTING ARTIFACT (PREDOMINANT)"
            explanation = (
                f"Increase predominantly explained by change in testing composition ({artifact_contribution:.0f}%), not real outbreak. "
                f"High-risk proportion changed from {prop_high_hist*100:.0f}% to {prop_high_curr*100:.0f}% (change: {testing_shift*100:+.0f}%). "
                f"Infection levels in both groups remained relatively stable: "
                f"high-risk {high_curr_prop*100:.1f}% (was {high_hist_prop*100:.1f}%), "
                f"low-risk {low_curr_prop*100:.1f}% (was {low_hist_prop*100:.1f}%). "
                f"Recommendation: verify if target testing population changed in this territory."
            )
        elif artifact_severity == "MIXED":
            outbreak_type = "MIXED: OUTBREAK + TESTING ARTIFACT"
            explanation = (
                f"Increase due to BOTH real outbreak AND testing composition change. "
                f"Testing artifact explains {artifact_contribution:.0f}% of increase, "
                f"real epidemic change explains {100-artifact_contribution:.0f}%. "
                f"High-risk proportion changed from {prop_high_hist*100:.0f}% to {prop_high_curr*100:.0f}%. "
                f"Infection rates: high-risk {high_curr_prop*100:.1f}% (was {high_hist_prop*100:.1f}%), "
                f"low-risk {low_curr_prop*100:.1f}% (was {low_hist_prop*100:.1f}%). "
                f"Recommendation: address both testing strategy AND epidemic situation."
            )
        else:  # MINOR_ARTIFACT
            outbreak_type = "OUTBREAK WITH MINOR TESTING ARTIFACT"
            explanation = (
                f"Real outbreak detected. Testing composition changed but contributes minimally ({artifact_contribution:.0f}%). "
                f"High-risk: {high_curr_prop*100:.1f}% (was {high_hist_prop*100:.1f}%). "
                f"Low-risk: {low_curr_prop*100:.1f}% (was {low_hist_prop*100:.1f}%). "
                f"Focus on epidemic response."
            )
    else:
        outbreak_type = "STABLE SITUATION"
        explanation = (
            f"Situation stable in both risk groups. "
            f"High-risk: {high_curr_prop*100:.1f}% (was {high_hist_prop*100:.1f}%). "
            f"Low-risk: {low_curr_prop*100:.1f}% (was {low_hist_prop*100:.1f}%). "
            f"Indicators within expected statistical variation."
        )

    # Compute 95% Wilson credibility intervals for EACH group
    # based on HISTORICAL data — this is the "expected range"
    def _wilson_ci(k, n):
        """95% Wilson CI for proportion k/n. Returns (lower, upper) or (0,0) if n==0."""
        if n < 1:
            return 0.0, 0.0
        # Clopper-Pearson (exact) interval: conservative and well-behaved
        # for the very small test counts common in sparse hexagons, where
        # the Wilson interval can be anti-conservative.
        lo, hi = proportion_confint(int(k), int(n), alpha=0.05, method='beta')
        return float(lo), float(hi)

    high_hist_k = high_hist_prop * high_n_hist
    low_hist_k  = low_hist_prop  * low_n_hist
    high_ci_lo, high_ci_hi = _wilson_ci(high_hist_k, high_n_hist)
    low_ci_lo,  low_ci_hi  = _wilson_ci(low_hist_k,  low_n_hist)

    return {
        'high_outbreak': high_outbreak,
        'low_outbreak': low_outbreak,
        'testing_artifact': testing_artifact,
        'artifact_contribution': artifact_contribution,
        'artifact_severity': artifact_severity,
        'outbreak_type': outbreak_type,
        'explanation': explanation,
        'high_pvalue': high_pvalue,
        'low_pvalue': low_pvalue,
        'testing_shift': testing_shift,
        'stratification_method': 'HARD',
        'high_observed_curr': high_curr_prop,
        'low_observed_curr': low_curr_prop,
        'high_ci_lower': high_ci_lo,
        'high_ci_upper': high_ci_hi,
        'low_ci_lower': low_ci_lo,
        'low_ci_upper': low_ci_hi,
    }


def aggregate_stats_hard_stratified(gdf_admin: gpd.GeoDataFrame, gdf_cases: gpd.GeoDataFrame,
                                    start: pd.Timestamp, end: pd.Timestamp,
                                    b_start: pd.Timestamp, b_end: pd.Timestamp) -> pd.DataFrame:
    """
    Aggregate stats with HARD STRATIFICATION (2 rows per territory: high + low groups).

    Creates separate observations for high-risk and low-risk groups to enable:
    - Outbreak detection in specific risk groups
    - Testing artifact detection (change in testing composition between periods)
    - Decomposition of increase into: real outbreak vs testing strategy change

    This method attempts HARD stratification for all territories, but marks which ones
    have sufficient data (≥3 tests in both groups). Territories without sufficient data
    will fall back to SOFT stratification in run_model().

    Args:
        gdf_admin: GeoDataFrame with administrative boundaries
        gdf_cases: GeoDataFrame with HIV testing cases (must have 'risk_group' column)
        start: Start date of current analysis period
        end: End date of current analysis period
        b_start: Start date of baseline period
        b_end: End date of baseline period

    Returns:
        DataFrame with 2 rows per territory (high + low), columns:
        - territory_idx, territory_name, risk_group
        - all_tested_curr, recent_count_curr, recent_proportion_curr
        - all_tested_hist, recent_count_hist, recent_proportion_hist
        - can_use_hard: True if both groups have ≥3 tests in current period
    """
    # Check if risk_group exists
    if 'risk_group' not in gdf_cases.columns:
        logger.warning("risk_group column not found - HARD stratification requires risk_group data")
        logger.warning("Falling back to SOFT stratification for all territories")
        return None

    # Get territory name column
    name_col = None
    for col in ['ADM3_EN', 'ADM2_EN', 'ADM1_EN', 'ADM3_UA',
                'ADM2_UA', 'name', 'h3_id', 'h3index', 'territory_id']:
        if col in gdf_admin.columns:
            name_col = col
            break

    if name_col is None:
        # Fallback: use the index as a string identifier
        logger.warning("No territory name column found — using index as fallback")
        gdf_admin = gdf_admin.copy()
        gdf_admin['_territory_idx'] = gdf_admin.index.astype(str)
        name_col = '_territory_idx'

    logger.info(f"\nPreparing HARD STRATIFICATION data (2 rows per territory: high + low groups)")

    # Current period
    curr_all = gdf_cases[(gdf_cases['test_date'] >= start) & (gdf_cases['test_date'] <= end)]
    curr_recent = curr_all[curr_all['type'] == 'recent']

    # Historical period
    hist_all = gdf_cases[(gdf_cases['test_date'] >= b_start) & (gdf_cases['test_date'] <= b_end)]
    hist_recent = hist_all[hist_all['type'] == 'recent']

    # Spatial joins
    curr_all_joined = gpd.sjoin(curr_all, gdf_admin[['geometry']], how='left', predicate='within')
    curr_recent_joined = gpd.sjoin(curr_recent, gdf_admin[['geometry']], how='left', predicate='within')
    hist_all_joined = gpd.sjoin(hist_all, gdf_admin[['geometry']], how='left', predicate='within')
    hist_recent_joined = gpd.sjoin(hist_recent, gdf_admin[['geometry']], how='left', predicate='within')

    # Build stratified data (2 rows per territory: high + low)
    rows = []

    for territory_idx in range(len(gdf_admin)):
        territory_name = gdf_admin.iloc[territory_idx][name_col]

        # Current period - high risk
        curr_high_all = curr_all_joined[
            (curr_all_joined['index_right'] == territory_idx) &
            (curr_all_joined['risk_group'] == 'high')
        ]
        curr_high_recent = curr_recent_joined[
            (curr_recent_joined['index_right'] == territory_idx) &
            (curr_recent_joined['risk_group'] == 'high')
        ]

        # Current period - low risk
        curr_low_all = curr_all_joined[
            (curr_all_joined['index_right'] == territory_idx) &
            (curr_all_joined['risk_group'] == 'low')
        ]
        curr_low_recent = curr_recent_joined[
            (curr_recent_joined['index_right'] == territory_idx) &
            (curr_recent_joined['risk_group'] == 'low')
        ]

        # Historical period - high risk
        hist_high_all = hist_all_joined[
            (hist_all_joined['index_right'] == territory_idx) &
            (hist_all_joined['risk_group'] == 'high')
        ]
        hist_high_recent = hist_recent_joined[
            (hist_recent_joined['index_right'] == territory_idx) &
            (hist_recent_joined['risk_group'] == 'high')
        ]

        # Historical period - low risk
        hist_low_all = hist_all_joined[
            (hist_all_joined['index_right'] == territory_idx) &
            (hist_all_joined['risk_group'] == 'low')
        ]
        hist_low_recent = hist_recent_joined[
            (hist_recent_joined['index_right'] == territory_idx) &
            (hist_recent_joined['risk_group'] == 'low')
        ]

        # Counts
        n_curr_high = len(curr_high_all)
        n_curr_high_recent = len(curr_high_recent)
        n_curr_low = len(curr_low_all)
        n_curr_low_recent = len(curr_low_recent)

        n_hist_high = len(hist_high_all)
        n_hist_high_recent = len(hist_high_recent)
        n_hist_low = len(hist_low_all)
        n_hist_low_recent = len(hist_low_recent)

        # Proportions
        prop_curr_high = n_curr_high_recent / n_curr_high if n_curr_high > 0 else 0.0
        prop_curr_low = n_curr_low_recent / n_curr_low if n_curr_low > 0 else 0.0
        prop_hist_high = n_hist_high_recent / n_hist_high if n_hist_high > 0 else 0.0
        prop_hist_low = n_hist_low_recent / n_hist_low if n_hist_low > 0 else 0.0

        # Check if HARD stratification is possible (≥3 tests in both groups in current period)
        can_use_hard = (n_curr_high >= 3 and n_curr_low >= 3)

        # High-risk row
        rows.append({
            'territory_idx': territory_idx,
            'territory_name': territory_name,
            'risk_group': 'high',
            'all_tested_curr': n_curr_high,
            'recent_count_curr': n_curr_high_recent,
            'recent_proportion_curr': prop_curr_high,
            'all_tested_hist': n_hist_high,
            'recent_count_hist': n_hist_high_recent,
            'recent_proportion_hist': prop_hist_high,
            'can_use_hard': can_use_hard
        })

        # Low-risk row
        rows.append({
            'territory_idx': territory_idx,
            'territory_name': territory_name,
            'risk_group': 'low',
            'all_tested_curr': n_curr_low,
            'recent_count_curr': n_curr_low_recent,
            'recent_proportion_curr': prop_curr_low,
            'all_tested_hist': n_hist_low,
            'recent_count_hist': n_hist_low_recent,
            'recent_proportion_hist': prop_hist_low,
            'can_use_hard': can_use_hard
        })

    df_hard = pd.DataFrame(rows)

    # Calculate testing intensity for each territory-risk_group combination
    logger.info("Calculating testing intensity (test-months) for HARD stratification...")
    df_hard['testing_intensity_curr'] = 0.0
    df_hard['testing_intensity_hist'] = 0.0
    df_hard['n_active_months_curr'] = 0
    df_hard['n_active_months_hist'] = 0

    for idx, row in df_hard.iterrows():
        territory_idx = row['territory_idx']
        risk_group = row['risk_group']
        territory_geom = gdf_admin.iloc[territory_idx].geometry

        # Filter cases for this territory and risk group
        territory_cases_curr = curr_all[
            (curr_all.geometry.within(territory_geom)) &
            (curr_all['risk_group'] == risk_group)
        ]
        territory_cases_hist = hist_all[
            (hist_all.geometry.within(territory_geom)) &
            (hist_all['risk_group'] == risk_group)
        ]

        # Calculate intensity for current period
        intensity_curr = calculate_testing_intensity(territory_cases_curr, start, end)
        df_hard.at[idx, 'testing_intensity_curr'] = intensity_curr['weighted_intensity']
        df_hard.at[idx, 'n_active_months_curr'] = intensity_curr['n_active_months']

        # Calculate intensity for historical period
        intensity_hist = calculate_testing_intensity(territory_cases_hist, b_start, b_end)
        df_hard.at[idx, 'testing_intensity_hist'] = intensity_hist['weighted_intensity']
        df_hard.at[idx, 'n_active_months_hist'] = intensity_hist['n_active_months']

    # Log statistics
    n_territories = len(gdf_admin)
    n_can_use_hard = df_hard[df_hard['risk_group'] == 'high']['can_use_hard'].sum()
    n_must_use_soft = n_territories - n_can_use_hard

    logger.info(f"   Total territories: {n_territories}")
    logger.info(f"   Can use HARD (≥3 tests in both groups): {n_can_use_hard} ({n_can_use_hard/n_territories*100:.1f}%)")
    logger.info(f"   Must use SOFT (insufficient data): {n_must_use_soft} ({n_must_use_soft/n_territories*100:.1f}%)")

    # Log testing intensity summary
    high_rows = df_hard[df_hard['risk_group'] == 'high']
    low_rows = df_hard[df_hard['risk_group'] == 'low']
    logger.info(f"\nTesting Intensity Summary (HARD stratification):")
    logger.info(f"   High-risk group:")
    logger.info(f"     Current: mean={high_rows['testing_intensity_curr'].mean():.1f} test-months, "
               f"median={high_rows['testing_intensity_curr'].median():.1f}")
    logger.info(f"     Historical: mean={high_rows['testing_intensity_hist'].mean():.1f} test-months, "
               f"median={high_rows['testing_intensity_hist'].median():.1f}")
    logger.info(f"   Low-risk group:")
    logger.info(f"     Current: mean={low_rows['testing_intensity_curr'].mean():.1f} test-months, "
               f"median={low_rows['testing_intensity_curr'].median():.1f}")
    logger.info(f"     Historical: mean={low_rows['testing_intensity_hist'].mean():.1f} test-months, "
               f"median={low_rows['testing_intensity_hist'].median():.1f}")

    # Classify network stability for each risk group separately
    logger.info("Classifying network stability for HARD stratification...")

    # Collect intensity values by risk group
    high_intensities_curr = high_rows['testing_intensity_curr'].values
    high_intensities_hist = high_rows['testing_intensity_hist'].values
    low_intensities_curr = low_rows['testing_intensity_curr'].values
    low_intensities_hist = low_rows['testing_intensity_hist'].values

    # Initialize columns
    df_hard['network_stability'] = 'UNKNOWN'
    df_hard['network_stability_z'] = 0.0
    df_hard['intensity_change_pct'] = 0.0

    # Classify high-risk group
    for idx in high_rows.index:
        intensity_curr = df_hard.at[idx, 'testing_intensity_curr']
        intensity_hist = df_hard.at[idx, 'testing_intensity_hist']

        stability = classify_network_stability(
            intensity_curr, intensity_hist,
            high_intensities_curr, high_intensities_hist
        )

        df_hard.at[idx, 'network_stability'] = stability['stability']
        df_hard.at[idx, 'network_stability_z'] = stability['z_score'] if stability['z_score'] is not None else 0.0
        df_hard.at[idx, 'intensity_change_pct'] = stability['relative_change'] * 100 if stability['relative_change'] is not None else 0.0

    # Classify low-risk group
    for idx in low_rows.index:
        intensity_curr = df_hard.at[idx, 'testing_intensity_curr']
        intensity_hist = df_hard.at[idx, 'testing_intensity_hist']

        stability = classify_network_stability(
            intensity_curr, intensity_hist,
            low_intensities_curr, low_intensities_hist
        )

        df_hard.at[idx, 'network_stability'] = stability['stability']
        df_hard.at[idx, 'network_stability_z'] = stability['z_score'] if stability['z_score'] is not None else 0.0
        df_hard.at[idx, 'intensity_change_pct'] = stability['relative_change'] * 100 if stability['relative_change'] is not None else 0.0

    # Log stability distribution by risk group
    logger.info(f"[OK] Network stability classified (HARD stratification):")
    for risk_group in ['high', 'low']:
        group_rows = df_hard[df_hard['risk_group'] == risk_group]
        stability_counts = group_rows['network_stability'].value_counts()
        logger.info(f"  {risk_group.capitalize()}-risk group:")
        for category, count in stability_counts.items():
            pct = count / len(group_rows) * 100
            logger.info(f"    {category}: {count} ({pct:.1f}%)")

    return df_hard


def aggregate_stats(cfg, testing_sites_df, gdf_admin: gpd.GeoDataFrame, gdf_cases: gpd.GeoDataFrame,
                   start: pd.Timestamp, end: pd.Timestamp,
                   b_start: pd.Timestamp, b_end: pd.Timestamp) -> gpd.GeoDataFrame:
    """
    Aggregate HIV testing statistics.

    CRITICAL FIX: Now distinguishes between structural zeros (no testing site)
    and sampling zeros (site exists but 0 recent cases).

    Args:
        gdf_admin: GeoDataFrame with administrative boundaries
        gdf_cases: GeoDataFrame with HIV testing cases
        start: Start date of current analysis period
        end: End date of current analysis period
        b_start: Start date of baseline period
        b_end: End date of baseline period

    Returns:
        GeoDataFrame with aggregated statistics per territory
    """
    gdf_admin = gdf_admin.copy()
    gdf_admin['all_tested_curr'] = 0
    gdf_admin['recent_count_curr'] = 0
    gdf_admin['all_tested_hist'] = 0
    gdf_admin['recent_count_hist'] = 0

    # Initialize site_present flag to distinguish structural vs sampling zeros
    gdf_admin['site_present'] = False

    # Current period
    curr_all = gdf_cases[(gdf_cases['test_date'] >= start) & (gdf_cases['test_date'] <= end)]
    curr_recent = curr_all[curr_all['type'] == 'recent']

    # Baseline period
    hist_all = gdf_cases[(gdf_cases['test_date'] >= b_start) & (gdf_cases['test_date'] <= b_end)]
    hist_recent = hist_all[hist_all['type'] == 'recent']

    # Spatial joins - ONE TIME for all territories
    # Validate CRS before spatial operations — reproject cases to match hexagons
    for name, gdf in [('curr_all', curr_all), ('curr_recent', curr_recent),
                      ('hist_all', hist_all), ('hist_recent', hist_recent)]:
        if gdf.crs is None:
            logger.warning(f"{name} has no CRS — setting to EPSG:4326")
            gdf = gdf.set_crs('EPSG:4326', allow_override=True)
        if gdf.crs != gdf_admin.crs:
            logger.warning(f"CRS mismatch in {name}: {gdf.crs} vs {gdf_admin.crs} — reprojecting")
            gdf = gdf.to_crs(gdf_admin.crs)
        if name == 'curr_all': curr_all = gdf
        elif name == 'curr_recent': curr_recent = gdf
        elif name == 'hist_all': hist_all = gdf
        elif name == 'hist_recent': hist_recent = gdf

    # Log records lost due to sjoin failures
    n_curr_all_before = len(curr_all)
    n_curr_recent_before = len(curr_recent)
    n_hist_all_before = len(hist_all)
    n_hist_recent_before = len(hist_recent)

    joined_curr_all = gpd.sjoin(curr_all, gdf_admin[['geometry']], how='left', predicate='within')
    joined_curr_recent = gpd.sjoin(curr_recent, gdf_admin[['geometry']], how='left', predicate='within')
    joined_hist_all = gpd.sjoin(hist_all, gdf_admin[['geometry']], how='left', predicate='within')
    joined_hist_recent = gpd.sjoin(hist_recent, gdf_admin[['geometry']], how='left', predicate='within')

    # Count records with missing index_right (failed to match any polygon)
    n_curr_all_lost = joined_curr_all['index_right'].isna().sum()
    n_curr_recent_lost = joined_curr_recent['index_right'].isna().sum()
    n_hist_all_lost = joined_hist_all['index_right'].isna().sum()
    n_hist_recent_lost = joined_hist_recent['index_right'].isna().sum()

    # Log warnings if >1% records lost
    total_lost = n_curr_all_lost + n_curr_recent_lost + n_hist_all_lost + n_hist_recent_lost
    total_records = n_curr_all_before + n_curr_recent_before + n_hist_all_before + n_hist_recent_before

    if total_records > 0:
        pct_lost = (total_lost / total_records) * 100
        if pct_lost > 1.0:
            logger.warning(f"[WARN] SJOIN DATA LOSS: {total_lost}/{total_records} records ({pct_lost:.2f}%) failed to match any polygon")
            logger.warning(f"  Current all: {n_curr_all_lost}/{n_curr_all_before} ({n_curr_all_lost/n_curr_all_before*100:.1f}%)")
            logger.warning(f"  Current recent: {n_curr_recent_lost}/{n_curr_recent_before} ({n_curr_recent_lost/n_curr_recent_before*100 if n_curr_recent_before > 0 else 0:.1f}%)")
            logger.warning(f"  Historical all: {n_hist_all_lost}/{n_hist_all_before} ({n_hist_all_lost/n_hist_all_before*100 if n_hist_all_before > 0 else 0:.1f}%)")
            logger.warning(f"  Historical recent: {n_hist_recent_lost}/{n_hist_recent_before} ({n_hist_recent_lost/n_hist_recent_before*100 if n_hist_recent_before > 0 else 0:.1f}%)")
            logger.warning("  Possible causes: boundary sites, incorrect coordinates, missing polygons")
        elif total_lost > 0:
            logger.info(f"SJOIN: {total_lost}/{total_records} records ({pct_lost:.2f}%) outside polygons (acceptable)")
        else:
            logger.info(f"SJOIN: All {total_records} records matched successfully")

    # Aggregate using groupby (FAST - no loop!)
    curr_all_counts = joined_curr_all.groupby('index_right').size()
    curr_recent_counts = joined_curr_recent.groupby('index_right').size()
    hist_all_counts = joined_hist_all.groupby('index_right').size()
    hist_recent_counts = joined_hist_recent.groupby('index_right').size()

    # Map counts back to gdf_admin
    gdf_admin['all_tested_curr'] = gdf_admin.index.map(curr_all_counts).fillna(0).astype(int)
    gdf_admin['recent_count_curr'] = gdf_admin.index.map(curr_recent_counts).fillna(0).astype(int)
    gdf_admin['all_tested_hist'] = gdf_admin.index.map(hist_all_counts).fillna(0).astype(int)
    gdf_admin['recent_count_hist'] = gdf_admin.index.map(hist_recent_counts).fillna(0).astype(int)

    # Optional FRR correction (config-driven, off by default)
    frr = cfg.get('bayesian', {}).get('frr')
    if frr is not None and frr > 0:
        correction = np.round(gdf_admin['all_tested_curr'] * frr).astype(int)
        before = gdf_admin['recent_count_curr'].sum()
        gdf_admin['recent_count_curr'] = np.maximum(0, gdf_admin['recent_count_curr'] - correction)
        after = gdf_admin['recent_count_curr'].sum()
        logger.info(f"FRR correction: removed {before - after} false recent cases (FRR={frr})")
        # Also correct historical counts for consistency
        correction_hist = np.round(gdf_admin['all_tested_hist'] * frr).astype(int)
        gdf_admin['recent_count_hist'] = np.maximum(0, gdf_admin['recent_count_hist'] - correction_hist)

    # Mark territories with testing sites active during CURRENT period
    # Uses testing_sites sheet with activation/deactivation dates
    # This correctly handles wartime closures: if a site was active historically
    # but is now closed (e.g., occupied territory), it's NOT site_present
    try:
        if testing_sites_df is not None and len(testing_sites_df) > 0:
            df_sites = testing_sites_df
            # Filter sites active during current analysis period
            active_mask = df_sites['activation_date'] <= end
            # Deactivation: either no deactivation date (still active) or deactivated after period start
            active_mask = active_mask & (
                df_sites['deactivation_date'].isna() | (df_sites['deactivation_date'] >= start)
            )
            df_active_sites = df_sites[active_mask]

            if len(df_active_sites) > 0:
                # Spatial join: which hexagons contain active sites?
                gdf_active_sites = gpd.GeoDataFrame(
                    df_active_sites,
                    geometry=gpd.points_from_xy(df_active_sites.longitude, df_active_sites.latitude),
                    crs='EPSG:4326'
                )
                # Reproject sites to match hexagons CRS (not the other way around!)
                if gdf_active_sites.crs != gdf_admin.crs:
                    gdf_active_sites = gdf_active_sites.to_crs(gdf_admin.crs)
                sites_in_hex = gpd.sjoin(gdf_active_sites, gdf_admin, how='inner', predicate='within')
                hex_with_active_sites = set(sites_in_hex.index_right.unique())

                gdf_admin['site_present'] = gdf_admin.index.isin(hex_with_active_sites)
                logger.info(f"{len(df_active_sites)} sites active during {start.date()}-{end.date()}")
                logger.info(f"  → {gdf_admin['site_present'].sum()} hexagons with active sites")
            else:
                gdf_admin['site_present'] = gdf_admin['all_tested_curr'] > 0
                logger.warning("No sites active during current period — falling back to test counts")
        else:
            # No testing_sites sheet — fallback to original logic
            gdf_admin['site_present'] = (gdf_admin['all_tested_curr'] > 0) | (gdf_admin['all_tested_hist'] > 0)
            logger.info("No testing_sites data — using original site_present logic")
    except (KeyError, ValueError, AttributeError, IOError, FileNotFoundError) as e:
        logger.warning(f"Error loading testing_sites: {e} — using original site_present logic")
        gdf_admin['site_present'] = (gdf_admin['all_tested_curr'] > 0) | (gdf_admin['all_tested_hist'] > 0)

    # Log structural vs sampling zeros
    n_structural_zeros = (~gdf_admin['site_present']).sum()
    n_sampling_zeros = (gdf_admin['site_present'] & (gdf_admin['recent_count_curr'] == 0)).sum()
    n_active = gdf_admin['site_present'].sum()

    logger.info(f"Territory classification:")
    logger.info(f"  - Active sites: {n_active} ({n_active/len(gdf_admin)*100:.1f}%)")
    logger.info(f"  - Structural zeros (no site): {n_structural_zeros} ({n_structural_zeros/len(gdf_admin)*100:.1f}%)")
    logger.info(f"  - Sampling zeros (site present, 0 recent): {n_sampling_zeros}")

    # Proportions
    gdf_admin['recent_proportion_curr'] = np.where(
        gdf_admin['all_tested_curr'] > 0,
        gdf_admin['recent_count_curr'] / gdf_admin['all_tested_curr'],
        0
    )
    gdf_admin['recent_proportion_hist'] = np.where(
        gdf_admin['all_tested_hist'] > 0,
        gdf_admin['recent_count_hist'] / gdf_admin['all_tested_hist'],
        np.nan  # Use NaN instead of 0 for territories without historical data
    )

    # Calculate testing intensity for each territory
    logger.info("Calculating testing intensity (testo-months)...")

    gdf_admin['testing_intensity_curr'] = 0.0
    gdf_admin['testing_intensity_hist'] = 0.0
    gdf_admin['n_active_months_curr'] = 0
    gdf_admin['n_active_months_hist'] = 0

    for idx in range(len(gdf_admin)):
        territory_geom = gdf_admin.iloc[idx].geometry

        # Filter cases for this territory
        territory_cases_curr = curr_all[curr_all.geometry.within(territory_geom)]
        territory_cases_hist = hist_all[hist_all.geometry.within(territory_geom)]

        # Calculate intensity for current period
        intensity_curr = calculate_testing_intensity(territory_cases_curr, start, end)
        gdf_admin.at[idx, 'testing_intensity_curr'] = intensity_curr['weighted_intensity']
        gdf_admin.at[idx, 'n_active_months_curr'] = intensity_curr['n_active_months']

        # Calculate intensity for baseline period
        intensity_hist = calculate_testing_intensity(territory_cases_hist, b_start, b_end)
        gdf_admin.at[idx, 'testing_intensity_hist'] = intensity_hist['weighted_intensity']
        gdf_admin.at[idx, 'n_active_months_hist'] = intensity_hist['n_active_months']

    logger.info(f"[OK] Testing intensity calculated")
    logger.info(f"  Mean intensity (current): {gdf_admin['testing_intensity_curr'].mean():.1f} test-months")
    logger.info(f"  Mean intensity (baseline): {gdf_admin['testing_intensity_hist'].mean():.1f} test-months")

    # Classify network stability using z-score approach
    logger.info("Classifying network stability...")

    # Collect all intensity values for z-score calculation
    all_intensities_curr = gdf_admin['testing_intensity_curr'].values
    all_intensities_hist = gdf_admin['testing_intensity_hist'].values

    # Initialize columns
    gdf_admin['network_stability'] = 'UNKNOWN'
    gdf_admin['network_stability_z'] = 0.0
    gdf_admin['intensity_change_pct'] = 0.0

    for idx in range(len(gdf_admin)):
        intensity_curr = gdf_admin.at[idx, 'testing_intensity_curr']
        intensity_hist = gdf_admin.at[idx, 'testing_intensity_hist']

        # Classify stability
        stability = classify_network_stability(
            intensity_curr, intensity_hist,
            all_intensities_curr, all_intensities_hist
        )

        gdf_admin.at[idx, 'network_stability'] = stability['stability']
        gdf_admin.at[idx, 'network_stability_z'] = stability['z_score'] if stability['z_score'] is not None else 0.0
        gdf_admin.at[idx, 'intensity_change_pct'] = stability['relative_change'] * 100 if stability['relative_change'] is not None else 0.0

    # Log stability distribution
    stability_counts = gdf_admin['network_stability'].value_counts()
    logger.info(f"[OK] Network stability classified:")
    for category, count in stability_counts.items():
        pct = count / len(gdf_admin) * 100
        logger.info(f"  {category}: {count} ({pct:.1f}%)")

    return gdf_admin

