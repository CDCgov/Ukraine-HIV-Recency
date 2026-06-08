"""
Testing-network characterisation: per-site profiles, intensity and stability.

These five routines describe how the *testing network* changed between
the baseline and the analysis window. They feed into the recommendation
narrative ("X high-risk-focused sites closed in territory Y") rather
than into the model itself, but they are first-class outputs because
they are what tells the reader whether a hotspot is a real epidemiological
signal or a change in who came through the door.

* :func:`analyze_site_profile` -- per-site baseline-period profile: risk
  composition, volume, recent-infection rate.
* :func:`calculate_testing_intensity` -- "testo-months", i.e. weighted
  intensity averaged over *active* months (months with at least one
  test). This is robust to site opening / closing partway through.
* :func:`classify_network_stability` -- z-score-based stability flag
  comparing this territory's intensity change to the typical change
  across all territories at the same level.
* :func:`analyze_network_change` -- per-territory inventory of which
  sites closed, opened or remained between baseline and current period;
  uses :func:`analyze_site_profile` for per-site context.
* :func:`generate_network_explanation` -- string formatter that turns
  the two outputs above into a single human-readable sentence for the
  RECOMMENDATIONS report.
"""

from __future__ import annotations

from typing import Any, Dict

import geopandas as gpd
import numpy as np
import pandas as pd


def analyze_site_profile(site_id: str, gdf_cases: gpd.GeoDataFrame,
                         b_start: pd.Timestamp, b_end: pd.Timestamp) -> Dict[str, Any]:
    """Profile a testing site from its baseline-period activity.

    Returns risk composition, monthly volume and recent-infection rate.
    ``NO_DATA`` is returned when the site had no tests in the baseline
    window; ``UNKNOWN_RISK`` when the risk-group column is missing.
    """
    site_tests = gdf_cases[
        (gdf_cases['site_id'] == site_id) &
        (gdf_cases['test_date'] >= b_start) &
        (gdf_cases['test_date'] <= b_end)
    ]

    if len(site_tests) == 0:
        return {
            'site_id': site_id,
            'profile_category': 'NO_DATA',
            'high_risk_pct': None,
            'recent_infection_rate': None,
            'avg_tests_per_month': 0,
            'total_tests': 0,
            'description': 'No baseline data available',
        }

    total_tests = len(site_tests)

    if 'risk_group' in site_tests.columns:
        high_risk_tests = len(site_tests[site_tests['risk_group'] == 'high'])
        high_risk_pct = (high_risk_tests / total_tests * 100) if total_tests > 0 else 0
    else:
        high_risk_pct = None

    recent_infections = len(site_tests[site_tests['type'] == 'recent'])
    recent_rate = (recent_infections / total_tests * 100) if total_tests > 0 else 0

    months = (b_end - b_start).days / 30.44
    avg_tests_per_month = total_tests / months if months > 0 else 0

    if high_risk_pct is not None:
        if high_risk_pct >= 60:
            profile_category = 'HIGH_RISK_FOCUSED'
            risk_desc = 'high-risk population'
        elif high_risk_pct >= 30:
            profile_category = 'MIXED_POPULATION'
            risk_desc = 'mixed risk population'
        else:
            profile_category = 'LOW_RISK_FOCUSED'
            risk_desc = 'low-risk population'
    else:
        profile_category = 'UNKNOWN_RISK'
        risk_desc = 'unknown risk profile'

    if avg_tests_per_month >= 40:
        volume_desc = 'high-volume site'
    elif avg_tests_per_month >= 15:
        volume_desc = 'medium-volume site'
    else:
        volume_desc = 'low-volume site'

    description = f"{volume_desc} serving {risk_desc}"

    return {
        'site_id': site_id,
        'profile_category': profile_category,
        'high_risk_pct': high_risk_pct,
        'recent_infection_rate': recent_rate,
        'avg_tests_per_month': avg_tests_per_month,
        'total_tests': total_tests,
        'description': description,
        'risk_description': risk_desc,
        'volume_description': volume_desc,
    }


def calculate_testing_intensity(gdf_cases: gpd.GeoDataFrame,
                                start_date: pd.Timestamp,
                                end_date: pd.Timestamp) -> Dict[str, Any]:
    """Weighted testing-intensity (testo-months) for a period.

    Returns ``weighted_intensity`` (tests per *active* month, robust to
    sites opening / closing mid-period), ``n_active_months``,
    ``n_calendar_months`` and the monthly counts list.
    """
    period_cases = gdf_cases[
        (gdf_cases['test_date'] >= start_date) &
        (gdf_cases['test_date'] <= end_date)
    ]

    if len(period_cases) == 0:
        return {
            'weighted_intensity': 0,
            'n_active_months': 0,
            'n_calendar_months': 0,
            'monthly_tests': [],
        }

    period_cases = period_cases.copy()
    period_cases['year_month'] = period_cases['test_date'].dt.to_period('M')
    monthly_counts = period_cases.groupby('year_month').size()

    n_active_months = len(monthly_counts)
    total_tests = monthly_counts.sum()
    weighted_intensity = total_tests / n_active_months if n_active_months > 0 else 0

    n_calendar_months = ((end_date.year - start_date.year) * 12 +
                         (end_date.month - start_date.month) + 1)

    return {
        'weighted_intensity': weighted_intensity,
        'n_active_months': n_active_months,
        'n_calendar_months': n_calendar_months,
        'monthly_tests': monthly_counts.tolist(),
    }


def classify_network_stability(intensity_curr: float, intensity_hist: float,
                               all_intensities_curr: np.ndarray,
                               all_intensities_hist: np.ndarray) -> Dict[str, Any]:
    """Stability flag from a z-score on relative intensity change.

    The z-score compares this territory's relative change to the
    distribution of relative changes across all territories at the same
    level. With < 3 territories available the function falls back to a
    fixed-threshold rule (|Δ| < 0.25 stable, < 0.50 moderate, else major).
    """
    if intensity_hist == 0:
        return {
            'stability': 'NO_BASELINE',
            'z_score': None,
            'absolute_change': intensity_curr,
            'relative_change': None,
            'description': 'No baseline testing intensity data',
        }

    absolute_change = intensity_curr - intensity_hist
    relative_change = absolute_change / intensity_hist

    all_changes = []
    for curr, hist in zip(all_intensities_curr, all_intensities_hist):
        if hist > 0:
            change = (curr - hist) / hist
            all_changes.append(change)

    if len(all_changes) < 3:
        if abs(relative_change) < 0.25:
            stability = 'STABLE'
            desc = 'Testing network stable'
        elif abs(relative_change) < 0.50:
            stability = 'MODERATE_CHANGE'
            desc = 'Testing network changed moderately'
        else:
            stability = 'MAJOR_CHANGE'
            desc = 'Testing network changed significantly'

        return {
            'stability': stability,
            'z_score': None,
            'absolute_change': absolute_change,
            'relative_change': relative_change,
            'description': desc,
        }

    mean_change = np.mean(all_changes)
    std_change = np.std(all_changes)

    if std_change == 0:
        z_score = 0
    else:
        z_score = (relative_change - mean_change) / std_change

    abs_z = abs(z_score)

    if abs_z < 1.0:
        stability = 'STABLE'
        desc = 'Testing network stable (change within typical variation)'
    elif abs_z < 2.0:
        stability = 'MODERATE_CHANGE'
        desc = 'Testing network changed moderately (change unusual but not extreme)'
    else:
        stability = 'MAJOR_CHANGE'
        desc = 'Testing network changed significantly (change beyond typical variation)'

    return {
        'stability': stability,
        'z_score': z_score,
        'absolute_change': absolute_change,
        'relative_change': relative_change,
        'description': desc,
        'mean_change_level': mean_change,
        'std_change_level': std_change,
    }


def analyze_network_change(territory_idx: int, gdf_admin: gpd.GeoDataFrame,
                           gdf_cases: gpd.GeoDataFrame, df_sites: pd.DataFrame,
                           start: pd.Timestamp, end: pd.Timestamp,
                           b_start: pd.Timestamp, b_end: pd.Timestamp) -> Dict[str, Any]:
    """Inventory closed / opened / stable sites within a territory.

    A site is active in a window if its ``activation_date`` precedes the
    window's end and its ``deactivation_date`` (when present) follows the
    window's start. Closed = active in baseline, not in current; opened
    is the reverse; stable = active in both. Each entry carries the
    per-site profile from :func:`analyze_site_profile`.
    """
    if df_sites is None or len(df_sites) == 0:
        return {'has_data': False}

    territory_geom = gdf_admin.iloc[territory_idx].geometry

    gdf_sites = gpd.GeoDataFrame(
        df_sites,
        geometry=gpd.points_from_xy(df_sites.longitude, df_sites.latitude),
        crs=gdf_admin.crs,
    )

    sites_in_territory = gdf_sites[gdf_sites.geometry.within(territory_geom)]

    if len(sites_in_territory) == 0:
        return {'has_data': False}

    closed_sites = []
    opened_sites = []
    stable_sites = []

    for _, site in sites_in_territory.iterrows():
        site_id = site['site_id']
        activation = site['activation_date']
        deactivation = site['deactivation_date']

        active_in_baseline = (activation <= b_end) and (pd.isna(deactivation) or deactivation >= b_start)
        active_in_current = (activation <= end) and (pd.isna(deactivation) or deactivation >= start)

        if active_in_baseline:
            profile = analyze_site_profile(site_id, gdf_cases, b_start, b_end)
        else:
            profile = {
                'site_id': site_id,
                'profile_category': 'NEW_SITE',
                'description': 'newly opened site',
            }

        if active_in_baseline and not active_in_current:
            closed_sites.append(profile)
        elif not active_in_baseline and active_in_current:
            opened_sites.append(profile)
        elif active_in_baseline and active_in_current:
            stable_sites.append(profile)

    return {
        'has_data': True,
        'closed_sites': closed_sites,
        'opened_sites': opened_sites,
        'stable_sites': stable_sites,
        'n_closed': len(closed_sites),
        'n_opened': len(opened_sites),
        'n_stable': len(stable_sites),
    }


def generate_network_explanation(network_analysis: Dict[str, Any],
                                 stability: Dict[str, Any]) -> str:
    """Compose the per-territory network-change sentence for RECOMMENDATIONS."""
    if not network_analysis.get('has_data', False):
        return "Testing network data not available for this territory."

    parts = []

    parts.append(stability['description'] + ".")

    if network_analysis['n_closed'] > 0:
        closed = network_analysis['closed_sites']

        high_risk_closed = sum(1 for s in closed if s.get('profile_category') == 'HIGH_RISK_FOCUSED')
        low_risk_closed = sum(1 for s in closed if s.get('profile_category') == 'LOW_RISK_FOCUSED')
        mixed_closed = sum(1 for s in closed if s.get('profile_category') == 'MIXED_POPULATION')

        if network_analysis['n_closed'] == 1:
            site = closed[0]
            if site.get('high_risk_pct') is not None:
                parts.append(f"One site closed: {site['description']} "
                             f"({site['high_risk_pct']:.0f}% high-risk clients, "
                             f"{site['avg_tests_per_month']:.0f} tests/month).")
            else:
                parts.append(f"One site closed: {site['description']}.")
        else:
            parts.append(f"{network_analysis['n_closed']} sites closed:")
            if low_risk_closed > 0:
                parts.append(f"  • {low_risk_closed} low-risk focused site(s)")
            if mixed_closed > 0:
                parts.append(f"  • {mixed_closed} mixed population site(s)")
            if high_risk_closed > 0:
                parts.append(f"  • {high_risk_closed} high-risk focused site(s)")

    if network_analysis['n_opened'] > 0:
        if network_analysis['n_opened'] == 1:
            parts.append(f"One new site opened.")
        else:
            parts.append(f"{network_analysis['n_opened']} new sites opened.")

    if network_analysis['n_stable'] > 0:
        parts.append(f"{network_analysis['n_stable']} site(s) remained operational.")

    return " ".join(parts)
