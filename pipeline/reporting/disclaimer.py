"""
Facility-based surveillance disclaimer.

These helpers produce the warning text that must appear on every output
artefact (Excel report, dashboard, recommendations file, map metadata).
The disclaimer exists because the pipeline geocodes test results to the
facility location, not to the patient's residence -- the resulting maps
look like population-level prevalence maps but actually trace the
catchment areas of large testing centres. Without this caveat any reader
not deeply familiar with the data will misinterpret the maps.
"""

from __future__ import annotations

from typing import Any, Dict


def get_facility_based_disclaimer() -> Dict[str, str]:
    """Return the long-form and short-form disclaimer text in EN and UA."""
    disclaimer_en = (
        "[WARN] FACILITY-BASED SURVEILLANCE DISCLAIMER:\n"
        "This analysis measures HIV recency testing results at specific testing facilities, "
        "NOT the geographic distribution of HIV infections in the population. "
        "GPS coordinates represent testing site locations, not patient residences. "
        "Large facilities (e.g., regional AIDS centers) attract patients from wide catchment areas "
        "(150-200 km), systematically attributing regional signals to urban locations. "
        "Results reflect testing network characteristics and should NOT be interpreted as "
        "population-level HIV prevalence or incidence maps."
    )

    disclaimer_ua = (
        "[WARN] WARNING REGARDING FACILITY-BASED SURVEILLANCE:\n"
        "This analysis measures recent HIV infection testing results at specific "
        "testing facilities, NOT the geographic distribution of HIV infections in the population. "
        "GPS coordinates represent testing site locations, not patient residences. "
        "Large facilities (e.g., regional AIDS centers) attract patients from wide territories "
        "(150-200 km), systematically attributing regional signals to urban locations. "
        "Results reflect testing network characteristics and should NOT be interpreted "
        "as maps of HIV prevalence or incidence at the population level."
    )

    return {
        'en': disclaimer_en,
        'ua': disclaimer_ua,
        'short_en': "[WARN] Facility-based data: reflects testing sites, not population distribution",
        'short_ua': "[WARN] Facility-based data: reflects testing sites, not population distribution",
    }


def add_disclaimer_to_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Inject the disclaimer into an existing metadata dict (mutates and returns)."""
    disclaimer = get_facility_based_disclaimer()
    metadata['facility_based_warning'] = disclaimer
    metadata['data_interpretation_warning'] = (
        "Results represent testing facility characteristics, not population-level HIV distribution. "
        "See facility_based_warning for details."
    )
    return metadata
