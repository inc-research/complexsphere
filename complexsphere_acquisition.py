# -*- coding: utf-8 -*-
"""
atmoplex.acquisition
====================
NASA POWER data acquisition and formatting.

Covers notebook Cells 01, 03, and 05:
  - Cell 01 : Daily meteorological time-series (1995–present)
  - Cell 03 : 30-year climatological monthly constants
  - Cell 05 : Parse, format, assemble the daily dict and climatology dict

Public API
----------
fetch_daily_data(locations, start_date, end_date, parameters, community, save_csv)
fetch_climatology_data(locations, parameters, community, save_csv)
format_daily_data(raw_daily_dict)
format_climatology_data(raw_clim_dict)
load_from_csv(location_labels, daily_prefix, clim_prefix)
"""

from __future__ import annotations

import time
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants (matching the notebook exactly)
# ---------------------------------------------------------------------------

#: The 10 meteorological parameters used in the ADI framework.
#: Matches the notebook's ``parameters`` list in Cell 01.
DEFAULT_PARAMETERS: List[str] = [
    "ALLSKY_SFC_LW_DWN",   # Downward longwave radiation  (the 'Baseload Killer')
    "ALLSKY_SFC_SW_DWN",   # Downward shortwave radiation
    "T2M",                  # 2-metre air temperature
    "T2M_MAX",              # Daily maximum temperature
    "T2M_MIN",              # Daily minimum temperature
    "QV2M",                 # Specific humidity at 2 m
    "RH2M",                 # Relative humidity at 2 m
    "PRECTOTCORR",          # Precipitation (bias-corrected)
    "PS",                   # Surface pressure
    "WS2M",                 # Wind speed at 2 m
]

#: NASA POWER community code.  'SB' (Sustainable Buildings) is the correct
#: choice for thermal / energy analysis — it matches Cell 01 exactly.
DEFAULT_COMMUNITY: str = "SB"

#: Default four-node observation network for the Singapore study domain.
#: Matches Cell 01 ``locations`` / ``location_labels``.
SINGAPORE_NETWORK: List[Dict] = [
    {"name": "Singapore",   "lat": 1.35,  "lon": 103.82},
    {"name": "Kluang",      "lat": 2.03,  "lon": 103.33},
    {"name": "Malacca",     "lat": 2.19,  "lon": 102.25},
    {"name": "Kuala_Lumpur","lat": 3.14,  "lon": 101.69},
]

_MONTH_ABBR_TO_INT: Dict[str, int] = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5,  "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10,"NOV": 11, "DEC": 12,
}

_NASA_DAILY_BASE   = "https://power.larc.nasa.gov/api/temporal/daily/point"
_NASA_CLIM_BASE    = "https://power.larc.nasa.gov/api/temporal/climatology/point"

# ---------------------------------------------------------------------------
# 1.  Raw API fetch helpers
# ---------------------------------------------------------------------------

def fetch_daily_data(
    locations:  List[Dict],
    start_date: str = "19950101",
    end_date:   Optional[str] = None,
    parameters: List[str] = DEFAULT_PARAMETERS,
    community:  str = DEFAULT_COMMUNITY,
    save_csv:   bool = False,
    rate_limit_sleep: float = 1.0,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch daily meteorological data from the NASA POWER API for one or more
    locations and return raw (un-formatted) DataFrames.

    This function mirrors Cell 01 of the notebook exactly, including the
    CSV format, ``-END HEADER-`` stripping, and Julian-day date parsing.

    Parameters
    ----------
    locations : list of dict
        Each dict must have keys ``'name'``, ``'lat'``, ``'lon'``.
        Use :data:`SINGAPORE_NETWORK` for the default study domain.
    start_date : str
        Start date as ``'YYYYMMDD'`` string.  Default ``'19950101'``.
    end_date : str, optional
        End date as ``'YYYYMMDD'`` string.  Defaults to today.
    parameters : list of str
        NASA POWER parameter codes.  Defaults to :data:`DEFAULT_PARAMETERS`.
    community : str
        NASA POWER community code.  Must be ``'SB'`` for correct ADI
        parameters.  Default :data:`DEFAULT_COMMUNITY`.
    save_csv : bool
        If ``True``, writes each DataFrame to ``ADI_Target_{name}.csv``.
    rate_limit_sleep : float
        Seconds to sleep between requests (NASA recommends ≥ 1 s).

    Returns
    -------
    dict of str → pd.DataFrame
        Keys are location names; values are raw DataFrames with a
        ``DatetimeIndex`` named ``'Date'`` and a ``'Month'`` helper column.
    """
    if end_date is None:
        from datetime import date
        end_date = date.today().strftime("%Y%m%d")

    param_str = ",".join(parameters)
    results: Dict[str, pd.DataFrame] = {}

    for loc in locations:
        name = loc["name"]
        lat, lon = loc["lat"], loc["lon"]
        print(f"Requesting data for {name}: Lat={lat}, Lon={lon}")

        url = (
            f"{_NASA_DAILY_BASE}"
            f"?parameters={param_str}"
            f"&community={community}"
            f"&longitude={lon}&latitude={lat}"
            f"&start={start_date}&end={end_date}"
            f"&format=CSV"
        )

        response = requests.get(url, timeout=120)

        if response.status_code != 200:
            print(f"  [!] Error for {name}: HTTP {response.status_code}")
            time.sleep(rate_limit_sleep)
            continue

        # --- Strip the NASA CSV header (everything up to '-END HEADER-') ---
        raw_text   = response.text
        header_end = raw_text.find("-END HEADER-")
        data_str   = raw_text[header_end + len("-END HEADER-"):].strip()

        df = pd.read_csv(StringIO(data_str))

        # --- Reconstruct a proper DatetimeIndex from YEAR + Julian day (DY) ---
        # NASA POWER daily CSV has columns: YEAR, MO, DY
        df["Date"] = pd.to_datetime(
            df["YEAR"].astype(str) + df["DY"].astype(str),
            format="%Y%j",
        )
        df = df.drop(columns=["YEAR", "MO", "DY"], errors="ignore")
        df = df.set_index("Date")

        # --- Add Month helper column (used by entropy module) ---
        df["Month"] = df.index.month

        results[name] = df

        if save_csv:
            fname = f"ADI_Target_{name}.csv"
            df.to_csv(fname)
            print(f"  [✓] Saved {fname}")

        time.sleep(rate_limit_sleep)

    return results


def fetch_climatology_data(
    locations:  List[Dict],
    parameters: List[str] = DEFAULT_PARAMETERS,
    community:  str = DEFAULT_COMMUNITY,
    save_csv:   bool = False,
    rate_limit_sleep: float = 1.0,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch 30-year monthly climatological averages from NASA POWER.

    Mirrors Cell 03 of the notebook exactly (CSV format, community ``'SB'``).
    The raw response is returned un-transformed; call
    :func:`format_climatology_data` to produce the month-integer indexed
    ``(12 × 10)`` DataFrames expected by downstream modules.

    Returns
    -------
    dict of str → pd.DataFrame
        Raw climatology DataFrames keyed by location name.
    """
    param_str = ",".join(parameters)
    results: Dict[str, pd.DataFrame] = {}

    for loc in locations:
        name = loc["name"]
        lat, lon = loc["lat"], loc["lon"]
        print(f"Requesting Climatology for {name}: Lat={lat}, Lon={lon}")

        url = (
            f"{_NASA_CLIM_BASE}"
            f"?parameters={param_str}"
            f"&community={community}"
            f"&longitude={lon}&latitude={lat}"
            f"&format=CSV"
        )

        response = requests.get(url, timeout=120)

        if response.status_code != 200:
            print(f"  [!] Error for {name}: HTTP {response.status_code}")
            time.sleep(rate_limit_sleep)
            continue

        raw_text   = response.text
        header_end = raw_text.find("-END HEADER-")
        data_str   = raw_text[header_end + len("-END HEADER-"):].strip()

        df = pd.read_csv(StringIO(data_str))
        results[name] = df

        if save_csv:
            fname = f"ADI_Climatology_{name}.csv"
            df.to_csv(fname, index=False)
            print(f"  [✓] Saved {fname}")

        time.sleep(rate_limit_sleep)

    return results


# ---------------------------------------------------------------------------
# 2.  Formatting helpers  (Cell 05 logic)
# ---------------------------------------------------------------------------

def format_daily_data(raw_daily_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Ensure every daily DataFrame has a clean ``DatetimeIndex``, numeric
    columns only, and a ``'Month'`` helper column.

    This is a no-op if :func:`fetch_daily_data` was used (it already applies
    these transforms).  Call this when loading from CSV files.

    Parameters
    ----------
    raw_daily_dict : dict
        Output of :func:`fetch_daily_data` or a dict of CSV-loaded DataFrames.

    Returns
    -------
    dict of str → pd.DataFrame
        Cleaned DataFrames.
    """
    cleaned: Dict[str, pd.DataFrame] = {}
    for label, df in raw_daily_dict.items():
        df = df.copy()

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df = df.set_index(pd.to_datetime(df["Date"]))
                df = df.drop(columns=["Date"], errors="ignore")
            else:
                df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        # Add Month helper column if absent
        if "Month" not in df.columns:
            df["Month"] = df.index.month

        cleaned[label] = df

    print(f"  [✓] Formatted {len(cleaned)} daily DataFrames.")
    return cleaned


def format_climatology_data(
    raw_clim_dict: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Transpose and re-index the raw NASA POWER climatology response into the
    ``(12 × 10)`` month-integer indexed form expected by downstream modules.

    This exactly replicates the Cell 05 formatting block:
      1. ``set_index('PARAMETER').transpose()``
      2. Drop the ``'ANN'`` (annual average) row.
      3. Map month abbreviations (``'JAN'``…``'DEC'``) → integers 1–12.
      4. Drop any non-numeric columns.

    Parameters
    ----------
    raw_clim_dict : dict
        Output of :func:`fetch_climatology_data`.

    Returns
    -------
    dict of str → pd.DataFrame
        Formatted climatology DataFrames, indexed by integer month (1–12),
        with one column per meteorological parameter.
    """
    formatted: Dict[str, pd.DataFrame] = {}
    for label, df in raw_clim_dict.items():
        df = df.copy()

        # Transpose: rows become parameters, columns become months/annual
        if "PARAMETER" in df.columns:
            df = df.set_index("PARAMETER").transpose()
        # else assume already transposed

        # Drop annual aggregate row
        if "ANN" in df.index:
            df = df.drop("ANN")

        # Map month abbreviations to integers
        df.index = df.index.map(
            lambda m: _MONTH_ABBR_TO_INT.get(str(m).upper(), m)
        )
        df.index.name = "Month"

        # Keep only numeric columns
        df = df.select_dtypes(include=[np.number])

        formatted[label] = df

    print(f"  [✓] Formatted {len(formatted)} climatology DataFrames.")
    return formatted


# ---------------------------------------------------------------------------
# 3.  CSV loader  (convenience for offline / cached workflows)
# ---------------------------------------------------------------------------

def load_from_csv(
    location_labels: List[str],
    daily_prefix:    str = "ADI_Target_",
    clim_prefix:     str = "ADI_Climatology_",
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Load pre-downloaded daily and climatology CSVs from disk and apply the
    same formatting transforms as the live API path.

    Parameters
    ----------
    location_labels : list of str
        Location names matching the CSV filename stems, e.g.
        ``['Singapore', 'Kluang', 'Malacca', 'Kuala_Lumpur']``.
    daily_prefix : str
        Filename prefix for daily CSVs, default ``'ADI_Target_'``.
    clim_prefix : str
        Filename prefix for climatology CSVs, default ``'ADI_Climatology_'``.

    Returns
    -------
    (daily_data_dict, climatology_dict)
        Both dicts keyed by location name.
    """
    daily_raw: Dict[str, pd.DataFrame]  = {}
    clim_raw:  Dict[str, pd.DataFrame]  = {}

    for label in location_labels:
        daily_file = f"{daily_prefix}{label}.csv"
        clim_file  = f"{clim_prefix}{label}.csv"

        df_daily = pd.read_csv(daily_file, parse_dates=["Date"])
        df_daily = df_daily.set_index("Date")
        daily_raw[label] = df_daily

        df_clim = pd.read_csv(clim_file)
        clim_raw[label] = df_clim

        print(f"  [✓] {label}: loaded daily and climatology from CSV.")

    daily_data_dict  = format_daily_data(daily_raw)
    climatology_dict = format_climatology_data(clim_raw)

    return daily_data_dict, climatology_dict
