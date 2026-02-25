# -*- coding: utf-8 -*-
"""
complexsphere.causality
==================
Temporal causality analysis: ADF stationarity testing, cross-correlation,
and bidirectional Granger causality tests.

Covers notebook Cell 35:
  - Cell 35 : ADF stationarity check, 1st-difference if non-stationary,
              cross-correlation at lags −3…+3,
              grangercausalitytests in both directions (maxlag=2)

Public API
----------
run_granger_causality(monthly_adi_index_df, velocity_df, max_lag, verbose)
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


# ---------------------------------------------------------------------------
# Helper: ADF stationarity enforcement
# ---------------------------------------------------------------------------

def _ensure_stationarity(series: pd.Series, name: str) -> pd.Series:
    """
    Test for stationarity using the Augmented Dickey-Fuller test.

    If the series is not stationary (p ≥ 0.05), apply a first difference.
    Exactly mirrors the ``ensure_stationarity`` helper in Cell 35.

    Parameters
    ----------
    series : pd.Series
        The time-series to test.
    name : str
        Display name for the diagnostic print statement.

    Returns
    -------
    pd.Series
        The original series if already stationary, otherwise the 1st-differenced
        series (with NaN head dropped by the caller).
    """
    result  = adfuller(series.dropna())
    p_value = result[1]

    if p_value < 0.05:
        print(f"  [✓] {name} is Stationary (p={p_value:.4f})")
        return series
    else:
        print(
            f"  [!] {name} is Non-Stationary (p={p_value:.4f}). "
            f"Applying 1st Difference..."
        )
        return series.diff()


# ---------------------------------------------------------------------------
# Cell 35 — Full Granger causality pipeline
# ---------------------------------------------------------------------------

def run_granger_causality(
    monthly_adi_index_df: pd.DataFrame,
    velocity_df: pd.DataFrame,
    max_lag: int = 2,
    verbose: bool = True,
) -> Dict:
    """
    Execute the full temporal causality pipeline from Cell 35.

    Steps:
    1. Aggregate monthly ADI to annual means.
    2. Merge with ``velocity_df`` on year.
    3. Apply ADF stationarity tests; first-difference if non-stationary.
    4. Compute cross-correlation at lags −3 to +3 years.
    5. Run ``grangercausalitytests`` in both directions:
       - ``System_Velocity → Spatial_Entropy_Gradient``
       - ``Spatial_Entropy_Gradient → System_Velocity``

    Parameters
    ----------
    monthly_adi_index_df : pd.DataFrame
        Output of :func:`~atmoplex.kinetics.classify_states` (or any later
        stage that still contains ``'Spatial_Entropy_Gradient'``).
        Must have a PeriodIndex or DatetimeIndex.
    velocity_df : pd.DataFrame
        Output of :func:`~atmoplex.geometry.compute_system_velocity`.
        Index should be the end-year integer.
    max_lag : int
        Maximum lag for Granger causality tests.  Default 2 (matches Cell 35).
    verbose : bool
        Print diagnostic output.  Default ``True``.

    Returns
    -------
    dict with keys:
        ``'gc_data'``, ``'cross_correlation'``,
        ``'granger_velocity_to_gradient'``, ``'granger_gradient_to_velocity'``.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)

    if verbose:
        print("Executing Temporal Causality Analysis: "
              "System Velocity vs. Spatial Entropy Gradient...")

    # --- 1. Annual aggregation of ADI ---
    # Ensure DatetimeIndex for resample
    if isinstance(monthly_adi_index_df.index, pd.PeriodIndex):
        adi_ts = monthly_adi_index_df.copy()
        adi_ts.index = adi_ts.index.to_timestamp()
    else:
        adi_ts = monthly_adi_index_df.copy()
        adi_ts.index = pd.to_datetime(adi_ts.index)

    annual_adi = adi_ts[["Regional_ADI_Mean", "Spatial_Entropy_Gradient"]].resample("Y").mean()
    annual_adi["Year"] = annual_adi.index.year
    annual_adi = annual_adi.set_index("Year")

    vel = velocity_df.copy()
    vel.index.name = "Year"

    merged = pd.merge(
        annual_adi[["Spatial_Entropy_Gradient"]],
        vel[["System_Velocity_Riemannian"]],
        left_index=True, right_index=True, how="inner",
    )

    # --- 2. Stationarity checks ---
    if verbose:
        print("\n--- Testing Stationarity ---")
    gc_data = pd.DataFrame()
    gc_data["Spatial_Entropy_Gradient"] = _ensure_stationarity(
        merged["Spatial_Entropy_Gradient"], "Spatial Entropy Gradient"
    )
    gc_data["System_Velocity"] = _ensure_stationarity(
        merged["System_Velocity_Riemannian"], "System Velocity"
    )
    gc_data = gc_data.dropna()

    # --- 3. Cross-correlation (lead/lag relationship) ---
    if verbose:
        print("\n--- Cross-Correlation Analysis "
              "(Does System Velocity Lead or Lag the Entropy Gradient?) ---")

    lags = np.arange(-3, 4)
    ccf_vals = [
        gc_data["Spatial_Entropy_Gradient"].corr(
            gc_data["System_Velocity"].shift(lag)
        )
        for lag in lags
    ]
    cross_corr: Dict = {}

    for lag, corr in zip(lags, ccf_vals):
        if pd.isna(corr):
            continue
        if lag > 0:
            direction = "Velocity LEADS Entropy"
        elif lag < 0:
            direction = "Velocity LAGS Entropy"
        else:
            direction = "Coincident"
        if verbose:
            print(f"  Lag {lag:+d} ({direction}): {corr:.4f}")
        cross_corr[int(lag)] = {"direction": direction, "correlation": corr}

    # --- 4. Granger causality tests ---
    def _run_gc_test(data: pd.DataFrame, label: str) -> Optional[Dict]:
        try:
            gc_res = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            out = {}
            for lag in range(1, max_lag + 1):
                p_val = gc_res[lag][0]["ssr_ftest"][1]
                sig   = "***" if p_val < 0.01 else "**" if p_val < 0.05 \
                        else "*" if p_val < 0.1 else ""
                out[lag] = {"p_value": p_val, "significance": sig}
                if verbose:
                    print(f"  Lag {lag}: p-value = {p_val:.4f} {sig}")
            return out
        except Exception as e:
            if verbose:
                print(f"  Test failed (likely insufficient data variation): {e}")
            return None

    if verbose:
        print(f"\n--- Granger Causality Tests (Max Lag: {max_lag} Years) ---")
        print("Null Hypothesis 1: System Velocity DOES NOT "
              "Granger-cause the Spatial Entropy Gradient")
    gc_vel_to_grad = _run_gc_test(
        gc_data[["Spatial_Entropy_Gradient", "System_Velocity"]],
        "Velocity → Gradient",
    )

    if verbose:
        print("\nNull Hypothesis 2: Spatial Entropy Gradient "
              "DOES NOT Granger-cause System Velocity")
    gc_grad_to_vel = _run_gc_test(
        gc_data[["System_Velocity", "Spatial_Entropy_Gradient"]],
        "Gradient → Velocity",
    )

    if verbose:
        print("\n(Note: *** p<0.01, ** p<0.05, * p<0.10. "
              "Reject the Null Hypothesis if p < 0.05)")

    return {
        "gc_data":                        gc_data,
        "cross_correlation":              cross_corr,
        "granger_velocity_to_gradient":   gc_vel_to_grad,
        "granger_gradient_to_velocity":   gc_grad_to_vel,
    }
