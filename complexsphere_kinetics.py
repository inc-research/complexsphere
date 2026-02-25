# -*- coding: utf-8 -*-
"""
complexsphere.kinetics
=================
Thermodynamic state classification, temporal census, relativistic kinetics,
and rolling thermal asymmetry.

Covers notebook Cells 13, 15, 17, and 19:
  - Cell 13 : Percentage-change state classification (5 % threshold)
  - Cell 15 : Temporal state census vs 10-year seasonal baseline
  - Cell 17 : Rapidity → bounded velocity → Lorentz factor
  - Cell 19 : Rolling Thermal Asymmetry Multiple (6-month gain/loss ratio)

Public API
----------
classify_states(monthly_adi_index_df, threshold)
compute_temporal_census(monthly_adi_index_df, recent_months, historical_years)
calculate_relativistic_kinetics(monthly_adi_index_df, c_max)
calculate_thermal_asymmetry(monthly_adi_index_df, rolling_window)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Cell 13 — Percentage-change state classification
# ---------------------------------------------------------------------------

def _classify_state_change(pct_diff: float) -> str:
    """
    Map a month-over-month percentage change to a thermodynamic state label.

    Mirrors the ``classify_state_change`` function in Cell 13 exactly,
    including the 5 % (0.05) threshold.

    States
    ------
    - ``'Dissipative Gain'``  : > +5 % — breaking stationarity (expanding disorder)
    - ``'Reversible Gain'``   : 0 to +5 % — fluctuation within buffering capacity
    - ``'Reversible Loss'``   : −5 to 0 % — normal cooling / mean-reversion
    - ``'Dissipative Loss'``  : < −5 % — rapid structural cooling or regime shift
    - ``'Stationary (No Change)'`` : zero or NaN
    """
    if pd.isna(pct_diff) or pct_diff == 0:
        return "Stationary (No Change)"
    elif pct_diff > 0.05:
        return "Dissipative Gain"
    elif 0 < pct_diff <= 0.05:
        return "Reversible Gain"
    elif -0.05 <= pct_diff < 0:
        return "Reversible Loss"
    elif pct_diff < -0.05:
        return "Dissipative Loss"
    return "Stationary (No Change)"


def classify_states(
    monthly_adi_index_df: pd.DataFrame,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Apply the 5 % pct_change threshold to classify each month into one of
    four thermodynamic states.

    This exactly replicates Cell 13:

    1. Compute ``pct_change()`` of all columns.
    2. Apply :func:`_classify_state_change` to ``Regional_ADI_Mean`` →
       ``Regional_State``.
    3. Apply to ``Spatial_Entropy_Gradient`` → ``Spatial_Gradient_State``.
    4. Drop the first row (which has no percentage change).

    Parameters
    ----------
    monthly_adi_index_df : pd.DataFrame
        Output of :func:`~atmoplex.entropy.build_adi_index`.
    threshold : float
        Percentage threshold for Dissipative vs Reversible classification.
        Must match the 0.05 default to reproduce notebook results.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with ``'Regional_State'`` and
        ``'Spatial_Gradient_State'`` columns appended, first row dropped.

    Notes
    -----
    The function modifies and returns a *copy* of the input DataFrame.
    """
    df = monthly_adi_index_df.copy()

    adi_pct_change = df.pct_change()

    def _classify(pct_diff: float) -> str:
        if pd.isna(pct_diff) or pct_diff == 0:
            return "Stationary (No Change)"
        elif pct_diff > threshold:
            return "Dissipative Gain"
        elif 0 < pct_diff <= threshold:
            return "Reversible Gain"
        elif -threshold <= pct_diff < 0:
            return "Reversible Loss"
        elif pct_diff < -threshold:
            return "Dissipative Loss"
        return "Stationary (No Change)"

    if "Regional_ADI_Mean" in adi_pct_change.columns:
        df["Regional_State"] = adi_pct_change["Regional_ADI_Mean"].apply(_classify)

    if "Spatial_Entropy_Gradient" in adi_pct_change.columns:
        df["Spatial_Gradient_State"] = adi_pct_change["Spatial_Entropy_Gradient"].apply(_classify)

    # Drop the first row (no percentage change available)
    df = df.dropna(subset=["Regional_State", "Spatial_Gradient_State"], how="all")

    print("Thermodynamic Structural State Classifications Applied (5% Threshold):")
    cols = [c for c in ["Regional_ADI_Mean", "Regional_State",
                        "Spatial_Entropy_Gradient", "Spatial_Gradient_State"]
            if c in df.columns]
    print(df[cols].tail(10))

    return df


# ---------------------------------------------------------------------------
# Cell 15 — Temporal State Census
# ---------------------------------------------------------------------------

def compute_temporal_census(
    monthly_adi_index_df: pd.DataFrame,
    recent_months_lookback: int = 3,
    historical_years_baseline: int = 10,
    state_column: str = "Spatial_Gradient_State",
) -> pd.DataFrame:
    """
    Compare the recent state frequency to a 10-year seasonal baseline.

    This exactly replicates Cell 15.  The key insight is that only the *same
    calendar months* from the historical window are used as the comparison
    baseline (seasonal apples-to-apples comparison).

    Parameters
    ----------
    monthly_adi_index_df : pd.DataFrame
        Output of :func:`classify_states`.  Must contain a PeriodIndex
        or DatetimeIndex and the ``state_column``.
    recent_months_lookback : int
        How many of the most recent months to treat as the 'current window'.
        Default 3 (matches Cell 15).
    historical_years_baseline : int
        How many prior years to use as the seasonal baseline.  Default 10.
    state_column : str
        Column containing thermodynamic state labels.
        Default ``'Spatial_Gradient_State'``.

    Returns
    -------
    pd.DataFrame
        ``census_comparison`` with columns:
        - ``'Recent_Period (Count)'``
        - ``'Historical_Avg (Count)'``
        - ``'Deviation_from_Norm'``

        Sorted by ``Deviation_from_Norm`` descending.
    """
    df = monthly_adi_index_df.copy()

    # Ensure a usable DatetimeIndex for year/month extraction
    if isinstance(df.index, pd.PeriodIndex):
        idx = df.index.to_timestamp()
    else:
        idx = pd.to_datetime(df.index)
    df.index = idx

    recent_period_df = df.tail(recent_months_lookback)
    recent_months    = recent_period_df.index.month.unique()

    end_hist_year   = df.index.year.max() - 1
    start_hist_year = end_hist_year - historical_years_baseline + 1

    hist_mask = (
        (df.index.year >= start_hist_year)
        & (df.index.year <= end_hist_year)
        & (df.index.month.isin(recent_months))
    )
    historical_period_df = df[hist_mask]

    # State counts
    recent_counts    = recent_period_df[state_column].value_counts()
    historical_total = historical_period_df[state_column].value_counts()
    historical_avg   = historical_total / historical_years_baseline

    census_comparison = pd.DataFrame({
        "Recent_Period (Count)":  recent_counts,
        "Historical_Avg (Count)": historical_avg,
    }).fillna(0)

    census_comparison["Deviation_from_Norm"] = (
        census_comparison["Recent_Period (Count)"]
        - census_comparison["Historical_Avg (Count)"]
    )

    print(
        f"\nExecuted Temporal State Census: Last {recent_months_lookback} Months "
        f"vs {historical_years_baseline}-Year Seasonal Baseline."
    )
    print("\nSpatial Entropy Gradient — State Frequency Census (Regime Detection):")
    print(census_comparison.sort_values(by="Deviation_from_Norm", ascending=False))

    return census_comparison


# ---------------------------------------------------------------------------
# Cell 17 — Relativistic Kinetics (Rapidity → Velocity → Lorentz)
# ---------------------------------------------------------------------------

def _calculate_true_lorentz(v: float, c: float = 1.0) -> float:
    """
    Compute the Lorentz time-dilation factor γ for a given velocity v.

    Mirrors the ``calculate_true_lorentz`` function in Cell 17 exactly,
    including the 0.9999 cap to prevent float division-by-zero.
    """
    if pd.isna(v) or v == 0:
        return 1.0
    v_capped = np.clip(abs(v), 0.0, 0.9999 * c)
    return 1.0 / np.sqrt(1.0 - (v_capped ** 2 / c ** 2))


def calculate_relativistic_kinetics(
    monthly_adi_index_df: pd.DataFrame,
    c_max: float = 1.0,
) -> pd.DataFrame:
    """
    Calculate Rapidity, bounded System Velocity, and Lorentz factor for
    each month in the ADI timeline.

    Exactly replicates Cell 17:

    1. **Rapidity** = month-over-month diff of ``Spatial_Entropy_Gradient``.
    2. **Standardised Rapidity** = Rapidity / historical std dev.
    3. **System_Velocity_v** = ``c_max × tanh(Standardised_Rapidity)``
       (bounded strictly between −1 and +1).
    4. **Lorentz_Vector** = ``1 / sqrt(1 − v²/c²)`` per month.
    5. Reports mean velocity and mean Lorentz factor grouped by
       ``Spatial_Gradient_State``.

    Parameters
    ----------
    monthly_adi_index_df : pd.DataFrame
        Output of :func:`classify_states`.  Must contain
        ``'Spatial_Entropy_Gradient'`` and ``'Spatial_Gradient_State'``.
    c_max : float
        The relativistic speed limit.  Default 1.0 (matches notebook).

    Returns
    -------
    pd.DataFrame
        The input DataFrame with three new columns appended:
        ``'Entropy_Rapidity'``, ``'Standardized_Rapidity'``,
        ``'System_Velocity_v'``, and ``'Lorentz_Vector'``.
    """
    df = monthly_adi_index_df.copy()

    print("Calculating System Time Dilation via Relativistic Rapidity...")

    # --- Step 1: Rapidity (unbounded momentum of entropy) ---
    df["Entropy_Rapidity"] = df["Spatial_Entropy_Gradient"].diff()

    # --- Step 2: Standardise rapidity to 1.0 = 1-sigma historical move ---
    historical_std = df["Entropy_Rapidity"].std()
    if historical_std == 0 or pd.isna(historical_std):
        historical_std = 1.0
    df["Standardized_Rapidity"] = df["Entropy_Rapidity"] / historical_std

    # --- Step 3: Bounded velocity via tanh ---
    df["System_Velocity_v"] = c_max * np.tanh(df["Standardized_Rapidity"])

    # --- Step 4: Lorentz factor ---
    df["Lorentz_Vector"] = df["System_Velocity_v"].apply(
        lambda v: _calculate_true_lorentz(v, c=c_max)
    )

    # --- Step 5: State-level aggregation (Cell 17 reporting block) ---
    if "Spatial_Gradient_State" in df.columns:
        state_metrics = df.groupby("Spatial_Gradient_State").agg(
            **{
                "Mean_Velocity (v)":      ("System_Velocity_v", "mean"),
                "Mean_Time_Dilation (γ)": ("Lorentz_Vector",    "mean"),
            }
        )
        print("\nSystem Time Dilation (Lorentz Vectors) per Classification:")
        print(state_metrics.sort_values(by="Mean_Time_Dilation (γ)", ascending=False))

        print("\n--- Structural Physics Translation ---")
        max_state = state_metrics["Mean_Time_Dilation (γ)"].idxmax()
        max_gamma = state_metrics["Mean_Time_Dilation (γ)"].max()
        print(
            f"The highest rate of structural mutation occurs during "
            f"'{max_state}' phases (γ = {max_gamma:.3f})."
        )
        print(
            "During these phases, the atmospheric system departs most "
            "significantly from linear time-homogeneous assumptions."
        )
        print(
            "Models relying on historical baselines will experience maximum "
            "forecasting failure during this structural state."
        )

    return df


# ---------------------------------------------------------------------------
# Cell 19 — Rolling Thermal Asymmetry Multiple
# ---------------------------------------------------------------------------

def _get_temporal_lorentz(v: float, c: float = 1.0) -> float:
    """
    Lorentz factor for rolling calculations — returns NaN at the singularity
    (instead of a cap) to avoid distorting rolling sums.

    Mirrors the ``get_temporal_lorentz`` function in Cell 19 exactly.
    """
    if pd.isna(v) or v == 0:
        return 1.0
    v_sq = v ** 2
    if 1.0 - v_sq <= 0:
        return np.nan
    return 1.0 / np.sqrt(1.0 - v_sq)


def calculate_thermal_asymmetry(
    monthly_adi_index_df: pd.DataFrame,
    rolling_window: int = 6,
    c_max: float = 1.0,
) -> pd.DataFrame:
    """
    Calculate the Rolling Thermal Asymmetry Multiple.

    This exactly replicates Cell 19.  The Lorentz factor is re-applied to the
    full timeline (using the NaN-safe variant), then separated into expansion
    events (``v > 0``) and contraction events (``v < 0``).  A
    ``rolling_window``-month rolling sum of each is computed and their ratio
    gives the Thermal Asymmetry Multiple.

    A value **> 1.0** means entropy expansion is structurally dominant.
    A value **< 1.0** means entropy contraction / mean-reversion is dominant.

    Parameters
    ----------
    monthly_adi_index_df : pd.DataFrame
        Output of :func:`calculate_relativistic_kinetics`.  Must contain
        ``'System_Velocity_v'``.
    rolling_window : int
        Rolling window in months.  Default 6 (matches Cell 19).
    c_max : float
        Speed limit used in the Lorentz calculation.  Default 1.0.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with four new columns appended:
        ``'Lorentz_Vector'`` (overwritten with NaN-safe version),
        ``'Gain_Lorentz'``, ``'Loss_Lorentz'``,
        and ``'Thermal_Asymmetry_Multiple'``.

    Notes
    -----
    Cell 19 overwrites ``Lorentz_Vector`` with the NaN-safe variant; this
    function does the same so the column is consistent for downstream use.
    """
    df = monthly_adi_index_df.copy()

    print("Calculating Rolling Thermal Asymmetry (Momentum of Time Dilation)...")

    # Re-apply Lorentz with NaN-safe variant (Cell 19 behaviour)
    df["Lorentz_Vector"] = df["System_Velocity_v"].apply(
        lambda v: _get_temporal_lorentz(v, c=c_max)
    )

    # Separate gain and loss Lorentz components
    df["Gain_Lorentz"] = np.where(
        df["System_Velocity_v"] > 0, df["Lorentz_Vector"], 0.0
    )
    df["Loss_Lorentz"] = np.where(
        df["System_Velocity_v"] < 0, df["Lorentz_Vector"], 0.0
    )

    # Rolling sums
    rolling_gains  = df["Gain_Lorentz"].rolling(window=rolling_window).sum()
    rolling_losses = df["Loss_Lorentz"].rolling(window=rolling_window).sum()
    rolling_losses = rolling_losses.replace(0, np.nan)

    df["Thermal_Asymmetry_Multiple"] = rolling_gains / rolling_losses

    print(
        f"\nRolling {rolling_window}-Month Thermal Asymmetry Multiple calculated."
    )
    print("Values > 1.0 indicate systemic bias toward Entropy Expansion.")
    cols = [c for c in
            ["Spatial_Gradient_State", "Lorentz_Vector", "Thermal_Asymmetry_Multiple"]
            if c in df.columns]
    print(df[cols].tail(10))

    return df
