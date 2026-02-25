# -*- coding: utf-8 -*-
"""
complexsphere.volatility
===================
Seasonal OHLC volatility range analysis.

Covers notebook Cell 21:
  - Cell 21 : Rolling High–Low range vs 10-year same-calendar-month baseline
              → Volatility_Expansion_Multiple per ADI metric

Public API
----------
compute_ohlc_volatility_range(monthly_adi_index_df, recent_months, historical_years, target_metrics)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def compute_ohlc_volatility_range(
    monthly_adi_index_df: pd.DataFrame,
    recent_months_lookback: int = 6,
    historical_years_baseline: int = 10,
    target_metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compare the recent rolling High–Low range to the same calendar month's
    historical average range, and compute a Volatility Expansion Multiple.

    Exactly replicates Cell 21.

    The algorithm for each metric:

    1. Compute a rolling ``recent_months_lookback``-month High, Low, and Range.
    2. Take the most recent rolling Range value as ``Recent_Range``.
    3. Filter the historical rolling Range to the *same calendar month* over
       the prior ``historical_years_baseline`` years.
    4. Average those historical same-month Range values → ``Historical_Avg_Range``.
    5. ``Volatility_Expansion_Multiple = Recent_Range / Historical_Avg_Range``.

    A multiple > 1.0 means the current window is more volatile than the
    historical seasonal norm.

    Parameters
    ----------
    monthly_adi_index_df : pd.DataFrame
        Output of :func:`~atmoplex.kinetics.calculate_thermal_asymmetry` (or
        any DataFrame with a PeriodIndex/DatetimeIndex and the target columns).
    recent_months_lookback : int
        Rolling window size in months.  Default 6 (matches Cell 21).
    historical_years_baseline : int
        Number of prior years to use as the seasonal baseline.  Default 10.
    target_metrics : list of str, optional
        Columns to analyse.  Default ``['Regional_ADI_Mean',
        'Spatial_Entropy_Gradient']``.

    Returns
    -------
    pd.DataFrame
        ``range_summary_df`` indexed by metric name with columns:
        ``'Recent_High'``, ``'Recent_Low'``, ``'Recent_Range'``,
        ``'{N}Yr_Avg_Seasonal_Range'``, and ``'Volatility_Expansion_Multiple'``.
    """
    if target_metrics is None:
        target_metrics = ["Regional_ADI_Mean", "Spatial_Entropy_Gradient"]

    print(
        f"Executing OHLC Volatility Range Analysis: "
        f"Rolling {recent_months_lookback}-month window vs "
        f"{historical_years_baseline}-year seasonal baseline..."
    )

    # Ensure a plain DatetimeIndex for safe rolling operations (Cell 21 pattern)
    if isinstance(monthly_adi_index_df.index, pd.PeriodIndex):
        df_range = monthly_adi_index_df.copy()
        df_range.index = df_range.index.to_timestamp()
    else:
        df_range = monthly_adi_index_df.copy()
        df_range.index = pd.to_datetime(df_range.index)

    range_results = []

    for metric in target_metrics:
        if metric not in df_range.columns:
            print(f"  [!] Metric '{metric}' not found — skipping.")
            continue

        # A. Rolling OHLC range
        rolling_high  = df_range[metric].rolling(window=recent_months_lookback).max()
        rolling_low   = df_range[metric].rolling(window=recent_months_lookback).min()
        rolling_range = rolling_high - rolling_low

        # B. Most recent data point (represents the last N months)
        recent_range = rolling_range.iloc[-1]
        recent_high  = rolling_high.iloc[-1]
        recent_low   = rolling_low.iloc[-1]

        # C. Historical baseline — same calendar month, prior N years
        current_month = df_range.index[-1].month
        current_year  = df_range.index[-1].year

        hist_mask = (
            (df_range.index.month == current_month)
            & (df_range.index.year >= current_year - historical_years_baseline)
            & (df_range.index.year <  current_year)
        )
        historical_ranges = rolling_range[hist_mask]
        historical_avg = (
            historical_ranges.mean()
            if not historical_ranges.dropna().empty
            else np.nan
        )

        # D. Volatility Expansion Multiple
        if pd.isna(historical_avg) or historical_avg == 0:
            expansion_multiple = np.nan
        else:
            expansion_multiple = recent_range / historical_avg

        range_results.append({
            "Index_Metric": metric,
            "Recent_High":  recent_high,
            "Recent_Low":   recent_low,
            "Recent_Range": recent_range,
            f"{historical_years_baseline}Yr_Avg_Seasonal_Range": historical_avg,
            "Volatility_Expansion_Multiple": expansion_multiple,
        })

    range_summary_df = pd.DataFrame(range_results).set_index("Index_Metric")

    print("\nSeasonal High-to-Low Range Comparison:")
    print(range_summary_df)

    return range_summary_df
