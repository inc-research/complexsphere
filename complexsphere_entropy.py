# -*- coding: utf-8 -*-
"""
complexsphere.entropy
================
Monthly Shannon Entropy, ADI index assembly, and entropy velocity.

Covers notebook Cells 07, 09, and 11:
  - Cell 07 : Per-node monthly Shannon Entropy across 10 weather parameters
  - Cell 09 : Condense to spatial ADI indices; compute Spatial_Entropy_Gradient
  - Cell 11 : Month-over-month entropy velocity (adi_differences)

Public API
----------
compute_node_entropy(daily_data_dict)
build_adi_index(node_entropy_results, primary_nodes, reference_node)
compute_entropy_velocity(monthly_adi_index_df)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


# ---------------------------------------------------------------------------
# Cell 07 — Monthly Shannon Entropy per node
# ---------------------------------------------------------------------------

def compute_node_entropy(
    daily_data_dict: Dict[str, pd.DataFrame],
    location_labels: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Calculate monthly Shannon Entropy (the ADI) for each observation node.

    This is a faithful implementation of Cell 07.  For every calendar month,
    the empirical probability distribution of daily values is computed via
    ``value_counts(normalize=True)`` for each of the 10 meteorological
    parameters.  Shannon Entropy is then calculated in **bits** (``base=2``)
    using ``scipy.stats.entropy``.

    Parameters
    ----------
    daily_data_dict : dict of str → pd.DataFrame
        Output of :func:`~atmoplex.acquisition.format_daily_data`.
        Each DataFrame must have a ``DatetimeIndex`` (or ``PeriodIndex``)
        and numeric weather-parameter columns.
    location_labels : list of str, optional
        Subset of keys to process.  Defaults to all keys in the dict.

    Returns
    -------
    dict of str → pd.DataFrame
        ``node_entropy_results``: monthly entropy DataFrame per location.
        Index is a ``PeriodIndex`` with frequency ``'M'``; one column per
        weather parameter.

    Notes
    -----
    The grouping is ``groupby(index.to_period('M'))``, producing exactly one
    entropy value per calendar month — matching the notebook's output shape
    of ``(N_months × 10_parameters)``.
    """
    if location_labels is None:
        location_labels = list(daily_data_dict.keys())

    node_entropy_results: Dict[str, pd.DataFrame] = {}

    print("Calculating Monthly Shannon Entropy (ADI) across 30-year dataset...")

    for label in location_labels:
        df = daily_data_dict[label]

        # Drop Month helper column and any non-numeric columns
        target_vars = df.select_dtypes(include=[np.number]).drop(
            columns=["Month"], errors="ignore"
        )

        # Group by calendar month and compute entropy for each parameter
        monthly_entropy_df = target_vars.groupby(
            target_vars.index.to_period("M")
        ).apply(
            lambda x: x.apply(
                lambda col: scipy_entropy(
                    col.value_counts(normalize=True), base=2
                )
            )
        )

        monthly_entropy_df.index.name = "Year_Month"
        node_entropy_results[label] = monthly_entropy_df

        print(
            f"  [✓] Entropy calculation complete for {label} "
            f"({len(monthly_entropy_df)} months)"
        )

    return node_entropy_results


# ---------------------------------------------------------------------------
# Cell 09 — ADI index assembly + Spatial Entropy Gradient
# ---------------------------------------------------------------------------

def build_adi_index(
    node_entropy_results: Dict[str, pd.DataFrame],
    primary_nodes: Optional[List[str]] = None,
    reference_node: Optional[str] = None,
) -> pd.DataFrame:
    """
    Condense per-node entropy matrices into the master ADI index DataFrame.

    This replicates Cell 09 exactly:

    1. For each node, average the 10-parameter entropy values into a single
       scalar ``ADI`` score per month.
    2. Assemble all node scores into ``monthly_adi_index_df``.
    3. Add ``Regional_ADI_Mean`` = mean across all nodes.
    4. Add ``Spatial_Entropy_Gradient`` using the notebook formula::

           primary_node_avg = mean(primary_nodes)
           Spatial_Entropy_Gradient = primary_node_avg − reference_node

       For the default Singapore network this simplifies to
       ``(Kuala_Lumpur − Singapore) / 2``.

    Parameters
    ----------
    node_entropy_results : dict of str → pd.DataFrame
        Output of :func:`compute_node_entropy`.
    primary_nodes : list of str, optional
        Nodes to average for the gradient numerator.
        Default: ``['Singapore', 'Kuala_Lumpur']``.
    reference_node : str, optional
        The node subtracted from the primary-node average.
        Default: ``'Singapore'``.

    Returns
    -------
    pd.DataFrame
        ``monthly_adi_index_df`` — one column per node plus
        ``Regional_ADI_Mean`` and ``Spatial_Entropy_Gradient``.
    """
    if primary_nodes is None:
        primary_nodes = ["Singapore", "Kuala_Lumpur"]
    if reference_node is None:
        reference_node = "Singapore"

    print("Condensing multi-parameter entropy into spatial indices...")

    location_adi_means: Dict[str, pd.Series] = {}
    for label, entropy_df in node_entropy_results.items():
        # Mean across all weather parameters → scalar ADI score per month
        location_adi_means[label] = entropy_df.mean(axis=1)

    monthly_adi_index_df = pd.DataFrame(location_adi_means)

    # Regional mean (across all nodes)
    monthly_adi_index_df["Regional_ADI_Mean"] = monthly_adi_index_df.mean(axis=1)

    # Spatial Entropy Gradient
    # Notebook formula: primary_node_avg - reference_node
    available_primary = [n for n in primary_nodes if n in monthly_adi_index_df.columns]
    if available_primary and reference_node in monthly_adi_index_df.columns:
        primary_node_avg = monthly_adi_index_df[available_primary].mean(axis=1)
        monthly_adi_index_df["Spatial_Entropy_Gradient"] = (
            primary_node_avg - monthly_adi_index_df[reference_node]
        )
    else:
        print(
            f"  [!] Could not compute Spatial_Entropy_Gradient: "
            f"primary_nodes={primary_nodes}, reference_node={reference_node}"
        )

    print("\nMonthly ADI Multi-Node Index calculated.")
    print("Preview of the most recent regional conditions:")
    print(monthly_adi_index_df.tail())

    return monthly_adi_index_df


# ---------------------------------------------------------------------------
# Cell 11 — Entropy Velocity (month-over-month differences)
# ---------------------------------------------------------------------------

def compute_entropy_velocity(
    monthly_adi_index_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the month-over-month difference in all ADI index columns.

    This replicates Cell 11 exactly.  Each output column is renamed with a
    ``_Velocity`` suffix so that downstream callers can unambiguously identify
    the derived series.

    Parameters
    ----------
    monthly_adi_index_df : pd.DataFrame
        Output of :func:`build_adi_index`.

    Returns
    -------
    pd.DataFrame
        ``adi_differences`` — same shape as input with ``_Velocity``-suffixed
        column names, first row dropped (no velocity at the start of history).

    Notes
    -----
    A positive velocity means disorder is accelerating (loss of stationarity).
    A negative velocity means mean-reversion or temporary stabilisation.
    """
    adi_differences = monthly_adi_index_df.diff().dropna()
    adi_differences.columns = [
        f"{col}_Velocity" for col in adi_differences.columns
    ]

    print("Entropy Velocity — Month-over-Month Change in ADI:")
    print(adi_differences.tail())
    print(
        "\nSpatial Entropy Gradient Velocity "
        "(Positive = primary-node disorder diverging from secondary baseline):"
    )
    if "Spatial_Entropy_Gradient_Velocity" in adi_differences.columns:
        print(adi_differences[["Spatial_Entropy_Gradient_Velocity"]].tail())

    return adi_differences
