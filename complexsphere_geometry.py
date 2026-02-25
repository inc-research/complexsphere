# -*- coding: utf-8 -*-
"""
complexsphere.geometry
=================
Information geometry: Riemannian (Fisher-Rao) distance between correlation
matrices, annual matrix construction, and System-Time Velocity.

Covers notebook Cell 33:
  - Cell 33 : ``riemannian_distance``, annual correlation matrix loop,
              ``velocity_df`` with ``Velocity_Anomaly_Multiple``

Public API
----------
riemannian_distance(c1, c2)
build_annual_matrices(network_df, min_observations)
compute_system_velocity(annual_matrices)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import eigvals as sp_eigvals


# ---------------------------------------------------------------------------
# Core Riemannian distance
# ---------------------------------------------------------------------------

def riemannian_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    """
    Fisher Information (Riemannian) distance between two correlation matrices.

    Exactly replicates the ``riemannian_distance`` function in Cell 33:

    1. Regularise both matrices to be Positive Definite (add small diagonal
       offset if the minimum eigenvalue is below ``1e-6``).
    2. Solve the generalised eigenvalue problem ``C2 v = λ C1 v``.
    3. Return ``sqrt(Σ ln(λᵢ)²)``.

    This measures the geodesic distance in the space of symmetric positive
    definite matrices — i.e., the 'velocity' of structural change between
    two time periods.

    Parameters
    ----------
    c1 : np.ndarray
        Baseline correlation matrix (e.g., Year T).
    c2 : np.ndarray
        Subsequent correlation matrix (e.g., Year T+1).

    Returns
    -------
    float
        The Riemannian geodesic distance.
    """
    def _make_pd(mat: np.ndarray) -> np.ndarray:
        eig_min = np.min(np.real(np.linalg.eigvals(mat)))
        if eig_min < 1e-6:
            mat = mat + np.eye(mat.shape[0]) * (1e-6 - eig_min + 1e-6)
        return mat

    m1 = _make_pd(c1.copy())
    m2 = _make_pd(c2.copy())

    vals = sp_eigvals(m2, m1)
    vals = np.real(vals)
    vals = np.maximum(vals, 1e-15)

    return float(np.sqrt(np.sum(np.log(vals) ** 2)))


# ---------------------------------------------------------------------------
# Annual matrix construction
# ---------------------------------------------------------------------------

def build_annual_matrices(
    network_df: pd.DataFrame,
    min_observations: int = 30,
) -> Dict[int, np.ndarray]:
    """
    Construct a dictionary of annual Pearson correlation matrices.

    Exactly replicates the annual-matrix loop in Cell 33.  For each calendar
    year present in ``network_df``, the Pearson correlation matrix is
    computed if the year has at least ``min_observations`` rows.

    Parameters
    ----------
    network_df : pd.DataFrame
        The aligned cross-node daily DataFrame from
        :func:`~atmoplex.network.assemble_cross_node_network`.
        Must have a ``DatetimeIndex``.
    min_observations : int
        Minimum number of daily rows required to form a stable matrix.
        Default 30 (matches Cell 33).

    Returns
    -------
    dict of int → np.ndarray
        ``annual_matrices``: year → (N × N) correlation matrix as NumPy array.
    """
    years = sorted(network_df.index.year.unique())
    annual_matrices: Dict[int, np.ndarray] = {}

    print(f"Processing {len(years)} years of cross-node correlation structures...")

    for year in years:
        df_year = network_df[network_df.index.year == year]
        if len(df_year) > min_observations:
            annual_matrices[year] = df_year.corr(method="pearson").values

    print(f"  [✓] Annual matrices constructed: {len(annual_matrices)} years")
    return annual_matrices


# ---------------------------------------------------------------------------
# System-Time Velocity
# ---------------------------------------------------------------------------

def compute_system_velocity(
    annual_matrices: Dict[int, np.ndarray],
) -> pd.DataFrame:
    """
    Calculate the year-to-year Riemannian structural velocity.

    Exactly replicates Cell 33's velocity computation:

    1. For each consecutive year pair (T, T+1), compute
       ``riemannian_distance(matrix_T, matrix_{T+1})``.
    2. Assemble into ``velocity_df`` with columns
       ``'Transition_Period'`` and ``'System_Velocity_Riemannian'``.
    3. Add ``'Velocity_Anomaly_Multiple'`` =
       current velocity / historical mean velocity.

    Parameters
    ----------
    annual_matrices : dict of int → np.ndarray
        Output of :func:`build_annual_matrices`.

    Returns
    -------
    pd.DataFrame
        ``velocity_df`` indexed by the end year of each transition, with
        columns ``'Transition_Period'``, ``'System_Velocity_Riemannian'``,
        and ``'Velocity_Anomaly_Multiple'``.
    """
    valid_years = sorted(annual_matrices.keys())
    velocity_results = []

    for i in range(len(valid_years) - 1):
        y_t  = valid_years[i]
        y_t1 = valid_years[i + 1]

        dist = riemannian_distance(annual_matrices[y_t], annual_matrices[y_t1])

        velocity_results.append({
            "Transition_Period":         f"{y_t} -> {y_t1}",
            "Year_End":                  y_t1,
            "System_Velocity_Riemannian": dist,
        })

    velocity_df = pd.DataFrame(velocity_results).set_index("Year_End")

    hist_avg = velocity_df["System_Velocity_Riemannian"].mean()
    velocity_df["Velocity_Anomaly_Multiple"] = (
        velocity_df["System_Velocity_Riemannian"] / hist_avg
    )

    print("\nSystem-Time Velocity Matrix "
          "(Rate of Change in Annual Atmospheric Correlation Structure):")
    print(velocity_df.tail(10))

    print(f"\n[+] Historical Baseline Velocity: {hist_avg:.4f}")
    current_multiple = velocity_df["Velocity_Anomaly_Multiple"].iloc[-1]

    if current_multiple > 1.5:
        print(f"ALERT: Current System Velocity is {current_multiple:.2f}x the historical average.")
        print(
            "The atmospheric correlation topology is in active flux. "
            "Historical baseline models are rapidly diverging from current structure."
        )
    elif current_multiple < 0.8:
        print(f"STATUS: Current System Velocity is {current_multiple:.2f}x the historical average.")
        print(
            "The atmospheric correlation topology is structurally stagnant. "
            "Mean-reverting statistical models will over-perform."
        )
    else:
        print(
            f"STATUS: Current System Velocity is {current_multiple:.2f}x "
            f"the historical average — within normal bounds."
        )

    return velocity_df
