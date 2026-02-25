# -*- coding: utf-8 -*-
"""
complexsphere.network
================
Cross-node correlation matrices, partial correlation (precision matrix method),
and Planar Maximally Filtered Graph (PMFG) construction.

Covers notebook Cells 23, 25, and 27:
  - Cell 23 : Assemble 40-variable cross-node daily DataFrame; compute
              Macro (full baseline) and Micro (last 180 days) Pearson matrices
  - Cell 25 : Compute Macro/Micro partial correlation matrices via pinv
  - Cell 27 : Build four PMFG topological skeletons (macro/micro × pearson/partial)
              and serialise to .pkl files

Public API
----------
assemble_cross_node_network(daily_data_dict, micro_window_days)
calculate_partial_correlation(df)
calculate_pmfg(corr_matrix_df, sort_by_absolute)
build_all_pmfgs(macro_df, micro_df, save_pkl, output_dir)
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Cell 23 — Cross-node correlation matrix assembly
# ---------------------------------------------------------------------------

def assemble_cross_node_network(
    daily_data_dict: Dict[str, pd.DataFrame],
    micro_window_days: int = 180,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build the 40-variable cross-node network DataFrame and derive
    Macro (full baseline) and Micro (recent ``micro_window_days`` days)
    Pearson correlation matrices.

    Exactly replicates Cell 23:

    1. For each node, drop the ``'Month'`` helper column and any non-numeric
       columns, add a location prefix to every column name.
    2. Concatenate all nodes horizontally into ``network_df``.
    3. Drop any rows with NaN (alignment across nodes).
    4. ``macro_df`` = full ``network_df``.
    5. ``micro_df`` = last ``micro_window_days`` rows of ``network_df``.
    6. Compute Pearson correlation matrices for both.

    Parameters
    ----------
    daily_data_dict : dict of str → pd.DataFrame
        Output of :func:`~atmoplex.acquisition.format_daily_data`.
    micro_window_days : int
        Number of recent daily rows for the micro window.  Default 180
        (≈ 6 months, matching Cell 23).

    Returns
    -------
    (network_df, macro_df, macro_corr_matrix, micro_corr_matrix)
        - ``network_df``       : full aligned 40-column DataFrame
        - ``macro_df``         : same as ``network_df`` (full baseline)
        - ``macro_corr_matrix``: Pearson correlation, full baseline
        - ``micro_corr_matrix``: Pearson correlation, last 180 days
    """
    print("Constructing Multi-Node Spatio-Temporal Correlation Network Matrices...")

    aligned_dfs = []
    for label, df in daily_data_dict.items():
        temp_df = (
            df.select_dtypes(include=[np.number])
              .drop(columns=["Month"], errors="ignore")
        )
        temp_df = temp_df.add_prefix(f"{label}_")
        aligned_dfs.append(temp_df)

    network_df = pd.concat(aligned_dfs, axis=1).dropna()

    macro_df = network_df
    micro_df = network_df.tail(micro_window_days)

    macro_corr_matrix = macro_df.corr(method="pearson")
    micro_corr_matrix = micro_df.corr(method="pearson")

    print(f"\n[✓] Macro Cross-Node Correlation Matrix: {macro_corr_matrix.shape[0]} nodes")
    print(f"[✓] Micro Cross-Node Correlation Matrix:  {micro_corr_matrix.shape[0]} nodes")

    return network_df, macro_df, macro_corr_matrix, micro_corr_matrix


# ---------------------------------------------------------------------------
# Cell 25 — Partial correlation (precision matrix method)
# ---------------------------------------------------------------------------

def calculate_partial_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the partial correlation matrix using the Precision Matrix
    (Inverse Covariance) method.

    Exactly replicates the ``calculate_partial_corr`` function from Cell 25:

    1. Compute the Pearson correlation matrix.
    2. Compute the pseudo-inverse (``np.linalg.pinv``) to handle
       multi-collinearity in the 40-variable network.
    3. Normalise: ``partial_corr[i,j] = −precision[i,j]
       / sqrt(precision[i,i] × precision[j,j])``.
    4. Set the diagonal to 1.0.

    Parameters
    ----------
    df : pd.DataFrame
        The aligned daily data (either ``macro_df`` or ``micro_df`` from
        :func:`assemble_cross_node_network`).

    Returns
    -------
    pd.DataFrame
        Partial correlation matrix (same shape and index/columns as the
        Pearson matrix).
    """
    pearson_corr      = df.corr(method="pearson")
    precision_matrix  = np.linalg.pinv(pearson_corr.values)
    diagonal          = np.sqrt(np.diag(precision_matrix))
    partial_corr_arr  = -precision_matrix / np.outer(diagonal, diagonal)
    np.fill_diagonal(partial_corr_arr, 1.0)

    return pd.DataFrame(
        partial_corr_arr,
        index=pearson_corr.columns,
        columns=pearson_corr.columns,
    )


# ---------------------------------------------------------------------------
# Cell 27 — PMFG construction
# ---------------------------------------------------------------------------

def calculate_pmfg(
    corr_matrix_df: pd.DataFrame,
    sort_by_absolute: bool = True,
) -> nx.Graph:
    """
    Build a Planar Maximally Filtered Graph (PMFG) from a correlation matrix.

    Exactly replicates the ``calculate_pmfg`` function from Cell 27:

    1. Extract all unique edge weights.
    2. Sort by absolute weight descending (strong relationships first,
       regardless of sign).
    3. Greedily insert edges while the graph remains planar
       (``nx.check_planarity``).
    4. Stop when the graph has exactly ``3(N − 2)`` edges — the topological
       bound for a planar graph on N nodes.

    Strong negative correlations (e.g., Temperature vs. Radiational Cooling)
    are as structurally informative as positive ones; ``sort_by_absolute=True``
    ensures they are not filtered out.

    Parameters
    ----------
    corr_matrix_df : pd.DataFrame
        A square correlation or partial-correlation matrix (DataFrame form so
        column labels are preserved as node names).
    sort_by_absolute : bool
        Sort by ``abs(weight)`` descending.  Default ``True`` (matches Cell 27).

    Returns
    -------
    nx.Graph
        The PMFG with ``3(N-2)`` edges and original signed weights stored as
        edge attributes.
    """
    nodes = corr_matrix_df.columns.tolist()
    n     = len(nodes)

    # Extract all unique edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w = corr_matrix_df.iloc[i, j]
            edges.append((w, nodes[i], nodes[j]))

    key_fn = (lambda x: abs(x[0])) if sort_by_absolute else (lambda x: x[0])
    edges.sort(key=key_fn, reverse=True)

    max_edges = 3 * (n - 2)
    pmfg = nx.Graph()
    pmfg.add_nodes_from(nodes)

    added = 0
    for weight, u, v in edges:
        pmfg.add_edge(u, v, weight=weight)
        if not nx.check_planarity(pmfg)[0]:
            pmfg.remove_edge(u, v)
        else:
            added += 1
        if added >= max_edges:
            break

    return pmfg


def build_all_pmfgs(
    macro_corr_matrix: pd.DataFrame,
    micro_corr_matrix: pd.DataFrame,
    macro_partial_matrix: pd.DataFrame,
    micro_partial_matrix: pd.DataFrame,
    save_pkl: bool = True,
    output_dir: str = ".",
) -> Dict[str, nx.Graph]:
    """
    Build and optionally serialise all four PMFG topological skeletons.

    Exactly replicates Cell 27's four-graph construction block:
      - ``pmfg_macro_pearson``
      - ``pmfg_micro_pearson``
      - ``pmfg_macro_partial``
      - ``pmfg_micro_partial``

    Parameters
    ----------
    macro_corr_matrix : pd.DataFrame
        Full-baseline Pearson correlation matrix.
    micro_corr_matrix : pd.DataFrame
        Recent-window Pearson correlation matrix.
    macro_partial_matrix : pd.DataFrame
        Full-baseline partial correlation matrix.
    micro_partial_matrix : pd.DataFrame
        Recent-window partial correlation matrix.
    save_pkl : bool
        If ``True``, serialise each graph to a ``.pkl`` file.  Default ``True``.
    output_dir : str
        Directory for ``.pkl`` files.  Default current directory.

    Returns
    -------
    dict of str → nx.Graph
        Keys: ``'macro_pearson'``, ``'micro_pearson'``,
              ``'macro_partial'``, ``'micro_partial'``.
    """
    print("Constructing Planar Maximally Filtered Graphs (PMFG)...")

    specs = {
        "macro_pearson": macro_corr_matrix,
        "micro_pearson": micro_corr_matrix,
        "macro_partial": macro_partial_matrix,
        "micro_partial": micro_partial_matrix,
    }

    graphs: Dict[str, nx.Graph] = {}
    for name, matrix in specs.items():
        g = calculate_pmfg(matrix, sort_by_absolute=True)
        graphs[name] = g
        print(
            f"  [✓] {name}: "
            f"Nodes={g.number_of_nodes()}, Edges={g.number_of_edges()}"
        )

        if save_pkl:
            fpath = os.path.join(output_dir, f"pmfg_{name}.pkl")
            with open(fpath, "wb") as fh:
                pickle.dump(g, fh)
            print(f"       → Saved {fpath}")

    print("\nPMFG generation complete.")
    return graphs
