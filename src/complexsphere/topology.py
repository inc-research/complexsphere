# -*- coding: utf-8 -*-
"""
complexspere.topology
=================
PMFG topological regime-shift analysis and Ipsen-Mikhailov spectral distance.

Covers notebook Cells 29 and 31:
  - Cell 29 : Jaccard edge similarity + Euclidean Laplacian spectral distance
  - Cell 31 : Vectorised Ipsen-Mikhailov distance (Lorentzian integral form)

Public API
----------
analyze_topological_shift(pmfg_macro, pmfg_micro, title)
calculate_ipsen_mikhailov_distance(pmfg_macro, pmfg_micro, gamma, title)
analyze_all_networks(pmfg_graphs)
"""

from __future__ import annotations

import pickle
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Shared helper: absolute-weight Laplacian preparation
# ---------------------------------------------------------------------------

def _make_absolute_weights(G: nx.Graph) -> nx.Graph:
    """
    Return a copy of G with all edge weights replaced by their absolute value
    plus a small epsilon.

    This is required before computing the Laplacian matrix because negative
    edge weights break the normalised Laplacian's positive-semidefiniteness.
    Matches the ``make_weights_absolute`` helper in Cells 29 and 31.
    """
    G_abs = G.copy()
    for u, v, d in G_abs.edges(data=True):
        if "weight" in d:
            d["weight"] = abs(d["weight"]) + 1e-9
    return G_abs


def _laplacian_eigenvalues(G: nx.Graph) -> np.ndarray:
    """
    Compute the sorted real eigenvalues of the normalised Laplacian of G.
    """
    G_abs = _make_absolute_weights(G)
    L = nx.normalized_laplacian_matrix(G_abs).todense()
    evals = np.sort(np.linalg.eigvals(L).real)
    return evals


# ---------------------------------------------------------------------------
# Cell 29 — Topological regime shift (Jaccard + Euclidean spectral distance)
# ---------------------------------------------------------------------------

def analyze_topological_shift(
    pmfg_macro: nx.Graph,
    pmfg_micro: nx.Graph,
    title_macro: str = "Macro (30-Yr)",
    title_micro: str = "Micro (6-Mo)",
) -> Dict:
    """
    Quantify the structural regime shift between two PMFG graphs using
    Jaccard edge similarity and Euclidean Laplacian spectral distance.

    Exactly replicates the ``analyze_topological_shift`` function from Cell 29,
    including the absolute-weight correction before Laplacian computation.

    Parameters
    ----------
    pmfg_macro : nx.Graph
        The historical (macro) PMFG graph.
    pmfg_micro : nx.Graph
        The recent (micro) PMFG graph.
    title_macro : str
        Label for the macro era (printed in the report).
    title_micro : str
        Label for the micro era (printed in the report).

    Returns
    -------
    dict with keys:
        ``'jaccard'``, ``'spectral_distance'``,
        ``'new_edges'``, ``'severed_edges'``.
    """
    # Edge sets (sorted tuples for canonical comparison)
    edges_macro = {tuple(sorted((u, v))) for u, v in pmfg_macro.edges()}
    edges_micro = {tuple(sorted((u, v))) for u, v in pmfg_micro.edges()}

    intersection = edges_macro & edges_micro
    union        = edges_macro | edges_micro
    jaccard      = len(intersection) / len(union) if union else 0.0

    new_edges     = edges_micro - edges_macro
    severed_edges = edges_macro - edges_micro

    # Euclidean spectral distance (normalised Laplacian eigenvalues)
    evals_macro = _laplacian_eigenvalues(pmfg_macro)
    evals_micro = _laplacian_eigenvalues(pmfg_micro)
    spectral_dist = float(np.linalg.norm(evals_macro - evals_micro))

    print(f"\n{'='*70}")
    print(f"TOPOLOGICAL REGIME SHIFT: {title_macro} vs {title_micro}")
    print(f"{'='*70}")
    print(f"Network Stability (Jaccard):        {jaccard:.4f} (1.0 = Identical)")
    print(f"Euclidean Spectral Distance:        {spectral_dist:.4f} (Higher = More Severe Shift)")

    print(f"\n[+] NEW DEPENDENCIES FORMED ({len(new_edges)} Edges):")
    for u, v in sorted(new_edges)[:10]:
        print(f"    -> {u}  <==>  {v}")
    if len(new_edges) > 10:
        print(f"    ... and {len(new_edges) - 10} more.")

    print(f"\n[-] STRUCTURAL DEPENDENCIES SEVERED ({len(severed_edges)} Edges):")
    for u, v in sorted(severed_edges)[:5]:
        print(f"    -> {u}  <==>  {v}")
    if len(severed_edges) > 5:
        print(f"    ... and {len(severed_edges) - 5} more.")
    print(f"{'='*70}\n")

    return {
        "jaccard":          jaccard,
        "spectral_distance": spectral_dist,
        "new_edges":         new_edges,
        "severed_edges":     severed_edges,
    }


# ---------------------------------------------------------------------------
# Cell 31 — Ipsen-Mikhailov spectral distance
# ---------------------------------------------------------------------------

def calculate_ipsen_mikhailov_distance(
    pmfg_macro: nx.Graph,
    pmfg_micro: nx.Graph,
    gamma: float = 0.08,
    title: str = "Network",
) -> Tuple[float, float]:
    """
    Calculate the Ipsen-Mikhailov (IM) distance and Euclidean spectral
    distance between two PMFG graphs.

    Exactly replicates the ``calculate_spectral_metrics`` function from
    Cell 31, including:
    - Normalised Laplacian eigenvalues (bounded 0–2).
    - Vectorised Lorentzian integral using NumPy broadcasting.
    - ``gamma = 0.08`` (the notebook default).

    The IM distance formula is::

        IM² = K(A,A) + K(B,B) − 2·K(A,B)

    where ``K(X,Y)`` is the averaged Lorentzian integral::

        K(X,Y) = (1/|X||Y|) Σᵢ Σⱼ  (2γ / π) / ((λᵢ − μⱼ)² + 4γ²)

    Parameters
    ----------
    pmfg_macro : nx.Graph
        Historical (macro) PMFG graph.
    pmfg_micro : nx.Graph
        Recent (micro) PMFG graph.
    gamma : float
        Lorentzian broadening parameter.  Default 0.08 (matches Cell 31).
    title : str
        Label for the printed report.

    Returns
    -------
    (im_distance, euclidean_distance)
        Both as floats.
    """
    evals_macro = _laplacian_eigenvalues(pmfg_macro)
    evals_micro = _laplacian_eigenvalues(pmfg_micro)
    N = len(evals_macro)

    def _lorentzian_integral(e1: np.ndarray, e2: np.ndarray, g: float) -> float:
        """Vectorised averaged Lorentzian kernel."""
        diff   = e1[:, None] - e2[None, :]
        kernel = (2 * g) / (np.pi * (diff ** 2 + 4 * g ** 2))
        return float(np.sum(kernel) / (len(e1) * len(e2)))

    term_macro = _lorentzian_integral(evals_macro, evals_macro, gamma)
    term_micro = _lorentzian_integral(evals_micro, evals_micro, gamma)
    term_cross = _lorentzian_integral(evals_macro, evals_micro, gamma)

    im_sq      = term_macro + term_micro - 2 * term_cross
    im_distance = float(np.sqrt(max(im_sq, 0)))  # max() prevents tiny negative float errors

    euclidean_distance = float(np.linalg.norm(evals_macro - evals_micro))

    print(f"\n{'='*60}")
    print(f"REGIME SHIFT METRICS: {title} Topology")
    print(f"{'='*60}")
    print(f"Node Count (N):                 {N}")
    print(f"Ipsen-Mikhailov Distance:       {im_distance:.5f} (gamma={gamma})")
    print(f"Euclidean Spectral Distance:    {euclidean_distance:.5f}")
    print(f"\n[+] Macro Spectral Bounds:  [{evals_macro[1]:.4f} ... {evals_macro[-1]:.4f}]")
    print(f"[+] Micro Spectral Bounds:  [{evals_micro[1]:.4f} ... {evals_micro[-1]:.4f}]")

    print("\n--- Structural Interpretation ---")
    if im_distance > 0.15:
        print("ALERT: High IM Distance detected. The regional climate network has undergone")
        print("a structural phase transition. Historical correlation models are no longer reliable.")
    else:
        print("STATUS: Low IM Distance. The current micro-regime topology remains within")
        print("normal historical bounds. Mean-reverting statistical models remain applicable.")
    print(f"{'='*60}\n")

    return im_distance, euclidean_distance


# ---------------------------------------------------------------------------
# Convenience: run all four network comparisons at once
# ---------------------------------------------------------------------------

def analyze_all_networks(
    pmfg_graphs: Dict[str, nx.Graph],
    gamma: float = 0.08,
) -> Dict[str, Dict]:
    """
    Run both topological analyses (Jaccard + IM) for the Pearson and Partial
    PMFG pairs and return a nested results dictionary.

    Parameters
    ----------
    pmfg_graphs : dict
        Output of :func:`~atmoplex.network.build_all_pmfgs`.
        Expected keys: ``'macro_pearson'``, ``'micro_pearson'``,
                       ``'macro_partial'``, ``'micro_partial'``.
    gamma : float
        Ipsen-Mikhailov broadening parameter.  Default 0.08.

    Returns
    -------
    dict with keys ``'pearson'`` and ``'partial'``, each containing:
        ``'jaccard'``, ``'spectral_distance'``, ``'new_edges'``,
        ``'severed_edges'``, ``'im_distance'``, ``'euclidean_distance'``.
    """
    results = {}

    for net_type in ("pearson", "partial"):
        macro = pmfg_graphs[f"macro_{net_type}"]
        micro = pmfg_graphs[f"micro_{net_type}"]

        label = "General Climate (Pearson)" if net_type == "pearson" \
                else "Causal Drivers (Partial)"

        shift = analyze_topological_shift(
            macro, micro,
            title_macro=f"Macro {net_type.capitalize()}",
            title_micro=f"Micro {net_type.capitalize()}",
        )
        im_dist, euc_dist = calculate_ipsen_mikhailov_distance(
            macro, micro, gamma=gamma, title=label
        )

        results[net_type] = {
            **shift,
            "im_distance":       im_dist,
            "euclidean_distance": euc_dist,
        }

    return results
