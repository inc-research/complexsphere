# -*- coding: utf-8 -*-
"""
complexsphere.memory
===============
Structural memory depth (Systemic Amnesia), ER=EPR Temporal Wormhole search,
and the ER=EPR Structural Analogue Report.

Covers notebook Cells 37, 39, and 41:
  - Cell 37 : Systemic Amnesia via spectral distance memory graph (top-k=3)
  - Cell 39 : ER=EPR Temporal Wormhole identification (full scan, min 3-yr gap)
  - Cell 41 : Structured analogue report with wormhole strength classification

Public API
----------
calculate_systemic_amnesia(annual_matrices, split_lookback_years, top_k)
identify_temporal_wormholes(annual_matrices, current_year, temporal_gap_min, lookback_window_years)
generate_atmospheric_epr_report(annual_matrices, location_labels, lookback_window_years)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh, eigvals as sp_eigvals

from .geometry import riemannian_distance


# ---------------------------------------------------------------------------
# Quantum Information Metrics (Cell 39 class)
# ---------------------------------------------------------------------------

class QuantumInformationMetrics:
    """
    Translates classical correlation matrices into quantum density matrices
    for information-theoretic entropy calculations.

    Exactly replicates the ``QuantumInformationMetrics`` class in Cell 39.
    """

    @staticmethod
    def normalize_to_density_matrix(
        c_matrix: np.ndarray, epsilon: float = 1e-8
    ) -> np.ndarray:
        """
        Translate a classical correlation matrix into a quantum density matrix ρ.

        1. Symmetrise.
        2. Eigen-decompose.
        3. Force positive semi-definiteness.
        4. Normalise trace to 1.
        """
        c_matrix = (c_matrix + c_matrix.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(c_matrix)
        eigenvalues = np.maximum(eigenvalues, 0)

        if np.any(eigenvalues < epsilon):
            eigenvalues += epsilon

        eigenvalues = eigenvalues / np.sum(eigenvalues)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    @staticmethod
    def von_neumann_entropy(c_matrix: np.ndarray) -> float:
        """
        Compute S(ρ) = −Tr(ρ ln ρ).

        Measures the absolute disorder of the climate network state.
        """
        rho = QuantumInformationMetrics.normalize_to_density_matrix(c_matrix)
        eigenvalues = eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        return float(-np.sum(eigenvalues * np.log(eigenvalues)))

    @staticmethod
    def mutual_information_bound(c1: np.ndarray, c2: np.ndarray) -> float:
        """
        Upper bound on the mutual information shared between two temporal states.
        This is the EPR link strength in the ER=EPR framework.
        """
        return min(
            QuantumInformationMetrics.von_neumann_entropy(c1),
            QuantumInformationMetrics.von_neumann_entropy(c2),
        )


# ---------------------------------------------------------------------------
# Cell 37 OOP scaffolding (preserved as library dataclasses)
# ---------------------------------------------------------------------------

@dataclass
class SystemState:
    """A single timestep correlation state."""
    timestep: int
    correlation_matrix: np.ndarray
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.correlation_matrix.shape[0] == self.correlation_matrix.shape[1], \
            "Correlation matrix must be square."


@dataclass
class MemoryLink:
    """A directed link from a current state to a structurally similar past state."""
    source_time: int
    target_time: int
    similarity: float
    temporal_distance: int = field(init=False)

    def __post_init__(self):
        assert self.source_time > self.target_time, "Source must be after target."
        self.temporal_distance = self.source_time - self.target_time


class TemporalMemoryGraph:
    """Graph of annual system states connected by structural similarity links."""

    def __init__(self):
        self.states: List[SystemState]  = []
        self.memory_links: List[MemoryLink] = []
        self.adjacency: Dict = defaultdict(list)

    def add_state(self, state: SystemState):
        self.states.append(state)

    def add_memory_link(self, link: MemoryLink):
        self.memory_links.append(link)
        self.adjacency[link.source_time].append(link)


class MemoryDetector:
    """Finds the ``top_k`` most structurally similar past states per timestep."""

    @staticmethod
    def spectral_distance(mat1: np.ndarray, mat2: np.ndarray) -> float:
        """Euclidean distance between sorted eigenvalue spectra."""
        e1 = np.sort(np.linalg.eigvalsh(mat1))
        e2 = np.sort(np.linalg.eigvalsh(mat2))
        return float(np.linalg.norm(e1 - e2))

    def build_memory_network(
        self, graph: TemporalMemoryGraph, top_k: int = 3
    ):
        """
        For each timestep, find the ``top_k`` most similar past states and
        create ``MemoryLink`` objects.

        Exactly replicates Cell 37's ``build_memory_network`` with
        ``top_k=3`` (the notebook default).
        """
        for current_idx in range(1, len(graph.states)):
            current_state = graph.states[current_idx]
            similarities  = []

            for past_idx in range(0, current_idx):
                past_state = graph.states[past_idx]
                dist = self.spectral_distance(
                    current_state.correlation_matrix,
                    past_state.correlation_matrix,
                )
                similarities.append((past_idx, dist))

            # Take top_k most similar (smallest spectral distance)
            selected = sorted(similarities, key=lambda x: x[1])[:top_k]

            for past_idx, dist in selected:
                link = MemoryLink(
                    source_time=current_idx,
                    target_time=past_idx,
                    similarity=dist,
                )
                graph.add_memory_link(link)


# ---------------------------------------------------------------------------
# Cell 37 — Systemic Amnesia
# ---------------------------------------------------------------------------

def calculate_systemic_amnesia(
    annual_matrices: Dict[int, np.ndarray],
    split_lookback_years: int = 8,
    top_k: int = 3,
) -> Dict:
    """
    Measure the Structural Memory Depth of the atmospheric correlation network
    by comparing a historical Macro era to a recent Micro era.

    Exactly replicates Cell 37, including the ``top_k=3`` depth averaging
    and the OOP ``TemporalMemoryGraph`` / ``MemoryDetector`` pipeline.

    Parameters
    ----------
    annual_matrices : dict of int → np.ndarray
        Output of :func:`~atmoplex.geometry.build_annual_matrices`.
    split_lookback_years : int
        Number of years to treat as the 'Micro' (recent) era.
        Default 8 (matches Cell 37: ``split_year = years[-8]``).
    top_k : int
        Number of most-similar past states to average over per timestep.
        Default 3 (matches Cell 37).

    Returns
    -------
    dict with keys:
        ``'Macro_Years'``, ``'Micro_Years'``,
        ``'Macro_Mean_Depth'``, ``'Macro_Max_Depth'``,
        ``'Micro_Mean_Depth'``, ``'Micro_Max_Depth'``,
        ``'Amnesia_Years_Lost'``, ``'Amnesia_Percentage'``.
    """
    years = sorted(annual_matrices.keys())
    split_year = years[-split_lookback_years] if len(years) >= split_lookback_years \
                 else years[len(years) // 2]

    macro_years = [y for y in years if y < split_year]
    micro_years = [y for y in years if y >= split_year]

    def _process_era(era_years: List[int]) -> Tuple[float, int]:
        graph = TemporalMemoryGraph()
        for t_idx, year in enumerate(era_years):
            state = SystemState(
                timestep=t_idx,
                correlation_matrix=annual_matrices[year],
                metadata={"year": year},
            )
            graph.add_state(state)

        detector = MemoryDetector()
        detector.build_memory_network(graph, top_k=top_k)

        depths = [link.temporal_distance for link in graph.memory_links] if graph.memory_links else [0]
        return float(np.mean(depths)), int(np.max(depths))

    macro_mean, macro_max = _process_era(macro_years)
    micro_mean, micro_max = _process_era(micro_years)

    amnesia_drop = macro_mean - micro_mean
    amnesia_pct  = (amnesia_drop / macro_mean * 100) if macro_mean > 0 else 0.0

    print(f"\n{'='*60}")
    print("SYSTEMIC AMNESIA ANALYSIS (Temporal Mean-Reversion)")
    print(f"{'='*60}")
    print(f"Historical Era ({macro_years[0]}-{macro_years[-1]}):")
    print(f"  -> Mean Memory Depth: {macro_mean:.2f} years")
    print(f"  -> Max Memory Reach:  {macro_max} years")
    print(f"\nCurrent Regime ({micro_years[0]}-{micro_years[-1]}):")
    print(f"  -> Mean Memory Depth: {micro_mean:.2f} years")
    print(f"  -> Max Memory Reach:  {micro_max} years")
    print("-" * 60)
    print(f"Systemic Amnesia Metric: {amnesia_drop:.2f} Years Lost ({amnesia_pct:.1f}% Decay)")

    print("\n--- Structural Interpretation ---")
    if amnesia_pct > 25.0:
        print("ALERT: Severe Systemic Amnesia Detected.")
        print("The current atmospheric correlation structure no longer references its historical baseline.")
        print("The system has lost its structural self-referencing capacity. Models relying on")
        print("rolling 10-to-30 year climatological averages will dramatically underestimate tail-risk events.")
    else:
        print("STATUS: Memory Intact.")
        print("The system is reliably referencing its historical states. Mean-reversion is functioning normally.")

    return {
        "Macro_Years":       macro_years,
        "Micro_Years":       micro_years,
        "Macro_Mean_Depth":  macro_mean,
        "Macro_Max_Depth":   macro_max,
        "Micro_Mean_Depth":  micro_mean,
        "Micro_Max_Depth":   micro_max,
        "Amnesia_Years_Lost": amnesia_drop,
        "Amnesia_Percentage": amnesia_pct,
    }


# ---------------------------------------------------------------------------
# Cell 39 — ER=EPR Temporal Wormhole identification
# ---------------------------------------------------------------------------

def identify_temporal_wormholes(
    annual_matrices: Dict[int, np.ndarray],
    current_year: Optional[int] = None,
    temporal_gap_min: int = 3,
    lookback_window_years: int = 10,
) -> pd.DataFrame:
    """
    Scan the full annual matrix history for ER=EPR temporal wormholes.

    Exactly replicates Cell 39:

    1. Computes Von Neumann Entropy for all years.
    2. Scans **all years** with a minimum ``temporal_gap_min``-year separation
       from ``current_year`` (not just a fixed lookback window) for wormhole
       candidates.
    3. For each candidate: Fisher-Rao geodesic distance, Mutual Information
       Bound, and Entanglement Score = MI / fisher_dist.
    4. Calculates the baseline entanglement ratio from the lookback window
       and adds ``Wormhole_Strength_Multiple``.

    Parameters
    ----------
    annual_matrices : dict of int → np.ndarray
        Output of :func:`~atmoplex.geometry.build_annual_matrices`.
    current_year : int, optional
        The reference year.  Defaults to the most recent year in the dict.
    temporal_gap_min : int
        Minimum chronological separation for a valid wormhole.  Default 3
        (matches Cell 39).
    lookback_window_years : int
        Window used to compute the baseline entanglement ratio.  Default 10.

    Returns
    -------
    pd.DataFrame
        Wormhole candidates sorted by ``Entanglement_Score`` descending,
        with columns ``'Historical_Analogue_Year'``,
        ``'Chronological_Gap'``, ``'Fisher_Geodesic_Distance'``,
        ``'Mutual_Information_Bound'``, ``'Entanglement_Score'``,
        ``'Wormhole_Strength_Multiple'``.
    """
    years = sorted(annual_matrices.keys())
    if current_year is None:
        current_year = years[-1]

    current_matrix  = annual_matrices[current_year]
    current_entropy = QuantumInformationMetrics.von_neumann_entropy(current_matrix)

    print(f"\n[+] Current Regime ({current_year}) Von Neumann Entropy: {current_entropy:.4f} nats")
    print("-" * 70)
    print(f"Scanning temporal spacetime for highly entangled historical analogues "
          f"(EPR Pairs) for {current_year}...")

    # Scan ALL past years with minimum temporal gap (not just lookback window)
    candidate_years = [y for y in years if y <= current_year - temporal_gap_min]
    wormhole_candidates = []

    for past_year in candidate_years:
        past_matrix = annual_matrices[past_year]

        fisher_dist       = riemannian_distance(past_matrix, current_matrix)
        mutual_info       = QuantumInformationMetrics.mutual_information_bound(
                                past_matrix, current_matrix)
        entanglement_score = mutual_info / fisher_dist if fisher_dist > 0 else 0.0

        wormhole_candidates.append({
            "Historical_Analogue_Year": past_year,
            "Chronological_Gap":        current_year - past_year,
            "Fisher_Geodesic_Distance": fisher_dist,
            "Mutual_Information_Bound": mutual_info,
            "Entanglement_Score":       entanglement_score,
        })

    if not wormhole_candidates:
        return pd.DataFrame()

    wormhole_df = (
        pd.DataFrame(wormhole_candidates)
          .sort_values(by="Entanglement_Score", ascending=False)
          .reset_index(drop=True)
    )

    # Baseline entanglement ratio from the lookback window
    lookback = [y for y in years if current_year - (lookback_window_years + 1) <= y < current_year]
    baseline_ratios = []
    for i in range(len(lookback)):
        for j in range(i + 1, len(lookback)):
            y1, y2  = lookback[i], lookback[j]
            f_dist  = riemannian_distance(annual_matrices[y1], annual_matrices[y2])
            m_info  = QuantumInformationMetrics.mutual_information_bound(
                          annual_matrices[y1], annual_matrices[y2])
            if f_dist > 0:
                baseline_ratios.append(m_info / f_dist)

    mean_baseline = float(np.mean(baseline_ratios)) if baseline_ratios else 1.0
    wormhole_df["Wormhole_Strength_Multiple"] = (
        wormhole_df["Entanglement_Score"] / mean_baseline
    )

    print("\nTOP TEMPORAL WORMHOLES (ER=EPR Pairs):")
    print(wormhole_df.head(3).set_index("Historical_Analogue_Year"))

    print("\n--- Structural Analogue Findings ---")
    best       = wormhole_df.iloc[0]
    best_yr    = int(best["Historical_Analogue_Year"])
    best_gap   = int(best["Chronological_Gap"])
    print(f"Primary Analogue: Current structure is mathematically entangled with {best_yr}.")
    print(f"Despite being {best_gap} chronological years apart, they occupy the same Information Geometry.")

    return wormhole_df


# ---------------------------------------------------------------------------
# Cell 41 — ER=EPR Structural Analogue Report
# ---------------------------------------------------------------------------

def generate_atmospheric_epr_report(
    annual_matrices: Dict[int, np.ndarray],
    location_labels: Optional[List[str]] = None,
    lookback_window_years: int = 10,
) -> Dict:
    """
    Generate the structured ER=EPR Structural Analogue Report.

    Exactly replicates the ``generate_atmospheric_epr_report`` function in
    Cell 41.  The report covers:

    1. Von Neumann Entropy of the current year vs 10-year historical mean.
    2. Primary temporal wormhole: Fisher-Rao distance, MI bound,
       Entanglement Ratio.
    3. Baseline entanglement ratio and Wormhole Strength Multiple.
    4. Coupling classification (STRONG ≥ 1.5 × / MODERATE ≥ 1.0 × / WEAK).

    Parameters
    ----------
    annual_matrices : dict of int → np.ndarray
        Output of :func:`~atmoplex.geometry.build_annual_matrices`.
    location_labels : list of str, optional
        Location names for the report header.
    lookback_window_years : int
        Window for historical entropy mean and baseline ratio.  Default 10.

    Returns
    -------
    dict with keys:
        ``'current_year'``, ``'current_entropy'``, ``'mean_historical_entropy'``,
        ``'entropy_delta_pct'``, ``'primary_wormhole'`` (dict),
        ``'wormhole_strength_multiple'``, ``'wormhole_df'`` (pd.DataFrame).
    """
    if location_labels is None:
        location_labels = ["Unknown"]

    years          = sorted(annual_matrices.keys())
    current_year   = years[-1]
    lookback_window = [y for y in years if current_year - (lookback_window_years + 1) <= y < current_year]

    if not lookback_window:
        print("Insufficient data for a 10-year lookback.")
        return {}

    current_matrix  = annual_matrices[current_year]
    current_entropy = QuantumInformationMetrics.von_neumann_entropy(current_matrix)

    hist_entropies       = [QuantumInformationMetrics.von_neumann_entropy(annual_matrices[y])
                            for y in lookback_window]
    mean_hist_entropy    = float(np.mean(hist_entropies))
    entropy_delta_pct    = (current_entropy - mean_hist_entropy) / mean_hist_entropy * 100

    # Wormhole candidates within lookback window
    wormhole_candidates = []
    for past_year in lookback_window:
        past_matrix = annual_matrices[past_year]
        fisher_dist = riemannian_distance(past_matrix, current_matrix)
        mutual_info = QuantumInformationMetrics.mutual_information_bound(past_matrix, current_matrix)
        ratio       = mutual_info / fisher_dist if fisher_dist > 0 else 0.0
        wormhole_candidates.append({
            "Target_Year":        past_year,
            "Temporal_Distance":  current_year - past_year,
            "Fisher_Rao_Distance": fisher_dist,
            "Mutual_Info_Bound":  mutual_info,
            "Entanglement_Ratio": ratio,
        })

    wormhole_df     = pd.DataFrame(wormhole_candidates)
    primary_wormhole = wormhole_df.loc[wormhole_df["Entanglement_Ratio"].idxmax()]

    baseline_ratios = []
    for i in range(len(lookback_window)):
        for j in range(i + 1, len(lookback_window)):
            y1, y2  = lookback_window[i], lookback_window[j]
            f_dist  = riemannian_distance(annual_matrices[y1], annual_matrices[y2])
            m_info  = QuantumInformationMetrics.mutual_information_bound(
                          annual_matrices[y1], annual_matrices[y2])
            if f_dist > 0:
                baseline_ratios.append(m_info / f_dist)

    mean_baseline         = float(np.mean(baseline_ratios)) if baseline_ratios else 1.0
    wormhole_strength_mul = float(primary_wormhole["Entanglement_Ratio"]) / mean_baseline

    # --- Print structured report ---
    W = 70
    print(f"\n{'='*W}")
    print(f"ER=EPR STRUCTURAL ANALOGUE REPORT: {current_year} vs Last {lookback_window_years} Years".center(W))
    print(f"Region: {', '.join(location_labels)}".center(W))
    print(f"{'='*W}")

    print("\n1. SYSTEM INFORMATION STATE (Von Neumann Entropy):")
    print(f"   • Current Von Neumann Entropy:       {current_entropy:.4f} nats")
    print(f"   • {lookback_window_years}-Year Historical Mean:            {mean_hist_entropy:.4f} nats")
    print(f"   • Entropy Deviation (Current):        {entropy_delta_pct:+.2f}%")
    if entropy_delta_pct < -10:
        print("     [!] System is abnormally rigid (low entropy). "
              "Elevated probability of sudden regime transition.")
    elif entropy_delta_pct > 10:
        print("     [!] System is in an elevated disorder state. "
              "Historical pattern models are structurally unreliable.")

    target_yr = int(primary_wormhole["Target_Year"])
    print("\n2. PRIMARY TEMPORAL ANALOGUE (ER=EPR Wormhole):")
    print(f"   • Most Structurally Similar Year:     {target_yr}")
    print(f"   • Chronological Separation:           {int(primary_wormhole['Temporal_Distance'])} years")
    print(f"   • Fisher-Rao Geodesic Distance:       {primary_wormhole['Fisher_Rao_Distance']:.4f}")
    print(f"   • Mutual Information Bound:           {primary_wormhole['Mutual_Info_Bound']:.4f}")
    print(f"   • Entanglement Ratio:                 {primary_wormhole['Entanglement_Ratio']:.4f}")

    print("\n3. ANALOGUE RELIABILITY:")
    print(f"   • {lookback_window_years}-Year Average Entanglement Ratio: {mean_baseline:.4f}")
    print(f"   • Wormhole Strength Multiple:         {wormhole_strength_mul:.2f}x baseline")

    print(f"\n{'='*W}")
    print("STRUCTURAL INTERPRETATION:")
    print(f"{'='*W}")

    if wormhole_strength_mul >= 1.5:
        print(f"STRONG COUPLING: The current atmospheric correlation geometry is deeply")
        print(f"analogous to {target_yr}. Structural patterns from {target_yr} provide the")
        print(f"strongest available historical reference for the current regime.")
    elif wormhole_strength_mul >= 1.0:
        print(f"MODERATE COUPLING: Structural similarity with {target_yr} is significant.")
        print(f"Weight analogue patterns alongside standard climatological baselines.")
    else:
        print(f"WEAK ENTANGLEMENT / SYSTEMIC AMNESIA: No strong structural analogue found.")
        print(f"The system is in a novel thermodynamic state. Substantially increase")
        print(f"uncertainty bounds on historical pattern-based analyses.")
    print(f"{'='*W}\n")

    return {
        "current_year":             current_year,
        "current_entropy":          current_entropy,
        "mean_historical_entropy":  mean_hist_entropy,
        "entropy_delta_pct":        entropy_delta_pct,
        "primary_wormhole":         primary_wormhole.to_dict(),
        "wormhole_strength_multiple": wormhole_strength_mul,
        "wormhole_df":              wormhole_df,
    }
