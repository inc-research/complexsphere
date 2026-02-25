# -*- coding: utf-8 -*-
"""
complexsphere.report
===============
Metric harvester and per-run quantitative tear-sheet renderer.

Covers notebook Cells 43 and 45:
  - Cell 43 : Harvest all 8 metric modules into a single flat dict
  - Cell 45 : Render the 4-section ADI quantitative report with scored flags

Public API
----------
harvest_metrics(...)
render_tear_sheet(metrics)
"""

from __future__ import annotations

import pickle
from datetime import datetime
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

from .geometry import riemannian_distance
from .memory import QuantumInformationMetrics


# ---------------------------------------------------------------------------
# Private helpers (mirror the inline helpers in Cell 43)
# ---------------------------------------------------------------------------

def _compute_amnesia_inline(
    annual_matrices: Dict[int, np.ndarray],
    split_lookback_years: int = 8,
) -> tuple:
    """Lightweight amnesia calculation for the harvester (top-k=3 matching)."""
    if not annual_matrices:
        return np.nan, np.nan, np.nan

    years      = sorted(annual_matrices.keys())
    split_year = years[-split_lookback_years] if len(years) >= split_lookback_years \
                 else years[len(years) // 2]
    macro_years = [y for y in years if y < split_year]
    micro_years = [y for y in years if y >= split_year]

    def _spec_dist(m1, m2):
        e1 = np.sort(np.linalg.eigvalsh(m1))
        e2 = np.sort(np.linalg.eigvalsh(m2))
        return float(np.linalg.norm(e1 - e2))

    def _mean_depth(era_years, top_k=3):
        depths = []
        for i in range(1, len(era_years)):
            dists = [
                (j, _spec_dist(annual_matrices[era_years[i]], annual_matrices[era_years[j]]))
                for j in range(i)
            ]
            for past_idx, _ in sorted(dists, key=lambda x: x[1])[:top_k]:
                depths.append(i - past_idx)
        return float(np.mean(depths)) if depths else 0.0

    macro_mean = _mean_depth(macro_years)
    micro_mean = _mean_depth(micro_years)
    drop = macro_mean - micro_mean
    pct  = (drop / macro_mean * 100) if macro_mean > 0 else 0.0
    return pct, macro_mean, micro_mean


def _compute_wormholes_inline(
    annual_matrices: Dict[int, np.ndarray],
    current_year: int,
    lookback_window_years: int = 10,
    min_gap: int = 3,
) -> tuple:
    """Inline wormhole computation for the harvester."""
    years = sorted(annual_matrices.keys())
    current_matrix  = annual_matrices[current_year]
    current_entropy = QuantumInformationMetrics.von_neumann_entropy(current_matrix)

    hist_entropies = [
        QuantumInformationMetrics.von_neumann_entropy(annual_matrices[y])
        for y in years if y < current_year
    ]
    mean_hist = float(np.mean(hist_entropies)) if hist_entropies else np.nan
    entropy_delta = (current_entropy - mean_hist) / mean_hist * 100 if mean_hist else np.nan

    candidates = []
    for py in [y for y in years if y <= current_year - min_gap]:
        fd = riemannian_distance(annual_matrices[py], current_matrix)
        mi = QuantumInformationMetrics.mutual_information_bound(
                 annual_matrices[py], current_matrix)
        score = mi / fd if fd > 0 else 0.0
        candidates.append({
            "Historical_Analogue_Year": py,
            "Chronological_Gap":        current_year - py,
            "Fisher_Geodesic_Distance": fd,
            "Mutual_Information_Bound": mi,
            "Entanglement_Score":       score,
        })

    if not candidates:
        return None, current_entropy, mean_hist, entropy_delta

    wdf = pd.DataFrame(candidates).sort_values("Entanglement_Score", ascending=False)

    lookback = [y for y in years if current_year - (lookback_window_years + 1) <= y < current_year]
    baseline_ratios = []
    for i in range(len(lookback)):
        for j in range(i + 1, len(lookback)):
            fd2 = riemannian_distance(annual_matrices[lookback[i]], annual_matrices[lookback[j]])
            mi2 = QuantumInformationMetrics.mutual_information_bound(
                      annual_matrices[lookback[i]], annual_matrices[lookback[j]])
            if fd2 > 0:
                baseline_ratios.append(mi2 / fd2)

    mean_baseline = float(np.mean(baseline_ratios)) if baseline_ratios else 1.0
    wdf["Wormhole_Strength_Multiple"] = wdf["Entanglement_Score"] / mean_baseline
    return wdf, current_entropy, mean_hist, entropy_delta


def _edge_topology(
    G_macro: Optional[nx.Graph],
    G_micro: Optional[nx.Graph],
) -> tuple:
    """New edges, severed edges, Jaccard similarity."""
    if G_macro is None or G_micro is None:
        return 0, 0, np.nan
    em    = {tuple(sorted(e)) for e in G_macro.edges()}
    ec    = {tuple(sorted(e)) for e in G_micro.edges()}
    union = em | ec
    return len(ec - em), len(em - ec), (len(em & ec) / len(union) if union else np.nan)


def _load_pmfg(fname: str) -> Optional[nx.Graph]:
    try:
        with open(fname, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Cell 43 — Metric Harvester
# ---------------------------------------------------------------------------

def harvest_metrics(
    annual_matrices:      Optional[Dict[int, np.ndarray]] = None,
    monthly_adi_index_df: Optional[pd.DataFrame]          = None,
    velocity_df:          Optional[pd.DataFrame]          = None,
    range_summary_df:     Optional[pd.DataFrame]          = None,
    census_comparison:    Optional[pd.DataFrame]          = None,
    pmfg_graphs:          Optional[Dict[str, nx.Graph]]   = None,
    pmfg_pkl_dir:         str = ".",
    im_partial:           float = np.nan,
    euc_partial:          float = np.nan,
    im_pearson:           float = np.nan,
    euc_pearson:          float = np.nan,
    location_labels:      Optional[List[str]] = None,
    lookback_window_years: int = 10,
) -> Dict:
    """
    Harvest all 8 metric modules into a single flat dictionary.

    Exactly replicates Cell 43's Metric Harvester, covering:

    1. **Temporal context** — reference year, location labels
    2. **Systemic Amnesia** — pct decay, macro/micro mean depth
    3. **ER=EPR Temporal Analogue** — wormhole year, strength, fisher dist
    4. **ADI Volatility Range** — expansion multiple for ADI and gradient
    5. **Thermal Asymmetry & Current State** — asymmetry multiple, state label
    6. **System-Time Velocity** — current multiple, transition period
    7. **PMFG Topology** — Jaccard, IM distance, new/severed edges (×2 networks)
    8. **State Census** — Dissipative Loss recent count and deviation

    Parameters
    ----------
    annual_matrices : dict, optional
        Output of :func:`~atmoplex.geometry.build_annual_matrices`.
    monthly_adi_index_df : pd.DataFrame, optional
        Output of :func:`~atmoplex.kinetics.calculate_thermal_asymmetry`.
    velocity_df : pd.DataFrame, optional
        Output of :func:`~atmoplex.geometry.compute_system_velocity`.
    range_summary_df : pd.DataFrame, optional
        Output of :func:`~atmoplex.volatility.compute_ohlc_volatility_range`.
    census_comparison : pd.DataFrame, optional
        Output of :func:`~atmoplex.kinetics.compute_temporal_census`.
    pmfg_graphs : dict, optional
        Output of :func:`~atmoplex.network.build_all_pmfgs`.
        If ``None``, tries to load from ``pmfg_pkl_dir``.
    pmfg_pkl_dir : str
        Directory containing PMFG ``.pkl`` files.
    im_partial, euc_partial, im_pearson, euc_pearson : float
        Spectral distances from :func:`~atmoplex.topology.analyze_all_networks`.
    location_labels : list of str, optional
        Location names for the report header.
    lookback_window_years : int
        Wormhole lookback window.

    Returns
    -------
    dict
        Flat dictionary of all ``report_*`` scalars ready for
        :func:`render_tear_sheet`.
    """
    m: Dict = {}

    print("Harvesting atmospheric metrics for report generation...")

    # --- MODULE 1: Temporal context ---
    if annual_matrices:
        years_all = sorted(annual_matrices.keys())
        m["report_current_year"] = years_all[-1]
    else:
        m["report_current_year"] = datetime.now().year
    m["report_location_labels"] = location_labels or ["Unknown"]
    print(f"  [✓] Reference year = {m['report_current_year']}")
    print(f"  [✓] Observation nodes: {', '.join(m['report_location_labels'])}")

    # --- MODULE 2: Systemic Amnesia ---
    if annual_matrices:
        amnesia_pct, macro_depth, micro_depth = _compute_amnesia_inline(annual_matrices)
    else:
        amnesia_pct = macro_depth = micro_depth = np.nan

    m["report_amnesia_pct"]       = amnesia_pct
    m["report_macro_mem_depth"]   = macro_depth
    m["report_micro_mem_depth"]   = micro_depth
    print(f"  [✓] Structural Amnesia: {amnesia_pct:.1f}% decay  "
          f"(Hist depth={macro_depth:.2f} yrs → Current={micro_depth:.2f} yrs)")

    # --- MODULE 3: ER=EPR Temporal Analogue ---
    if annual_matrices:
        _wdf, vn_entropy, hist_entropy_mean, entropy_delta = _compute_wormholes_inline(
            annual_matrices, m["report_current_year"], lookback_window_years
        )
    else:
        _wdf = None
        vn_entropy = hist_entropy_mean = entropy_delta = np.nan

    m["report_vn_entropy"]         = vn_entropy
    m["report_hist_entropy_mean"]  = hist_entropy_mean
    m["report_entropy_delta_pct"]  = entropy_delta

    if _wdf is not None and not _wdf.empty:
        _primary = _wdf.iloc[0]
        m["report_analogue_year"]        = int(_primary["Historical_Analogue_Year"])
        m["report_analogue_gap"]         = int(_primary["Chronological_Gap"])
        m["report_analogue_fisher_dist"] = float(_primary["Fisher_Geodesic_Distance"])
        m["report_analogue_mutual_info"] = float(_primary["Mutual_Information_Bound"])
        m["report_analogue_entanglement"]= float(_primary["Entanglement_Score"])
        m["report_wormhole_strength"]    = float(_primary["Wormhole_Strength_Multiple"])
        m["report_wormhole_df"]          = _wdf.head(3).copy()
    else:
        m["report_analogue_year"]        = m["report_current_year"] - 10
        m["report_analogue_gap"]         = 10
        m["report_analogue_fisher_dist"] = m["report_analogue_mutual_info"] = np.nan
        m["report_analogue_entanglement"]= m["report_wormhole_strength"] = np.nan
        m["report_wormhole_df"]          = pd.DataFrame()

    print(f"  [✓] Primary analogue: {m['report_analogue_year']} "
          f"(gap={m['report_analogue_gap']} yrs, "
          f"strength={m['report_wormhole_strength']:.2f}x baseline)")

    # --- MODULE 4: ADI Volatility Range ---
    if range_summary_df is not None and "Regional_ADI_Mean" in range_summary_df.index:
        m["report_adi_vol_multiple"]  = float(range_summary_df.loc["Regional_ADI_Mean", "Volatility_Expansion_Multiple"])
        m["report_adi_recent_high"]   = float(range_summary_df.loc["Regional_ADI_Mean", "Recent_High"])
        m["report_adi_recent_low"]    = float(range_summary_df.loc["Regional_ADI_Mean", "Recent_Low"])
        yr_col = [c for c in range_summary_df.columns if "Yr_Avg" in c]
        m["report_adi_hist_avg_range"]= float(range_summary_df.loc["Regional_ADI_Mean", yr_col[0]]) if yr_col else np.nan
    else:
        m["report_adi_vol_multiple"] = m["report_adi_recent_high"] = \
        m["report_adi_recent_low"]   = m["report_adi_hist_avg_range"] = np.nan

    if range_summary_df is not None and "Spatial_Entropy_Gradient" in range_summary_df.index:
        m["report_gradient_vol_multiple"] = float(range_summary_df.loc["Spatial_Entropy_Gradient", "Volatility_Expansion_Multiple"])
    else:
        m["report_gradient_vol_multiple"] = np.nan

    print(f"  [✓] ADI Volatility Expansion: {m['report_adi_vol_multiple']:.2f}x  |  "
          f"Gradient Vol: {m['report_gradient_vol_multiple']:.2f}x")

    # --- MODULE 5: Thermal Asymmetry & Current State ---
    if monthly_adi_index_df is not None and "Thermal_Asymmetry_Multiple" in monthly_adi_index_df.columns:
        valid_asym = monthly_adi_index_df["Thermal_Asymmetry_Multiple"].dropna()
        m["report_thermal_asymmetry"]     = float(valid_asym.iloc[-1]) if not valid_asym.empty else np.nan
        m["report_current_spatial_state"] = str(monthly_adi_index_df["Spatial_Gradient_State"].iloc[-1]) \
            if "Spatial_Gradient_State" in monthly_adi_index_df.columns else "Unknown"
        m["report_current_adi"]           = float(monthly_adi_index_df["Regional_ADI_Mean"].iloc[-1]) \
            if "Regional_ADI_Mean" in monthly_adi_index_df.columns else np.nan
    else:
        m["report_thermal_asymmetry"]     = np.nan
        m["report_current_spatial_state"] = "Unknown"
        m["report_current_adi"]           = np.nan

    _ta_str = "N/A" if pd.isna(m["report_thermal_asymmetry"]) else f"{m['report_thermal_asymmetry']:.4f}"
    print(f"  [✓] Thermal Asymmetry: {_ta_str}  |  State: {m['report_current_spatial_state']}")

    # --- MODULE 6: System-Time Velocity ---
    if velocity_df is not None and "Velocity_Anomaly_Multiple" in velocity_df.columns:
        m["report_velocity_multiple"]   = float(velocity_df["Velocity_Anomaly_Multiple"].iloc[-1])
        m["report_current_velocity"]    = float(velocity_df["System_Velocity_Riemannian"].iloc[-1])
        m["report_hist_avg_velocity"]   = float(velocity_df["System_Velocity_Riemannian"].mean())
        m["report_velocity_transition"] = str(velocity_df["Transition_Period"].iloc[-1]) \
            if "Transition_Period" in velocity_df.columns else "N/A"
    else:
        m["report_velocity_multiple"] = m["report_current_velocity"] = \
        m["report_hist_avg_velocity"] = np.nan
        m["report_velocity_transition"] = "N/A"

    print(f"  [✓] System Velocity: {m['report_current_velocity']:.3f} "
          f"({m['report_velocity_multiple']:.2f}x historical avg) "
          f"— Transition: {m['report_velocity_transition']}")

    # --- MODULE 7: PMFG Topology ---
    m["report_im_partial"]  = float(im_partial)
    m["report_euc_partial"] = float(euc_partial)
    m["report_im_pearson"]  = float(im_pearson)
    m["report_euc_pearson"] = float(euc_pearson)

    # Load PMFG graphs (from dict or pkl files)
    if pmfg_graphs:
        G_mac_par = pmfg_graphs.get("macro_partial")
        G_mic_par = pmfg_graphs.get("micro_partial")
        G_mac_pea = pmfg_graphs.get("macro_pearson")
        G_mic_pea = pmfg_graphs.get("micro_pearson")
    else:
        import os
        G_mac_par = _load_pmfg(os.path.join(pmfg_pkl_dir, "pmfg_macro_partial.pkl"))
        G_mic_par = _load_pmfg(os.path.join(pmfg_pkl_dir, "pmfg_micro_partial.pkl"))
        G_mac_pea = _load_pmfg(os.path.join(pmfg_pkl_dir, "pmfg_macro_pearson.pkl"))
        G_mic_pea = _load_pmfg(os.path.join(pmfg_pkl_dir, "pmfg_micro_pearson.pkl"))

    m["report_partial_new_edges"],    m["report_partial_severed_edges"],    m["report_partial_jaccard"]    = _edge_topology(G_mac_par, G_mic_par)
    m["report_pearson_new_edges"],    m["report_pearson_severed_edges"],    m["report_pearson_jaccard"]    = _edge_topology(G_mac_pea, G_mic_pea)
    m["report_pmfg_node_count"] = G_mac_par.number_of_nodes() if G_mac_par else 0

    print(f"  [✓] PMFG (Partial): Jaccard={m['report_partial_jaccard']:.3f}, IM={m['report_im_partial']:.5f}")
    print(f"  [✓] PMFG (Pearson): Jaccard={m['report_pearson_jaccard']:.3f}, IM={m['report_im_pearson']:.5f}")

    # --- MODULE 8: State Census ---
    if census_comparison is not None:
        _dl_recent = float(census_comparison.loc["Dissipative Loss", "Recent_Period (Count)"])  \
            if "Dissipative Loss" in census_comparison.index else 0.0
        _dl_dev    = float(census_comparison.loc["Dissipative Loss", "Deviation_from_Norm"]) \
            if "Dissipative Loss" in census_comparison.index else 0.0
        m["report_census_df"] = census_comparison.copy()
    else:
        _dl_recent = _dl_dev = 0.0
        m["report_census_df"] = pd.DataFrame()

    m["report_census_dissipative_loss_recent"] = _dl_recent
    m["report_census_dissipative_deviation"]   = _dl_dev
    print(f"  [✓] State Census: Dissipative Loss recent={_dl_recent:.0f}, deviation={_dl_dev:+.1f}")

    print("\n[✓] ALL ATMOSPHERIC METRICS HARVESTED — Run render_tear_sheet() to generate the report.")
    return m


# ---------------------------------------------------------------------------
# Cell 45 — Tear Sheet Renderer
# ---------------------------------------------------------------------------

def render_tear_sheet(metrics: Dict) -> None:
    """
    Render the 4-section Atmospheric Disorder Index quantitative tear sheet.

    Exactly replicates Cell 45.  Reads all ``report_*`` scalars from the
    ``metrics`` dictionary produced by :func:`harvest_metrics`.

    Sections
    --------
    1. Model Validity — Systemic Amnesia, Von Neumann Entropy, System Velocity
    2. Causal Network Structure — PMFG topology (Jaccard, IM, edge rewiring)
    3. Entropy Dynamics — OHLC volatility range, Thermal Asymmetry, State Census
    4. Temporal Structural Analogue — ER=EPR wormhole table and classification
    Structural Assessment Summary — 8-metric scorecard with ✔/⚠/✖ flags.

    Parameters
    ----------
    metrics : dict
        Output of :func:`harvest_metrics`.
    """
    W = 82

    def _div(char="-"): return char * W
    def _hdr(text, char="█"): return text.center(W, char)
    def _sec(title): return f"\n{_div()}\n  {title}\n{_div()}"

    # Pull all metrics (with safe defaults matching Cell 45 pattern)
    g = metrics
    _yr     = g.get("report_current_year",    datetime.now().year)
    _locs   = g.get("report_location_labels", ["Unknown"])
    _am_pct = g.get("report_amnesia_pct",     np.nan)
    _mac_m  = g.get("report_macro_mem_depth", np.nan)
    _mic_m  = g.get("report_micro_mem_depth", np.nan)
    _vn     = g.get("report_vn_entropy",      np.nan)
    _he     = g.get("report_hist_entropy_mean",np.nan)
    _ed     = g.get("report_entropy_delta_pct",np.nan)
    _an_yr  = g.get("report_analogue_year",   _yr - 10)
    _an_gap = g.get("report_analogue_gap",    0)
    _an_fd  = g.get("report_analogue_fisher_dist", np.nan)
    _an_mi  = g.get("report_analogue_mutual_info",  np.nan)
    _an_sc  = g.get("report_analogue_entanglement", np.nan)
    _wh_str = g.get("report_wormhole_strength",     np.nan)
    _wh_df  = g.get("report_wormhole_df",           pd.DataFrame())
    _adi_vm = g.get("report_adi_vol_multiple",  np.nan)
    _adi_hi = g.get("report_adi_recent_high",   np.nan)
    _adi_lo = g.get("report_adi_recent_low",    np.nan)
    _adi_ha = g.get("report_adi_hist_avg_range",np.nan)
    _gr_vm  = g.get("report_gradient_vol_multiple", np.nan)
    _ta     = g.get("report_thermal_asymmetry",     np.nan)
    _sg_st  = g.get("report_current_spatial_state", "Unknown")
    _adi_nw = g.get("report_current_adi",           np.nan)
    _vel_m  = g.get("report_velocity_multiple",     np.nan)
    _vel_c  = g.get("report_current_velocity",      np.nan)
    _vel_ha = g.get("report_hist_avg_velocity",     np.nan)
    _vel_tr = g.get("report_velocity_transition",   "N/A")
    _im_par = g.get("report_im_partial",  np.nan)
    _im_pea = g.get("report_im_pearson",  np.nan)
    _jc_par = g.get("report_partial_jaccard", np.nan)
    _jc_pea = g.get("report_pearson_jaccard", np.nan)
    _nw_par = g.get("report_partial_new_edges",    0)
    _sv_par = g.get("report_partial_severed_edges", 0)
    _nw_pea = g.get("report_pearson_new_edges",    0)
    _sv_pea = g.get("report_pearson_severed_edges", 0)
    _n_nd   = g.get("report_pmfg_node_count", 0)
    _cen_lo = g.get("report_census_dissipative_loss_recent", 0)
    _cen_dv = g.get("report_census_dissipative_deviation",   0)
    _cen_df = g.get("report_census_df", pd.DataFrame())

    # Decision flags (thresholds match Cell 45 exactly)
    _am_crit = (not pd.isna(_am_pct)) and (_am_pct > 25.0)
    _en_rig  = (not pd.isna(_ed))     and (_ed < -10)
    _en_cha  = (not pd.isna(_ed))     and (_ed > 10)
    _wh_str_ = (not pd.isna(_wh_str)) and (_wh_str >= 1.5)
    _wh_mod  = (not pd.isna(_wh_str)) and (1.0 <= _wh_str < 1.5)
    _adi_el  = (not pd.isna(_adi_vm)) and (_adi_vm > 1.2)
    _mo_hi   = (not pd.isna(_ta))     and (_ta >= 1.05)
    _mo_lo   = (not pd.isna(_ta))     and (_ta < 0.95)
    _vel_fl  = (not pd.isna(_vel_m))  and (_vel_m > 1.5)
    _vel_st  = (not pd.isna(_vel_m))  and (_vel_m < 0.8)
    _im_bk_p = (not pd.isna(_im_par)) and (_im_par > 0.15)
    _im_bk_e = (not pd.isna(_im_pea)) and (_im_pea > 0.15)

    if _am_crit:
        _regime, _rflag = "NON-STATIONARY TRANSITION REGIME", "⚠ ELEVATED"
    elif _en_rig or _en_cha:
        _regime, _rflag = "ANOMALOUS INFORMATION STATE",      "⚡ MONITOR"
    else:
        _regime, _rflag = "STATIONARY BASELINE REGIME",       "✓ STABLE"

    # --- Header ---
    print("\n" + "█" * W)
    print(_hdr("  ATMOSPHERIC DISORDER INDEX — QUANTITATIVE REPORT  "))
    print(_hdr(f"  Study Region: {', '.join(_locs)}  |  Reference Year: {_yr}  "))
    print(_hdr(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  "))
    print("█" * W)

    # ── SECTION 1: Model Validity ──────────────────────────────────────────
    print(_sec("SECTION 1 ▸ MODEL VALIDITY — Structural Memory and Information State"))

    print("\n  ► Systemic Amnesia (Structural Memory Depth Decay)")
    _ms = (f"{_am_pct:.1f}% decay  "
           f"[Historical mean depth: {_mac_m:.2f} yrs → Current: {_mic_m:.2f} yrs]"
           if not pd.isna(_am_pct) else "N/A")
    print(f"    {_ms}")
    if _am_crit:
        print(f"    ✖  CRITICAL: System has lost {_mac_m - _mic_m:.2f} years of structural memory.")
        print("       Rolling climatological models are structurally unreliable.")
    else:
        print("    ✔  Memory intact. Historical baseline models retain structural validity.")

    print("\n  ► Von Neumann Entropy (System Information State)")
    if not pd.isna(_vn):
        print(f"    Current: {_vn:.4f} nats  |  10-Year Mean: {_he:.4f} nats  |  Δ: {_ed:+.2f}%")
    if _en_rig:
        print("    ⚠  RIGID STATE: Abnormally low entropy. Elevated probability of regime transition.")
    elif _en_cha:
        print("    ⚠  HIGH DISORDER: Historical pattern models are structurally unreliable.")
    elif not pd.isna(_vn):
        print("    ✔  Entropy within normal bounds.")

    print("\n  ► System-Time Velocity (Riemannian Rate of Structural Change)")
    if not pd.isna(_vel_c):
        print(f"    {_vel_c:.4f}  ({_vel_m:.2f}x historical average of {_vel_ha:.4f})")
        print(f"    Transition: {_vel_tr}")
        if _vel_fl:
            print(f"    ⚠  RAPID STRUCTURAL CHANGE: Topology mutating {_vel_m:.2f}x faster than baseline.")
        elif _vel_st:
            print("    ✔  STRUCTURAL STAGNATION: Mean-reverting models are well-suited.")
        else:
            print("    ✔  System velocity within normal operating bounds.")

    # ── SECTION 2: PMFG Topology ───────────────────────────────────────────
    print(_sec("SECTION 2 ▸ CAUSAL NETWORK STRUCTURE — PMFG Topology (Macro vs Micro)"))
    print(f"\n  Network: {_n_nd} nodes (meteorological variables across 4 observation nodes)\n")
    print("  ┌──────────────────────────────────────────────────────────────────────┐")
    print("  │  NETWORK TYPE         Jaccard Stability   Ipsen-Mikhailov Distance   │")
    print("  ├──────────────────────────────────────────────────────────────────────┤")

    def _fmtj(v):
        if pd.isna(v): return "N/A      "
        return f"{v:.4f}" + (" ← DISRUPTED" if v < 0.5 else " ← STABLE" if v > 0.75 else " ← SHIFTED")

    def _fmtim(v, broken):
        if pd.isna(v): return "N/A"
        return f"{v:.5f}" + (" ⚠ PHASE BREAK" if broken else " ✔ Normal")

    print(f"  │  Partial (Conditional) {_fmtj(_jc_par):<22} {_fmtim(_im_par, _im_bk_p):<22}│")
    print(f"  │  Pearson (General)     {_fmtj(_jc_pea):<22} {_fmtim(_im_pea, _im_bk_e):<22}│")
    print("  └──────────────────────────────────────────────────────────────────────┘")

    print("\n  Conditional (Partial) Network Edge Rewiring:")
    print(f"    [+] {_nw_par:>3} new conditional dependencies formed.")
    print(f"    [-] {_sv_par:>3} historical conditional dependencies severed.")
    print("\n  General (Pearson) Network Edge Rewiring:")
    print(f"    [+] {_nw_pea:>3} new general correlations formed.")
    print(f"    [-] {_sv_pea:>3} general correlations severed.")

    # ── SECTION 3: Entropy Dynamics ────────────────────────────────────────
    print(_sec("SECTION 3 ▸ ENTROPY DYNAMICS — Momentum and State Classification"))

    print("\n  ► ADI Seasonal Volatility Range (OHLC Analysis)")
    if not pd.isna(_adi_vm):
        print(f"    Regional ADI band: [{_adi_lo:.4f} → {_adi_hi:.4f}]  |  Seasonal Vol Expansion: {_adi_vm:.2f}x")
        if _adi_el:
            print(f"    ⚠  ADI range is {_adi_vm:.2f}x wider than the {_adi_ha:.4f} seasonal average.")
            print("       Standard deviation-based climatological bounds underestimate actual variability.")
        else:
            print("    ✔  ADI volatility range within historical seasonal norms.")
        if not pd.isna(_gr_vm):
            _gf = " ⚠ SUPPRESSED" if _gr_vm < 0.8 else (" ⚠ ELEVATED" if _gr_vm > 1.2 else " ✔ Normal")
            print(f"    Spatial Entropy Gradient Volatility: {_gr_vm:.2f}x seasonal norm{_gf}")
    else:
        print("    N/A")

    print("\n  ► Relativistic Thermal Asymmetry Multiple (6-Month Rolling)")
    print(f"    Current Spatial Gradient State: {_sg_st}  |  Regional ADI: {_adi_nw:.4f}"
          if not pd.isna(_adi_nw) else f"    Current Spatial Gradient State: {_sg_st}")
    if pd.isna(_ta):
        print("    Thermal Asymmetry: N/A — classification via Spatial Gradient State:")
        if   _sg_st == "Dissipative Loss":  print("    ✔  Dissipative Loss: Entropy contraction dominant.")
        elif _sg_st == "Dissipative Gain":  print("    ⚠  Dissipative Gain: Entropy expansion accelerating.")
        else:                               print("    ✔  Reversible dynamics dominant.")
    else:
        print(f"    Thermal Asymmetry Multiple: {_ta:.4f}")
        if   _mo_hi: print("    ⚠  Asymmetry ≥ 1.05: Entropy expansion is structurally dominant.")
        elif _mo_lo: print("    ✔  Asymmetry < 0.95: Entropy contraction dominant. Mean-reversion active.")
        else:        print("    →  Momentum near equilibrium (0.95–1.05). No strong directional bias.")

    print("\n  ► Entropy State Census (Recent vs Historical Frequency)")
    if not _cen_df.empty:
        print(f"    Dissipative Loss:  Recent={_cen_lo:.0f}  |  Deviation: {_cen_dv:+.1f} "
              f"({'⚠ ABNORMALLY HIGH' if _cen_dv > 1.5 else '✔ Normal'} occurrence rate)")
    else:
        print("    Census data unavailable.")

    # ── SECTION 4: Temporal Analogue ───────────────────────────────────────
    print(_sec("SECTION 4 ▸ TEMPORAL STRUCTURAL ANALOGUE — ER=EPR Wormhole Analysis"))
    print("\n  Top Temporal Analogues by Entanglement Score:")
    if not _wh_df.empty:
        print(f"  {'Year':>6}  {'Gap (yrs)':>10}  {'Fisher-Rao Dist':>16}  "
              f"{'Mutual Info':>12}  {'Score':>10}  {'Strength':>10}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*16}  {'-'*12}  {'-'*10}  {'-'*10}")
        for _, row in _wh_df.iterrows():
            strength = row.get("Wormhole_Strength_Multiple", np.nan)
            s_str  = f"{strength:.2f}x" if not pd.isna(strength) else "N/A"
            marker = " ◄ PRIMARY" if row["Historical_Analogue_Year"] == _an_yr else ""
            print(f"  {int(row['Historical_Analogue_Year']):>6}  "
                  f"{int(row['Chronological_Gap']):>10}  "
                  f"{row['Fisher_Geodesic_Distance']:>16.4f}  "
                  f"{row['Mutual_Information_Bound']:>12.4f}  "
                  f"{row['Entanglement_Score']:>10.6f}  "
                  f"{s_str:>10}{marker}")

    print(f"\n  Primary Structural Analogue: {_an_yr}  "
          f"(Chronological separation: {_an_gap} years)")
    if _wh_str_:
        print(f"  ✔  STRONG ENTANGLEMENT ({_wh_str:.2f}x baseline).")
        print(f"     Structural patterns from {_an_yr} provide the strongest historical reference.")
    elif _wh_mod:
        print(f"  →  MODERATE ENTANGLEMENT ({_wh_str:.2f}x baseline).")
        print(f"     Weight {_an_yr} patterns alongside standard climatological baselines.")
    else:
        print(f"  ✖  WEAK ENTANGLEMENT / SYSTEMIC AMNESIA ({_wh_str:.2f}x baseline).")
        print("     No strong analogue found. Increase uncertainty bounds substantially.")

    # ── Structural Assessment Summary ──────────────────────────────────────
    print("\n" + "=" * W)
    print(_hdr("  STRUCTURAL ASSESSMENT SUMMARY  ", "="))
    print("=" * W)

    def _flag(crit, warn, val_str):
        if crit: return f"  ✖  CRITICAL   {val_str}"
        if warn: return f"  ⚠  WARNING    {val_str}"
        return          f"  ✔  NORMAL     {val_str}"

    print(_flag(_am_crit, False,
                f"Systemic Amnesia:                  {_am_pct:.1f}% decay" if not pd.isna(_am_pct) else "N/A"))
    print(_flag(_en_rig or _en_cha, False,
                f"Von Neumann Entropy Deviation:     {_ed:+.1f}%" if not pd.isna(_ed) else "N/A"))
    print(_flag(_vel_fl, _vel_st,
                f"System-Time Velocity:              {_vel_m:.2f}x historical avg" if not pd.isna(_vel_m) else "N/A"))
    print(_flag(not _wh_str_ and not _wh_mod, False,
                f"Temporal Analogue Strength:        {_wh_str:.2f}x baseline  → Analogue: {_an_yr}" if not pd.isna(_wh_str) else "N/A"))
    print(_flag(_adi_el, False,
                f"ADI Seasonal Volatility Expansion: {_adi_vm:.2f}x seasonal norm" if not pd.isna(_adi_vm) else "N/A"))
    print(_flag(_mo_hi, False,
                f"Relativistic Thermal Asymmetry:    {_ta:.4f}" if not pd.isna(_ta) else
                f"Relativistic Thermal Asymmetry:    N/A  (State: {_sg_st})"))
    print(_flag(_im_bk_p, False,
                f"PMFG Partial IM Spectral Distance: {_im_par:.5f}" if not pd.isna(_im_par) else "N/A"))
    print(_flag(_im_bk_e, False,
                f"PMFG Pearson IM Spectral Distance: {_im_pea:.5f}" if not pd.isna(_im_pea) else "N/A"))

    print(f"\n  OVERALL REGIME CLASSIFICATION:  {_regime}  {_rflag}")
    print("\n" + "=" * W)
    print(_hdr(f"  END OF REPORT  |  {', '.join(_locs)}  |  Reference Year: {_yr}  ", "="))
    print("=" * W + "\n")
