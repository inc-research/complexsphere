# -*- coding: utf-8 -*-
"""
complexsphere
========
Atmospheric Disorder Index (ADI) — Python Library
A faithful implementation of the watt_academic.ipynb computational
framework for multi-node atmospheric entropy analysis.

Module Map
----------
acquisition  Cells 01, 03, 05 — NASA POWER fetch + formatting
entropy      Cells 07, 09, 11 — Monthly Shannon entropy, ADI index, velocity
kinetics     Cells 13, 15, 17, 19 — State classification, census, Lorentz, asymmetry
volatility   Cell  21          — OHLC seasonal volatility range
network      Cells 23, 25, 27  — Cross-node matrices, partial corr, PMFG
topology     Cells 29, 31      — Topological shift, Ipsen-Mikhailov distance
geometry     Cell  33          — Riemannian distance, annual matrices, velocity
causality    Cell  35          — Granger causality (ADF + cross-corr + GC)
memory       Cells 37, 39, 41  — Systemic amnesia, ER=EPR wormholes, analogue report
report       Cells 43, 45      — Metric harvester + quantitative tear sheet

Quick-start
-----------
>>> from complexsphere.acquisition import SINGAPORE_NETWORK, load_from_csv
>>> from complexsphere.entropy      import compute_node_entropy, build_adi_index, compute_entropy_velocity
>>> from complexsphere.kinetics     import classify_states, compute_temporal_census
>>> from complexsphere.kinetics     import calculate_relativistic_kinetics, calculate_thermal_asymmetry
>>> from complexsphere.volatility   import compute_ohlc_volatility_range
>>> from complexsphere.network      import assemble_cross_node_network, calculate_partial_correlation, build_all_pmfgs
>>> from complexsphere.topology     import analyze_all_networks
>>> from complexsphere.geometry     import build_annual_matrices, compute_system_velocity
>>> from complexsphere.causality    import run_granger_causality
>>> from complexsphere.memory       import calculate_systemic_amnesia, identify_temporal_wormholes, generate_atmospheric_epr_report
>>> from complexsphere.report       import harvest_metrics, render_tear_sheet
"""

from .acquisition import (
    fetch_daily_data,
    fetch_climatology_data,
    format_daily_data,
    format_climatology_data,
    load_from_csv,
    SINGAPORE_NETWORK,
    DEFAULT_PARAMETERS,
    DEFAULT_COMMUNITY,
)

from .entropy import (
    compute_node_entropy,
    build_adi_index,
    compute_entropy_velocity,
)

from .kinetics import (
    classify_states,
    compute_temporal_census,
    calculate_relativistic_kinetics,
    calculate_thermal_asymmetry,
)

from .volatility import (
    compute_ohlc_volatility_range,
)

from .network import (
    assemble_cross_node_network,
    calculate_partial_correlation,
    calculate_pmfg,
    build_all_pmfgs,
)

from .topology import (
    analyze_topological_shift,
    calculate_ipsen_mikhailov_distance,
    analyze_all_networks,
)

from .geometry import (
    riemannian_distance,
    build_annual_matrices,
    compute_system_velocity,
)

from .causality import (
    run_granger_causality,
)

from .memory import (
    QuantumInformationMetrics,
    SystemState,
    MemoryLink,
    TemporalMemoryGraph,
    MemoryDetector,
    calculate_systemic_amnesia,
    identify_temporal_wormholes,
    generate_atmospheric_epr_report,
)

from .report import (
    harvest_metrics,
    render_tear_sheet,
)

__version__ = "1.0.0"
__all__ = [
    # acquisition
    "fetch_daily_data", "fetch_climatology_data",
    "format_daily_data", "format_climatology_data",
    "load_from_csv",
    "SINGAPORE_NETWORK", "DEFAULT_PARAMETERS", "DEFAULT_COMMUNITY",
    # entropy
    "compute_node_entropy", "build_adi_index", "compute_entropy_velocity",
    # kinetics
    "classify_states", "compute_temporal_census",
    "calculate_relativistic_kinetics", "calculate_thermal_asymmetry",
    # volatility
    "compute_ohlc_volatility_range",
    # network
    "assemble_cross_node_network", "calculate_partial_correlation",
    "calculate_pmfg", "build_all_pmfgs",
    # topology
    "analyze_topological_shift", "calculate_ipsen_mikhailov_distance",
    "analyze_all_networks",
    # geometry
    "riemannian_distance", "build_annual_matrices", "compute_system_velocity",
    # causality
    "run_granger_causality",
    # memory
    "QuantumInformationMetrics", "SystemState", "MemoryLink",
    "TemporalMemoryGraph", "MemoryDetector",
    "calculate_systemic_amnesia", "identify_temporal_wormholes",
    "generate_atmospheric_epr_report",
    # report
    "harvest_metrics", "render_tear_sheet",
]
