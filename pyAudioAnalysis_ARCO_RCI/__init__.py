"""
pyAudioAnalysis ARCO-RCI Extension
====================================

ARCO (Rational-Arc Fingerprint / Arcogram) and RCI (Rational Coherence Index)
functionality for analyzing periodic and quasi-periodic patterns in 1D signals.

This module provides:
- Arcogram computation from audio, ECG, protein-derived numeric tracks
- RCI (Rational Coherence Index) metrics for periodicity analysis
- Fixed-length ARCO-print vectors for machine learning
- Optional GPU-accelerated computation via PyTorch
- Command-line tools for batch processing

Main modules:
- arc_rci: Core NumPy/SciPy implementation
- torch_arc_rci: PyTorch GPU-accelerated variant
- losses: RCI auxiliary loss for training
- cli: Command-line interface

For detailed documentation, see docs/ARCO_RCI.md
"""

__version__ = "0.1.0"
__author__ = "pyAudioAnalysis Contributors"

from .arc_rci import (
    generate_rational_arcs,
    sequence_to_signal,
    compute_power_spectrum,
    integrate_arc_band,
    arcogram_window,
    arcogram_signal,
    compute_rci_from_arcogram,
    arco_print_aggregate,
    pick_major_arcs,
    ARCOSpectrum,
)

__all__ = [
    "generate_rational_arcs",
    "sequence_to_signal",
    "compute_power_spectrum",
    "integrate_arc_band",
    "arcogram_window",
    "arcogram_signal",
    "compute_rci_from_arcogram",
    "arco_print_aggregate",
    "pick_major_arcs",
    "ARCOSpectrum",
]
