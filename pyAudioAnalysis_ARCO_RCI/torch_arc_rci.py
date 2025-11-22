"""
ARCO-RCI PyTorch GPU Implementation
====================================

GPU-accelerated arcogram computation using PyTorch for efficient
batch processing and integration into deep learning training loops.

This module provides the same API as arc_rci.py but operates on
torch.Tensors and supports GPU acceleration.
"""

import torch
import torch.nn as nn
from fractions import Fraction
from typing import List, Tuple, Optional, Dict
import warnings

# Import CPU version for rational arc generation
from .arc_rci import generate_rational_arcs, pick_major_arcs


class TorchARCOSpectrum(nn.Module):
    """
    PyTorch GPU-accelerated ARCO spectrum computation.

    Designed for efficient batch processing during training.
    Precomputes and caches frequency grids and Gaussian weights.

    Parameters
    ----------
    max_q : int
        Maximum denominator for rational arcs
    base_sigma : float
        Base width for Gaussian integration
    fs : float
        Sampling frequency in Hz (used for caching)
    window_size : int
        Expected window size for precomputing weights
    device : str or torch.device
        Device to use ('cpu', 'cuda', or torch.device object)
    use_hann : bool
        Whether to apply Hann window
    major_max_den : int
        Maximum denominator for major arcs

    Attributes
    ----------
    rational_arcs : List[Fraction]
        List of rational fractions
    major_indices : List[int]
        Indices of major arcs
    weights_cache : Dict
        Cache for precomputed Gaussian weights

    Examples
    --------
    >>> device = 'cuda' if torch.cuda.is_available() else 'cpu'
    >>> arco = TorchARCOSpectrum(max_q=30, fs=16000, window_size=512, device=device)
    >>> batch_signals = torch.randn(16, 512).to(device)  # Batch of 16 signals
    >>> arcograms = arco.compute_batch(batch_signals)
    >>> print(arcograms.shape)  # (16, M) where M is number of arcs
    """

    def __init__(
        self,
        max_q: int = 30,
        base_sigma: float = 0.5,
        fs: float = 16000,
        window_size: int = 512,
        device: str = 'cpu',
        use_hann: bool = True,
        major_max_den: int = 7
    ):
        super().__init__()

        self.max_q = max_q
        self.base_sigma = base_sigma
        self.fs = fs
        self.window_size = window_size
        self.device = torch.device(device)
        self.use_hann = use_hann
        self.major_max_den = major_max_den

        # Generate rational arcs (CPU operation)
        self.rational_arcs = generate_rational_arcs(max_q=max_q)
        self.major_indices = pick_major_arcs(
            self.rational_arcs, max_den=major_max_den
        )

        # Cache for Gaussian weights
        self.weights_cache = {}

        # Precompute Hann window if needed
        if use_hann:
            hann = torch.hann_window(window_size, device=self.device)
            # Compute window energy correction
            self.window_correction = torch.sum(hann**2) / window_size
            self.register_buffer('hann_window', hann)
        else:
            self.window_correction = 1.0
            self.register_buffer('hann_window', None)

        # Precompute weights for default window size
        self._precompute_weights(window_size, fs)

    def _precompute_weights(self, N: int, fs: float):
        """
        Precompute and cache Gaussian weights for a given window size and fs.

        Parameters
        ----------
        N : int
            Window size in samples
        fs : float
            Sampling frequency
        """
        cache_key = (N, fs)
        if cache_key in self.weights_cache:
            return

        # Compute frequency bins
        freqs = torch.fft.rfftfreq(N, d=1.0/fs).to(self.device)
        nyquist = fs / 2.0

        # Compute Gaussian weights for each rational arc
        M = len(self.rational_arcs)
        weights = torch.zeros(M, len(freqs), device=self.device)

        for i, frac in enumerate(self.rational_arcs):
            f_center = float(frac) * nyquist
            q = frac.denominator
            sigma = self.base_sigma / (q**2)

            # Gaussian weights
            w = torch.exp(-0.5 * ((freqs - f_center) / sigma)**2)

            # Normalize
            w_sum = torch.sum(w)
            if w_sum > 0:
                w = w / w_sum

            weights[i] = w

        # Cache weights
        self.weights_cache[cache_key] = weights

    def _get_weights(self, N: int, fs: float) -> torch.Tensor:
        """
        Get cached weights or compute if not available.

        Parameters
        ----------
        N : int
            Window size
        fs : float
            Sampling frequency

        Returns
        -------
        torch.Tensor
            Gaussian weights, shape (M, n_freqs)
        """
        cache_key = (N, fs)
        if cache_key not in self.weights_cache:
            self._precompute_weights(N, fs)
        return self.weights_cache[cache_key]

    def compute_power_spectrum_batch(
        self,
        signals: torch.Tensor,
        use_hann: Optional[bool] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute power spectra for a batch of signals.

        Parameters
        ----------
        signals : torch.Tensor
            Batch of signals, shape (B, N)
        use_hann : Optional[bool]
            Override use_hann setting. If None, uses self.use_hann

        Returns
        -------
        freqs : torch.Tensor
            Frequency bins, shape (n_freqs,)
        power : torch.Tensor
            Power spectra, shape (B, n_freqs)
        """
        B, N = signals.shape
        use_hann = self.use_hann if use_hann is None else use_hann

        # Apply window if needed
        if use_hann and self.hann_window is not None and len(self.hann_window) == N:
            windowed = signals * self.hann_window
            correction = self.window_correction
        elif use_hann:
            # Create window on-the-fly if size doesn't match
            hann = torch.hann_window(N, device=self.device)
            correction = torch.sum(hann**2) / N
            windowed = signals * hann
        else:
            windowed = signals
            correction = 1.0

        # Compute FFT
        fft_result = torch.fft.rfft(windowed, dim=1)

        # Compute power
        power = torch.abs(fft_result)**2 / (N * correction)

        # Normalize each spectrum to sum to 1
        power_sum = torch.sum(power, dim=1, keepdim=True)
        power = power / (power_sum + 1e-10)  # Add epsilon to avoid division by zero

        # Frequency bins
        freqs = torch.fft.rfftfreq(N, d=1.0/self.fs).to(self.device)

        return freqs, power

    def compute_batch(
        self,
        signals: torch.Tensor,
        fs: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute arcograms for a batch of signal windows.

        Parameters
        ----------
        signals : torch.Tensor
            Batch of signal windows, shape (B, N)
        fs : Optional[float]
            Sampling frequency. If None, uses self.fs

        Returns
        -------
        torch.Tensor
            Arcograms, shape (B, M) where M = len(rational_arcs)
        """
        B, N = signals.shape
        fs = self.fs if fs is None else fs

        # Get weights for this window size
        weights = self._get_weights(N, fs)  # Shape: (M, n_freqs)
        M = weights.shape[0]

        # Compute power spectra
        freqs, power = self.compute_power_spectrum_batch(signals)
        # power shape: (B, n_freqs)

        # Compute arcograms via matrix multiplication
        # power: (B, n_freqs), weights.T: (n_freqs, M)
        # Result: (B, M)
        arcograms = torch.matmul(power, weights.t())

        return arcograms

    def compute_rci_batch(
        self,
        arcograms: torch.Tensor,
        major_indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Compute RCI for a batch of arcograms.

        Parameters
        ----------
        arcograms : torch.Tensor
            Batch of arcograms, shape (B, M)
        major_indices : Optional[List[int]]
            Indices of major arcs. If None, uses self.major_indices

        Returns
        -------
        torch.Tensor
            RCI values, shape (B,)
        """
        if major_indices is None:
            major_indices = self.major_indices

        # Extract major arc powers
        major_power = torch.sum(arcograms[:, major_indices], dim=1)

        # Total power
        total_power = torch.sum(arcograms, dim=1)

        # Compute RCI
        rci = major_power / (total_power + 1e-10)  # Add epsilon

        # Clip to [0, 1]
        rci = torch.clamp(rci, 0.0, 1.0)

        return rci

    def forward(self, signals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute arcograms and RCI for a batch.

        Parameters
        ----------
        signals : torch.Tensor
            Batch of signals, shape (B, N)

        Returns
        -------
        arcograms : torch.Tensor
            Shape (B, M)
        rci : torch.Tensor
            Shape (B,)
        """
        arcograms = self.compute_batch(signals)
        rci = self.compute_rci_batch(arcograms)
        return arcograms, rci

    def get_arc_frequencies(self, fs: Optional[float] = None) -> torch.Tensor:
        """
        Get center frequencies for all rational arcs.

        Parameters
        ----------
        fs : Optional[float]
            Sampling frequency. If None, uses self.fs

        Returns
        -------
        torch.Tensor
            Frequencies in Hz, shape (M,)
        """
        fs = self.fs if fs is None else fs
        nyquist = fs / 2.0
        freqs = torch.tensor(
            [float(frac) * nyquist for frac in self.rational_arcs],
            device=self.device
        )
        return freqs

    def get_arc_labels(self) -> List[str]:
        """
        Get string labels for all rational arcs.

        Returns
        -------
        List[str]
            Labels like "1/3", "2/5", etc.
        """
        return [f"{frac.numerator}/{frac.denominator}" for frac in self.rational_arcs]


def compute_arcogram_similarity(
    arcogram1: torch.Tensor,
    arcogram2: torch.Tensor,
    method: str = 'cosine'
) -> torch.Tensor:
    """
    Compute similarity between two batches of arcograms.

    Parameters
    ----------
    arcogram1 : torch.Tensor
        First batch, shape (B, M)
    arcogram2 : torch.Tensor
        Second batch, shape (B, M)
    method : str
        Similarity metric: 'cosine' or 'correlation'

    Returns
    -------
    torch.Tensor
        Similarity values, shape (B,)
        Range: [-1, 1] for correlation, [0, 1] for cosine
    """
    if method == 'cosine':
        # Cosine similarity
        norm1 = torch.norm(arcogram1, dim=1, keepdim=True)
        norm2 = torch.norm(arcogram2, dim=1, keepdim=True)

        # Avoid division by zero
        norm1 = torch.clamp(norm1, min=1e-10)
        norm2 = torch.clamp(norm2, min=1e-10)

        normalized1 = arcogram1 / norm1
        normalized2 = arcogram2 / norm2

        similarity = torch.sum(normalized1 * normalized2, dim=1)

    elif method == 'correlation':
        # Pearson correlation
        mean1 = torch.mean(arcogram1, dim=1, keepdim=True)
        mean2 = torch.mean(arcogram2, dim=1, keepdim=True)

        centered1 = arcogram1 - mean1
        centered2 = arcogram2 - mean2

        numerator = torch.sum(centered1 * centered2, dim=1)
        denominator = torch.sqrt(
            torch.sum(centered1**2, dim=1) * torch.sum(centered2**2, dim=1)
        )

        similarity = numerator / (denominator + 1e-10)

    else:
        raise ValueError(f"Unknown similarity method '{method}'")

    return similarity


def batch_sliding_window(
    signal: torch.Tensor,
    window_size: int,
    step: int
) -> torch.Tensor:
    """
    Extract sliding windows from a batch of signals efficiently.

    Parameters
    ----------
    signal : torch.Tensor
        Batch of signals, shape (B, T)
    window_size : int
        Window size in samples
    step : int
        Step size in samples

    Returns
    -------
    torch.Tensor
        Windows, shape (B, n_windows, window_size)
    """
    B, T = signal.shape

    # Use unfold to extract sliding windows
    # unfold(dimension, size, step)
    windows = signal.unfold(1, window_size, step)
    # Result shape: (B, n_windows, window_size)

    return windows
