"""
ARCO-RCI Core Implementation
=============================

Core NumPy/SciPy implementation of ARCO (Rational-Arc Fingerprint) and
RCI (Rational Coherence Index) for 1D signal analysis.

Frequency Mapping Convention:
- Uses normalized fraction-of-Nyquist mapping
- For a rational r ∈ (0, 0.5), the center frequency is:
  f_center_hz = float(r) * (fs/2)
- This keeps interpretations consistent across different sampling rates
"""

import numpy as np
from scipy import signal as scipy_signal
from fractions import Fraction
from typing import List, Tuple, Optional, Union
import warnings


def generate_rational_arcs(max_q: int = 30) -> List[Fraction]:
    """
    Generate a sorted list of unique reduced rational fractions in (0, 0.5).

    These rationals represent frequencies as fractions of the Nyquist frequency.
    Rationals with r > 0.5 are mirrored to 1-r, and exactly 1/2 is excluded.

    Parameters
    ----------
    max_q : int
        Maximum denominator for rational fractions

    Returns
    -------
    List[Fraction]
        Sorted list of unique reduced Fraction objects in (0, 0.5)

    Examples
    --------
    >>> arcs = generate_rational_arcs(max_q=7)
    >>> print(arcs[:5])
    [Fraction(1, 7), Fraction(1, 6), Fraction(1, 5), Fraction(1, 4), Fraction(2, 7)]
    """
    rationals = set()

    for q in range(2, max_q + 1):
        for p in range(1, q):
            frac = Fraction(p, q)
            # Convert to float for comparison
            r = float(frac)

            # Mirror fractions > 0.5 and exclude exactly 0.5
            if r > 0.5:
                frac = Fraction(1) - frac
                r = float(frac)

            # Exclude 0.5 exactly
            if r != 0.5 and r > 0:
                rationals.add(frac)

    # Sort by float value
    return sorted(rationals, key=lambda x: float(x))


def sequence_to_signal(signal: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Prepare a numeric sequence for ARCO analysis.

    Parameters
    ----------
    signal : np.ndarray
        1D numeric array (float or int)
    normalize : bool
        If True, apply z-score normalization (zero mean, unit std)

    Returns
    -------
    np.ndarray
        Prepared 1D signal array

    Examples
    --------
    >>> signal = np.array([1, 2, 3, 4, 5])
    >>> normalized = sequence_to_signal(signal, normalize=True)
    >>> print(f"Mean: {normalized.mean():.2e}, Std: {normalized.std():.2f}")
    Mean: 0.00e+00, Std: 1.00
    """
    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {signal.shape}")

    if normalize:
        mean = signal.mean()
        std = signal.std()
        if std > 0:
            signal = (signal - mean) / std
        else:
            warnings.warn("Signal has zero std; skipping normalization")

    return signal


def compute_power_spectrum(
    signal: np.ndarray,
    fs: float,
    use_hann: bool = True,
    normalize_power: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the power spectrum of a signal using FFT.

    Parameters
    ----------
    signal : np.ndarray
        1D signal array
    fs : float
        Sampling frequency in Hz
    use_hann : bool
        If True, apply Hann window before FFT
    normalize_power : bool
        If True, normalize power spectrum to sum to 1

    Returns
    -------
    freqs : np.ndarray
        Frequency bins in Hz
    power : np.ndarray
        Power spectrum values

    Notes
    -----
    - Uses np.fft.rfft for real-valued FFT
    - Window energy correction is applied if windowing is used
    """
    signal = np.asarray(signal, dtype=np.float64)
    N = len(signal)

    # Apply window if requested
    if use_hann:
        window = np.hanning(N)
        windowed_signal = signal * window
        # Correct for window energy loss
        window_correction = np.sum(window**2) / N
    else:
        windowed_signal = signal
        window_correction = 1.0

    # Compute FFT
    fft_result = np.fft.rfft(windowed_signal)

    # Compute power spectrum
    power = np.abs(fft_result)**2 / (N * window_correction)

    # Get frequency bins
    freqs = np.fft.rfftfreq(N, d=1.0/fs)

    # Normalize if requested
    if normalize_power:
        power_sum = np.sum(power)
        if power_sum > 0:
            power = power / power_sum

    return freqs, power


def integrate_arc_band(
    power: np.ndarray,
    freqs: np.ndarray,
    frac: Fraction,
    fs: float,
    base_sigma: float = 0.5
) -> float:
    """
    Integrate power within a Gaussian-weighted frequency band centered on a rational.

    Uses normalized fraction-of-Nyquist mapping:
    f_center = float(frac) * (fs/2)

    Parameters
    ----------
    power : np.ndarray
        Power spectrum values
    freqs : np.ndarray
        Frequency bins in Hz
    frac : Fraction
        Rational fraction defining the center frequency
    fs : float
        Sampling frequency in Hz
    base_sigma : float
        Base width parameter for Gaussian weighting
        Actual sigma = base_sigma / q^2 where q = frac.denominator

    Returns
    -------
    float
        Integrated power in the arc band

    Notes
    -----
    The Gaussian width decreases with denominator squared to provide
    finer resolution for higher-order rationals.
    """
    # Map rational to frequency using fraction-of-Nyquist convention
    nyquist = fs / 2.0
    f_center = float(frac) * nyquist

    # Compute Gaussian width
    q = frac.denominator
    sigma = base_sigma / (q**2)

    # Compute Gaussian weights
    weights = np.exp(-0.5 * ((freqs - f_center) / sigma)**2)

    # Normalize weights to sum to 1
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        weights = weights / weight_sum

    # Integrate power with weights
    integrated_power = np.sum(power * weights)

    return float(integrated_power)


def arcogram_window(
    signal_window: Union[np.ndarray, List[np.ndarray]],
    fs: float,
    rational_arcs: List[Fraction],
    base_sigma: float = 0.5,
    use_hann: bool = True
) -> np.ndarray:
    """
    Compute arcogram for a single window.

    Supports both single-track and multi-track inputs.

    Parameters
    ----------
    signal_window : np.ndarray or list of np.ndarray
        Single window of signal data
        - If 1D array: single track, shape (N,)
        - If 2D array: multi-track, shape (n_tracks, N)
        - If list: list of 1D arrays, one per track
    fs : float
        Sampling frequency in Hz
    rational_arcs : List[Fraction]
        List of rational fractions to analyze
    base_sigma : float
        Base width for Gaussian integration
    use_hann : bool
        Whether to apply Hann window before FFT

    Returns
    -------
    np.ndarray
        Arcogram vector
        - Shape (M,) for single track, where M = len(rational_arcs)
        - Shape (n_tracks, M) for multi-track
    """
    # Handle multi-track input
    signal_window = np.asarray(signal_window)

    if signal_window.ndim == 1:
        # Single track
        freqs, power = compute_power_spectrum(signal_window, fs, use_hann=use_hann)
        arcogram = np.array([
            integrate_arc_band(power, freqs, frac, fs, base_sigma)
            for frac in rational_arcs
        ])
        return arcogram

    elif signal_window.ndim == 2:
        # Multi-track: shape (n_tracks, N)
        n_tracks = signal_window.shape[0]
        M = len(rational_arcs)
        arcograms = np.zeros((n_tracks, M))

        for i in range(n_tracks):
            freqs, power = compute_power_spectrum(
                signal_window[i], fs, use_hann=use_hann
            )
            arcograms[i] = np.array([
                integrate_arc_band(power, freqs, frac, fs, base_sigma)
                for frac in rational_arcs
            ])

        return arcograms

    else:
        raise ValueError(
            f"signal_window must be 1D or 2D, got shape {signal_window.shape}"
        )


def arcogram_signal(
    signal: np.ndarray,
    fs: float,
    window_size: int,
    step: int,
    rational_arcs: List[Fraction],
    base_sigma: float = 0.5,
    use_hann: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute arcograms across an entire signal using sliding windows.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
        - Shape (N,) for single track
        - Shape (n_tracks, N) for multi-track
    fs : float
        Sampling frequency in Hz
    window_size : int
        Window size in samples
    step : int
        Step size in samples (for sliding window)
    rational_arcs : List[Fraction]
        List of rational fractions to analyze
    base_sigma : float
        Base width for Gaussian integration
    use_hann : bool
        Whether to apply Hann window before FFT

    Returns
    -------
    arcograms : np.ndarray
        Arcogram matrix
        - Shape (n_windows, M) for single track
        - Shape (n_windows, n_tracks, M) for multi-track
        where M = len(rational_arcs)
    rci_windows : np.ndarray
        RCI value for each window, shape (n_windows,)
    positions : np.ndarray
        Center sample index for each window, shape (n_windows,)
    """
    signal = np.asarray(signal)

    # Determine if single or multi-track
    if signal.ndim == 1:
        is_multitrack = False
        signal_length = len(signal)
    elif signal.ndim == 2:
        is_multitrack = True
        n_tracks, signal_length = signal.shape
    else:
        raise ValueError(f"signal must be 1D or 2D, got shape {signal.shape}")

    # Compute window positions
    window_starts = list(range(0, signal_length - window_size + 1, step))
    n_windows = len(window_starts)
    M = len(rational_arcs)

    # Initialize output arrays
    if is_multitrack:
        arcograms = np.zeros((n_windows, n_tracks, M))
    else:
        arcograms = np.zeros((n_windows, M))

    rci_windows = np.zeros(n_windows)
    positions = np.zeros(n_windows, dtype=int)

    # Process each window
    for i, start in enumerate(window_starts):
        end = start + window_size
        positions[i] = start + window_size // 2

        if is_multitrack:
            window = signal[:, start:end]
        else:
            window = signal[start:end]

        # Compute arcogram for this window
        arc = arcogram_window(window, fs, rational_arcs, base_sigma, use_hann)
        arcograms[i] = arc

        # Compute RCI for this window
        # For multi-track, average RCI across tracks
        if is_multitrack:
            rci_per_track = np.array([
                compute_rci_from_arcogram(arc[j])
                for j in range(n_tracks)
            ])
            rci_windows[i] = np.mean(rci_per_track)
        else:
            rci_windows[i] = compute_rci_from_arcogram(arc)

    return arcograms, rci_windows, positions


def pick_major_arcs(rational_arcs: List[Fraction], max_den: int = 7) -> List[int]:
    """
    Select indices of "major" rational arcs with low denominators.

    These correspond to the strongest, most fundamental periodicities.

    Parameters
    ----------
    rational_arcs : List[Fraction]
        List of all rational arcs
    max_den : int
        Maximum denominator to consider as "major"

    Returns
    -------
    List[int]
        Indices of major arcs in rational_arcs

    Examples
    --------
    >>> arcs = generate_rational_arcs(max_q=20)
    >>> major_idx = pick_major_arcs(arcs, max_den=7)
    >>> major_fracs = [arcs[i] for i in major_idx]
    """
    indices = [
        i for i, frac in enumerate(rational_arcs)
        if frac.denominator <= max_den
    ]
    return indices


def compute_rci_from_arcogram(
    arcogram: np.ndarray,
    major_indices: Optional[List[int]] = None
) -> float:
    """
    Compute Rational Coherence Index (RCI) from an arcogram.

    RCI measures the fraction of power in low-denominator (major) rationals,
    indicating the strength of coherent periodic structure.

    Parameters
    ----------
    arcogram : np.ndarray
        Arcogram vector, shape (M,)
    major_indices : Optional[List[int]]
        Indices of major arcs. If None, uses arcs with denominator <= 7

    Returns
    -------
    float
        RCI value in [0, 1], where higher values indicate stronger periodicity

    Notes
    -----
    RCI = sum(arcogram[major_indices]) / sum(arcogram)
    Result is clipped to [0, 1] and handles zero-division cases.
    """
    arcogram = np.asarray(arcogram)

    if arcogram.ndim != 1:
        raise ValueError(f"arcogram must be 1D, got shape {arcogram.shape}")

    total_power = np.sum(arcogram)

    # Handle zero total power
    if total_power <= 0:
        return 0.0

    # Use all indices as "major" if not specified
    # (In practice, you should pass major_indices from pick_major_arcs)
    if major_indices is None:
        # Default: assume first ~1/3 are major (heuristic)
        # Better to explicitly pass major_indices
        major_indices = list(range(len(arcogram) // 3))

    major_power = np.sum(arcogram[major_indices])
    rci = major_power / total_power

    # Clip to [0, 1]
    rci = np.clip(rci, 0.0, 1.0)

    return float(rci)


def arco_print_aggregate(
    arcograms: np.ndarray,
    method: str = 'mean'
) -> np.ndarray:
    """
    Aggregate arcograms to produce a fixed-length feature vector for ML.

    Parameters
    ----------
    arcograms : np.ndarray
        Arcogram array
        - Shape (n_windows, M) for single track
        - Shape (n_windows, n_tracks, M) for multi-track
    method : str
        Aggregation method: 'mean', 'median', 'max', 'std', or 'all'
        If 'all', concatenates [mean, std, max, median]

    Returns
    -------
    np.ndarray
        Aggregated feature vector
        - Shape (M,) for single aggregation method
        - Shape (4*M,) if method='all' (for multi-track)
        - Shape (n_tracks*M,) or (n_tracks*4*M,) for multi-track

    Examples
    --------
    >>> arcograms = np.random.rand(100, 50)  # 100 windows, 50 arcs
    >>> feature = arco_print_aggregate(arcograms, method='mean')
    >>> print(feature.shape)
    (50,)
    """
    arcograms = np.asarray(arcograms)

    if arcograms.ndim == 2:
        # Single track: (n_windows, M)
        axis = 0
    elif arcograms.ndim == 3:
        # Multi-track: (n_windows, n_tracks, M)
        # Aggregate over windows (axis=0), keep tracks separate
        axis = 0
    else:
        raise ValueError(
            f"arcograms must be 2D or 3D, got shape {arcograms.shape}"
        )

    if method == 'mean':
        result = np.mean(arcograms, axis=axis)
    elif method == 'median':
        result = np.median(arcograms, axis=axis)
    elif method == 'max':
        result = np.max(arcograms, axis=axis)
    elif method == 'std':
        result = np.std(arcograms, axis=axis)
    elif method == 'all':
        mean_feat = np.mean(arcograms, axis=axis)
        std_feat = np.std(arcograms, axis=axis)
        max_feat = np.max(arcograms, axis=axis)
        median_feat = np.median(arcograms, axis=axis)

        # Concatenate all features
        if arcograms.ndim == 2:
            result = np.concatenate([mean_feat, std_feat, max_feat, median_feat])
        else:
            # For multi-track, flatten properly
            result = np.concatenate([
                mean_feat.flatten(),
                std_feat.flatten(),
                max_feat.flatten(),
                median_feat.flatten()
            ])
    else:
        raise ValueError(
            f"Unknown aggregation method '{method}'. "
            f"Choose from: mean, median, max, std, all"
        )

    return result


class ARCOSpectrum:
    """
    Convenience class for ARCO spectrum computation.

    Precomputes and caches rational arcs and provides simple interface
    for arcogram computation.

    Parameters
    ----------
    max_q : int
        Maximum denominator for rational arcs
    base_sigma : float
        Base width for Gaussian integration
    use_hann : bool
        Whether to apply Hann window
    major_max_den : int
        Maximum denominator for "major" arcs (used in RCI)

    Attributes
    ----------
    rational_arcs : List[Fraction]
        Cached list of rational arcs
    major_indices : List[int]
        Indices of major arcs

    Examples
    --------
    >>> arco = ARCOSpectrum(max_q=30, base_sigma=0.5)
    >>> signal = np.sin(2*np.pi*5*np.arange(1000)/100)  # 5 Hz sine
    >>> arcograms, rci, pos = arco.compute(signal, fs=100, window_size=200, step=50)
    >>> print(f"Mean RCI: {rci.mean():.3f}")
    """

    def __init__(
        self,
        max_q: int = 30,
        base_sigma: float = 0.5,
        use_hann: bool = True,
        major_max_den: int = 7
    ):
        self.max_q = max_q
        self.base_sigma = base_sigma
        self.use_hann = use_hann
        self.major_max_den = major_max_den

        # Precompute and cache rational arcs
        self.rational_arcs = generate_rational_arcs(max_q=max_q)
        self.major_indices = pick_major_arcs(
            self.rational_arcs, max_den=major_max_den
        )

    def compute_window(
        self,
        signal_window: np.ndarray,
        fs: float
    ) -> np.ndarray:
        """Compute arcogram for a single window."""
        return arcogram_window(
            signal_window, fs, self.rational_arcs,
            self.base_sigma, self.use_hann
        )

    def compute(
        self,
        signal: np.ndarray,
        fs: float,
        window_size: int,
        step: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute arcograms across entire signal."""
        return arcogram_signal(
            signal, fs, window_size, step,
            self.rational_arcs, self.base_sigma, self.use_hann
        )

    def compute_rci(self, arcogram: np.ndarray) -> float:
        """Compute RCI from arcogram using cached major indices."""
        return compute_rci_from_arcogram(arcogram, self.major_indices)

    def get_arc_frequencies(self, fs: float) -> np.ndarray:
        """
        Get center frequencies in Hz for all rational arcs.

        Parameters
        ----------
        fs : float
            Sampling frequency

        Returns
        -------
        np.ndarray
            Array of frequencies in Hz
        """
        nyquist = fs / 2.0
        return np.array([float(frac) * nyquist for frac in self.rational_arcs])

    def get_arc_labels(self) -> List[str]:
        """
        Get string labels for all rational arcs.

        Returns
        -------
        List[str]
            List of labels like "1/3", "2/5", etc.
        """
        return [f"{frac.numerator}/{frac.denominator}" for frac in self.rational_arcs]
