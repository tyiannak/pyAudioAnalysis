#!/usr/bin/env python
"""
ARCO-RCI Command Line Interface
================================

CLI for computing arcograms and RCI from audio files.

Usage examples:
    # Single file
    python -m pyAudioAnalysis_ARCO_RCI.cli --input audio.wav --save-arcogram arcogram.csv --save-plot arcogram.png

    # Directory
    python -m pyAudioAnalysis_ARCO_RCI.cli --input audio_folder/ --save-arcogram results/

    # Custom parameters
    python -m pyAudioAnalysis_ARCO_RCI.cli --input audio.wav --max-q 50 --window-size 500 --step 100
"""

import argparse
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional
import warnings

# Try to import pyAudioAnalysis for audio loading
try:
    from pyAudioAnalysis.audioBasicIO import read_audio_file, stereo_to_mono
    HAS_PYAUDIOANALYSIS = True
except ImportError:
    HAS_PYAUDIOANALYSIS = False
    warnings.warn("pyAudioAnalysis not found; will try scipy.io.wavfile")
    from scipy.io import wavfile

from .arc_rci import ARCOSpectrum, sequence_to_signal


def load_audio(file_path: str) -> tuple:
    """
    Load audio file and convert to mono if needed.

    Parameters
    ----------
    file_path : str
        Path to audio file

    Returns
    -------
    fs : int
        Sampling frequency
    signal : np.ndarray
        Audio signal (1D)
    """
    if HAS_PYAUDIOANALYSIS:
        fs, signal = read_audio_file(file_path)
        signal = stereo_to_mono(signal)
    else:
        # Fallback to scipy
        fs, signal = wavfile.read(file_path)
        if signal.ndim == 2:
            signal = np.mean(signal, axis=1)

    # Normalize to float
    if signal.dtype in [np.int16, np.int32]:
        signal = signal.astype(np.float64) / np.max(np.abs(signal))

    return fs, signal


def save_arcogram_csv(
    arcograms: np.ndarray,
    positions: np.ndarray,
    rational_arcs: List,
    output_path: str,
    fs: float
):
    """
    Save arcogram to CSV file.

    Parameters
    ----------
    arcograms : np.ndarray
        Arcogram matrix, shape (n_windows, M)
    positions : np.ndarray
        Window center positions
    rational_arcs : List[Fraction]
        List of rational arcs
    output_path : str
        Output CSV path
    fs : float
        Sampling frequency
    """
    # Create column names from rational arcs
    arc_labels = [f"{frac.numerator}/{frac.denominator}" for frac in rational_arcs]

    # Create DataFrame
    df = pd.DataFrame(arcograms, columns=arc_labels)
    df.insert(0, 'position_samples', positions)
    df.insert(1, 'time_seconds', positions / fs)

    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved arcogram to: {output_path}")


def save_rci_csv(
    rci_windows: np.ndarray,
    positions: np.ndarray,
    output_path: str,
    fs: float
):
    """
    Save RCI timeline to CSV.

    Parameters
    ----------
    rci_windows : np.ndarray
        RCI values per window
    positions : np.ndarray
        Window positions
    output_path : str
        Output path
    fs : float
        Sampling frequency
    """
    df = pd.DataFrame({
        'position_samples': positions,
        'time_seconds': positions / fs,
        'rci': rci_windows
    })

    df.to_csv(output_path, index=False)
    print(f"Saved RCI timeline to: {output_path}")


def plot_arcogram(
    arcograms: np.ndarray,
    positions: np.ndarray,
    rci_windows: np.ndarray,
    rational_arcs: List,
    output_path: str,
    fs: float,
    top_k: int = 10
):
    """
    Plot arcogram heatmap and RCI timeline.

    Parameters
    ----------
    arcograms : np.ndarray
        Arcogram matrix
    positions : np.ndarray
        Window positions
    rci_windows : np.ndarray
        RCI values
    rational_arcs : List[Fraction]
        Rational arcs
    output_path : str
        Output path
    fs : float
        Sampling frequency
    top_k : int
        Number of top arcs to show
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Convert positions to time
    times = positions / fs

    # Plot arcogram heatmap
    ax = axes[0]
    im = ax.imshow(
        arcograms.T,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        extent=[times.min(), times.max(), 0, len(rational_arcs)]
    )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rational Arc Index')
    ax.set_title('Arcogram Heatmap')
    plt.colorbar(im, ax=ax, label='Power')

    # Add some rational arc labels on y-axis
    step = max(1, len(rational_arcs) // 10)
    yticks = list(range(0, len(rational_arcs), step))
    yticklabels = [
        f"{rational_arcs[i].numerator}/{rational_arcs[i].denominator}"
        for i in yticks
    ]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    # Plot RCI timeline
    ax = axes[1]
    ax.plot(times, rci_windows, 'b-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RCI')
    ax.set_title('Rational Coherence Index Over Time')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    plt.close()


def print_top_arcs(
    arcograms: np.ndarray,
    rational_arcs: List,
    fs: float,
    top_k: int = 10
):
    """
    Print top-k rational arcs by power.

    Parameters
    ----------
    arcograms : np.ndarray
        Arcogram matrix
    rational_arcs : List[Fraction]
        Rational arcs
    fs : float
        Sampling frequency
    top_k : int
        Number of top arcs to show
    """
    # Average power across windows
    mean_power = np.mean(arcograms, axis=0)

    # Get top-k indices
    top_indices = np.argsort(mean_power)[::-1][:top_k]

    print(f"\nTop {top_k} Rational Arcs:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Fraction':<12} {'Freq (Hz)':<12} {'Power':<12}")
    print("-" * 70)

    nyquist = fs / 2.0
    for rank, idx in enumerate(top_indices, 1):
        frac = rational_arcs[idx]
        freq = float(frac) * nyquist
        power = mean_power[idx]

        frac_str = f"{frac.numerator}/{frac.denominator}"
        print(f"{rank:<6} {frac_str:<12} {freq:<12.2f} {power:<12.6f}")

    print("-" * 70)


def process_file(
    input_file: str,
    args: argparse.Namespace,
    arco: ARCOSpectrum
) -> dict:
    """
    Process a single audio file.

    Parameters
    ----------
    input_file : str
        Path to input file
    args : argparse.Namespace
        Command-line arguments
    arco : ARCOSpectrum
        ARCO spectrum computer

    Returns
    -------
    dict
        Results dictionary
    """
    print(f"\nProcessing: {input_file}")

    # Load audio
    try:
        fs, signal = load_audio(input_file)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return None

    # Override fs if specified
    if args.fs is not None:
        fs = args.fs

    print(f"  Sampling rate: {fs} Hz")
    print(f"  Duration: {len(signal)/fs:.2f} seconds")

    # Normalize signal
    signal = sequence_to_signal(signal, normalize=True)

    # Compute window parameters in samples
    window_size = int(args.window_size * fs / 1000)  # Convert ms to samples
    step = int(args.step * fs / 1000)

    print(f"  Window size: {window_size} samples ({args.window_size} ms)")
    print(f"  Step size: {step} samples ({args.step} ms)")

    # Compute arcogram
    arcograms, rci_windows, positions = arco.compute(
        signal, fs, window_size, step
    )

    print(f"  Computed {len(positions)} windows")
    print(f"  Mean RCI: {rci_windows.mean():.4f} ± {rci_windows.std():.4f}")
    print(f"  RCI range: [{rci_windows.min():.4f}, {rci_windows.max():.4f}]")

    # Print top arcs
    if args.top_k > 0:
        print_top_arcs(arcograms, arco.rational_arcs, fs, args.top_k)

    return {
        'arcograms': arcograms,
        'rci_windows': rci_windows,
        'positions': positions,
        'fs': fs,
        'file': input_file
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compute ARCO arcograms and RCI from audio files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/output
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input audio file or directory'
    )
    parser.add_argument(
        '--save-arcogram',
        help='Save arcogram to CSV/NPY (use .csv or .npy extension)'
    )
    parser.add_argument(
        '--save-rci',
        help='Save RCI timeline to CSV'
    )
    parser.add_argument(
        '--save-plot',
        help='Save visualization to PNG/PDF'
    )

    # Audio parameters
    parser.add_argument(
        '--fs',
        type=float,
        default=None,
        help='Override sampling rate (Hz). If not set, use file\'s native rate'
    )
    parser.add_argument(
        '--window-size',
        type=float,
        default=500.0,
        help='Window size in milliseconds'
    )
    parser.add_argument(
        '--step',
        type=float,
        default=100.0,
        help='Step size in milliseconds'
    )

    # ARCO parameters
    parser.add_argument(
        '--max-q',
        type=int,
        default=30,
        help='Maximum denominator for rational arcs'
    )
    parser.add_argument(
        '--base-sigma',
        type=float,
        default=0.5,
        help='Base width for Gaussian integration'
    )

    # Output options
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of top arcs to print (0 to disable)'
    )

    args = parser.parse_args()

    # Create ARCO spectrum computer
    print("Initializing ARCO spectrum computer...")
    print(f"  max_q: {args.max_q}")
    print(f"  base_sigma: {args.base_sigma}")

    arco = ARCOSpectrum(
        max_q=args.max_q,
        base_sigma=args.base_sigma
    )

    print(f"  Generated {len(arco.rational_arcs)} rational arcs")

    # Determine if input is file or directory
    if os.path.isfile(args.input):
        files = [args.input]
    elif os.path.isdir(args.input):
        # Find all audio files
        extensions = ['*.wav', '*.WAV', '*.mp3', '*.MP3', '*.aiff', '*.AIFF']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(args.input, ext)))
        files = sorted(files)

        if len(files) == 0:
            print(f"Error: No audio files found in {args.input}")
            return 1

        print(f"Found {len(files)} audio files")
    else:
        print(f"Error: Input path not found: {args.input}")
        return 1

    # Process files
    results = []
    for file_path in files:
        result = process_file(file_path, args, arco)
        if result is not None:
            results.append(result)

    if len(results) == 0:
        print("Error: No files processed successfully")
        return 1

    # Save outputs
    if len(results) == 1:
        # Single file
        result = results[0]

        if args.save_arcogram:
            ext = os.path.splitext(args.save_arcogram)[1].lower()
            if ext == '.csv':
                save_arcogram_csv(
                    result['arcograms'],
                    result['positions'],
                    arco.rational_arcs,
                    args.save_arcogram,
                    result['fs']
                )
            elif ext == '.npy':
                np.save(args.save_arcogram, result['arcograms'])
                print(f"Saved arcogram to: {args.save_arcogram}")
            else:
                print(f"Warning: Unknown extension {ext}, saving as .npy")
                np.save(args.save_arcogram, result['arcograms'])

        if args.save_rci:
            save_rci_csv(
                result['rci_windows'],
                result['positions'],
                args.save_rci,
                result['fs']
            )

        if args.save_plot:
            plot_arcogram(
                result['arcograms'],
                result['positions'],
                result['rci_windows'],
                arco.rational_arcs,
                args.save_plot,
                result['fs'],
                args.top_k
            )

    else:
        # Multiple files
        print(f"\nProcessed {len(results)} files")
        print(f"Overall mean RCI: {np.mean([r['rci_windows'].mean() for r in results]):.4f}")

        # If output paths specified, save each file separately
        if args.save_arcogram or args.save_rci or args.save_plot:
            if args.save_arcogram and os.path.isdir(args.save_arcogram):
                out_dir = args.save_arcogram
            elif args.save_rci and os.path.isdir(os.path.dirname(args.save_rci)):
                out_dir = os.path.dirname(args.save_rci)
            elif args.save_plot and os.path.isdir(os.path.dirname(args.save_plot)):
                out_dir = os.path.dirname(args.save_plot)
            else:
                out_dir = 'arco_output'
                os.makedirs(out_dir, exist_ok=True)

            print(f"\nSaving outputs to: {out_dir}")

            for result in results:
                basename = os.path.splitext(os.path.basename(result['file']))[0]

                if args.save_arcogram:
                    out_path = os.path.join(out_dir, f"{basename}_arcogram.csv")
                    save_arcogram_csv(
                        result['arcograms'],
                        result['positions'],
                        arco.rational_arcs,
                        out_path,
                        result['fs']
                    )

                if args.save_rci:
                    out_path = os.path.join(out_dir, f"{basename}_rci.csv")
                    save_rci_csv(
                        result['rci_windows'],
                        result['positions'],
                        out_path,
                        result['fs']
                    )

                if args.save_plot:
                    out_path = os.path.join(out_dir, f"{basename}_plot.png")
                    plot_arcogram(
                        result['arcograms'],
                        result['positions'],
                        result['rci_windows'],
                        arco.rational_arcs,
                        out_path,
                        result['fs'],
                        args.top_k
                    )

    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
