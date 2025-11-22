# ARCO-RCI Documentation

## Overview

ARCO (Rational-Arc Fingerprint / Arcogram) and RCI (Rational Coherence Index) are novel signal analysis techniques for detecting and quantifying periodic and quasi-periodic patterns in 1D signals.

This extension to pyAudioAnalysis provides:
- **Arcogram computation** from audio, ECG, protein-derived numeric tracks
- **RCI metrics** for quantifying periodicity strength
- **Fixed-length ARCO-print vectors** for machine learning
- **GPU-accelerated computation** via PyTorch
- **Command-line tools** for batch processing

## Table of Contents

1. [Concepts](#concepts)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Command-Line Usage](#command-line-usage)
6. [GPU Acceleration](#gpu-acceleration)
7. [Training Integration](#training-integration)
8. [Parameter Tuning](#parameter-tuning)
9. [Applications](#applications)
10. [Examples](#examples)

---

## Concepts

### Rational Arcs

Rational arcs are frequency bands centered at rational fractions of the Nyquist frequency. For a rational `r = a/q` where `a` and `q` are coprime integers:

- **Center frequency**: `f_center = r × (fs/2)` where `fs` is the sampling rate
- **Bandwidth**: Gaussian-weighted with `σ = base_sigma / q²`
- **Coverage**: All unique rationals in (0, 0.5) up to denominator `max_q`

**Why rationals?** Periodic signals have harmonic structure at rational multiples of fundamental frequencies. By analyzing power at these rational frequencies, we can detect and characterize periodicity.

### Arcogram

An **arcogram** is a power spectrum projected onto rational frequency bands:

```
Arcogram[i] = ∫ PowerSpectrum(f) × Gaussian(f; center=r_i, width=σ_i) df
```

where `r_i` is the i-th rational arc.

**Shape**: For `M` rational arcs, arcogram is a vector of length `M`.

**Interpretation**:
- High values at low-denominator rationals (1/3, 1/4, 1/5) → strong fundamental periodicity
- High values at high-denominator rationals (13/27, 17/31) → complex quasi-periodic structure
- Flat arcogram → noise or aperiodic signal

### RCI (Rational Coherence Index)

RCI measures the **coherence** of periodic structure:

```
RCI = (sum of power in major arcs) / (total arcogram power)
```

where **major arcs** are low-denominator rationals (typically denominator ≤ 7).

**Range**: [0, 1]
- **RCI ≈ 1**: Strong, coherent periodicity (e.g., pure sine wave)
- **RCI ≈ 0**: Weak or no periodicity (e.g., white noise)
- **RCI ≈ 0.5**: Moderate or complex periodicity

### Frequency Mapping Convention

This implementation uses **normalized fraction-of-Nyquist mapping**:

```python
nyquist = fs / 2
f_center_hz = float(rational) * nyquist
```

**Example**:
- `fs = 1000 Hz` → Nyquist = 500 Hz
- Rational 1/5 → f_center = 0.2 × 500 = 100 Hz
- Rational 1/3 → f_center = 0.333 × 500 = 166.7 Hz

This keeps interpretations consistent across different sampling rates.

---

## Installation

### Requirements

```bash
# Core dependencies (already in pyAudioAnalysis)
numpy>=1.20.3
scipy>=1.6.3
matplotlib>=3.4.2
pandas>=1.2.4

# Optional: PyTorch for GPU acceleration
torch>=1.9.0  # Install from pytorch.org with CUDA if desired
```

### Install pyAudioAnalysis with ARCO-RCI

```bash
# Clone repository
git clone https://github.com/tyiannak/pyAudioAnalysis.git
cd pyAudioAnalysis

# Install with dependencies
pip install -e .

# Verify installation
python -c "from pyAudioAnalysis_ARCO_RCI import ARCOSpectrum; print('Success!')"
```

---

## Quick Start

### Python API

```python
from pyAudioAnalysis.audioBasicIO import read_audio_file, stereo_to_mono
from pyAudioAnalysis_ARCO_RCI.arc_rci import ARCOSpectrum

# Load audio
fs, signal = read_audio_file('audio.wav')
signal = stereo_to_mono(signal)

# Create ARCO analyzer
arco = ARCOSpectrum(max_q=30, base_sigma=0.5)

# Compute arcograms with sliding windows
window_size = int(0.5 * fs)  # 500 ms
step = int(0.1 * fs)          # 100 ms

arcograms, rci_windows, positions = arco.compute(
    signal, fs, window_size, step
)

# Results
print(f"Computed {len(arcograms)} windows")
print(f"Mean RCI: {rci_windows.mean():.3f}")
print(f"RCI range: [{rci_windows.min():.3f}, {rci_windows.max():.3f}]")

# Extract ML-ready feature
from pyAudioAnalysis_ARCO_RCI.arc_rci import arco_print_aggregate
feature = arco_print_aggregate(arcograms, method='mean')
print(f"Feature shape: {feature.shape}")  # (M,) where M = number of arcs
```

### Command-Line

```bash
# Analyze single file
python -m pyAudioAnalysis_ARCO_RCI.cli \
    --input audio.wav \
    --save-arcogram arcogram.csv \
    --save-rci rci.csv \
    --save-plot arcogram.png \
    --window-size 500 \
    --step 100

# Analyze directory
python -m pyAudioAnalysis_ARCO_RCI.cli \
    --input audio_folder/ \
    --save-arcogram results/ \
    --max-q 50
```

---

## API Reference

### Core Functions

#### `generate_rational_arcs(max_q: int = 30) -> List[Fraction]`

Generate sorted list of unique rational fractions in (0, 0.5).

**Parameters**:
- `max_q`: Maximum denominator

**Returns**: List of `fractions.Fraction` objects

**Example**:
```python
from pyAudioAnalysis_ARCO_RCI.arc_rci import generate_rational_arcs

arcs = generate_rational_arcs(max_q=10)
print(f"Generated {len(arcs)} arcs")
print(arcs[:5])  # First 5: [1/10, 1/9, 1/8, 1/7, 1/6]
```

---

#### `arcogram_window(signal_window, fs, rational_arcs, base_sigma=0.5) -> np.ndarray`

Compute arcogram for a single window.

**Parameters**:
- `signal_window`: 1D array (single track) or 2D array (multi-track)
- `fs`: Sampling frequency (Hz)
- `rational_arcs`: List of rational fractions
- `base_sigma`: Base Gaussian width

**Returns**: Arcogram vector (M,) or matrix (n_tracks, M)

**Example**:
```python
import numpy as np
from pyAudioAnalysis_ARCO_RCI.arc_rci import arcogram_window, generate_rational_arcs

# Generate test signal
fs = 1000
t = np.arange(0, 1, 1/fs)
signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine

# Compute arcogram
arcs = generate_rational_arcs(max_q=30)
arcogram = arcogram_window(signal, fs, arcs)

print(f"Arcogram shape: {arcogram.shape}")
print(f"Peak arc index: {np.argmax(arcogram)}")
```

---

#### `arcogram_signal(signal, fs, window_size, step, rational_arcs, base_sigma=0.5) -> Tuple`

Compute arcograms across entire signal using sliding windows.

**Parameters**:
- `signal`: Input signal (1D or 2D for multi-track)
- `fs`: Sampling frequency
- `window_size`: Window size in samples
- `step`: Step size in samples
- `rational_arcs`: List of rationals
- `base_sigma`: Gaussian width parameter

**Returns**: Tuple of `(arcograms, rci_windows, positions)`
- `arcograms`: Shape (n_windows, M) or (n_windows, n_tracks, M)
- `rci_windows`: Shape (n_windows,)
- `positions`: Shape (n_windows,) - center sample index

---

#### `compute_rci_from_arcogram(arcogram, major_indices=None) -> float`

Compute RCI from arcogram.

**Parameters**:
- `arcogram`: 1D arcogram vector
- `major_indices`: Indices of major arcs (default: arcs with denominator ≤ 7)

**Returns**: RCI value in [0, 1]

**Example**:
```python
from pyAudioAnalysis_ARCO_RCI.arc_rci import (
    compute_rci_from_arcogram, pick_major_arcs
)

# Compute RCI with custom major arcs
major_idx = pick_major_arcs(arcs, max_den=5)
rci = compute_rci_from_arcogram(arcogram, major_idx)
```

---

#### `arco_print_aggregate(arcograms, method='mean') -> np.ndarray`

Aggregate arcograms into fixed-length feature vector.

**Parameters**:
- `arcograms`: Arcogram matrix (n_windows, M) or (n_windows, n_tracks, M)
- `method`: Aggregation method
  - `'mean'`: Average across windows
  - `'median'`: Median across windows
  - `'max'`: Maximum across windows
  - `'std'`: Standard deviation across windows
  - `'all'`: Concatenate [mean, std, max, median]

**Returns**: Feature vector

---

### ARCOSpectrum Class

Convenience class for ARCO analysis with caching.

```python
class ARCOSpectrum:
    def __init__(
        self,
        max_q: int = 30,
        base_sigma: float = 0.5,
        use_hann: bool = True,
        major_max_den: int = 7
    )
```

**Methods**:
- `compute_window(signal_window, fs)`: Compute arcogram for one window
- `compute(signal, fs, window_size, step)`: Compute arcograms for entire signal
- `compute_rci(arcogram)`: Compute RCI using cached major indices
- `get_arc_frequencies(fs)`: Get center frequencies in Hz
- `get_arc_labels()`: Get string labels ("1/3", "2/5", etc.)

**Example**:
```python
arco = ARCOSpectrum(max_q=30)
arcograms, rci, positions = arco.compute(signal, fs, 512, 128)

# Get frequency labels
freqs = arco.get_arc_frequencies(fs)
labels = arco.get_arc_labels()

print(f"Arc 1/3 is at {freqs[arco.rational_arcs.index(Fraction(1,3))]} Hz")
```

---

## Command-Line Usage

### Basic Usage

```bash
python -m pyAudioAnalysis_ARCO_RCI.cli --input FILE [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input, -i` | Input audio file or directory | Required |
| `--save-arcogram` | Save arcogram (.csv or .npy) | None |
| `--save-rci` | Save RCI timeline (.csv) | None |
| `--save-plot` | Save visualization (.png or .pdf) | None |
| `--fs` | Override sampling rate (Hz) | Auto-detect |
| `--window-size` | Window size (ms) | 500 |
| `--step` | Step size (ms) | 100 |
| `--max-q` | Maximum denominator | 30 |
| `--base-sigma` | Gaussian width parameter | 0.5 |
| `--top-k` | Number of top arcs to print | 10 |

### Examples

**Analyze with visualization**:
```bash
python -m pyAudioAnalysis_ARCO_RCI.cli \
    --input speech.wav \
    --save-plot speech_arco.png \
    --window-size 250 \
    --step 50
```

**Batch process directory**:
```bash
python -m pyAudioAnalysis_ARCO_RCI.cli \
    --input audio_dataset/ \
    --save-arcogram output/ \
    --save-rci output/ \
    --max-q 50
```

**Extract features for ML**:
```bash
python -m pyAudioAnalysis_ARCO_RCI.cli \
    --input training_data/ \
    --save-arcogram features.csv
```

---

## GPU Acceleration

For large-scale processing or training integration, use the PyTorch GPU variant.

### Installation

```bash
# Install PyTorch with CUDA (see pytorch.org for your system)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Usage

```python
import torch
from pyAudioAnalysis_ARCO_RCI.torch_arc_rci import TorchARCOSpectrum

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
arco = TorchARCOSpectrum(
    max_q=30,
    fs=16000,
    window_size=512,
    device=device
)

# Process batch of signals
batch_signals = torch.randn(32, 512).to(device)  # Batch of 32 windows
arcograms = arco.compute_batch(batch_signals)

# arcograms shape: (32, M) where M = number of arcs
print(arcograms.shape)

# Compute RCI
rci = arco.compute_rci_batch(arcograms)
print(f"Batch RCI: {rci}")
```

### Batch Processing

```python
# Process long signal in batches
from pyAudioAnalysis_ARCO_RCI.torch_arc_rci import batch_sliding_window

signal_tensor = torch.from_numpy(signal).float().unsqueeze(0).to(device)
# Shape: (1, T)

windows = batch_sliding_window(signal_tensor, window_size=512, step=128)
# Shape: (1, n_windows, 512)

# Flatten batch dimension
windows_flat = windows.reshape(-1, 512)

# Compute arcograms
arcograms = arco.compute_batch(windows_flat)
```

---

## Training Integration

Use RCI-based auxiliary losses to encourage periodic structure in generated signals.

### RCIAuxLoss

Encourages similarity between predicted and target arcograms across channels.

```python
from pyAudioAnalysis_ARCO_RCI.losses import RCIAuxLoss
import torch.nn as nn

# Setup loss
rci_loss = RCIAuxLoss(
    weight=0.05,           # Loss weight
    fs=16000,              # Sampling rate
    max_q=30,
    device='cuda',
    window_size=512,
    similarity_metric='correlation',  # or 'cosine'
    compare_with_target=True
)

# In training loop
optimizer.zero_grad()

# Forward pass
pred = model(input)  # Shape: (B, n_channels, T)
target = batch['target']  # Shape: (B, n_channels, T)

# Main reconstruction loss
main_loss = nn.L1Loss()(pred, target)

# RCI auxiliary loss
aux_loss = rci_loss(pred, target)

# Total loss
total_loss = main_loss + aux_loss

total_loss.backward()
optimizer.step()
```

### Other Loss Functions

```python
from pyAudioAnalysis_ARCO_RCI.losses import (
    RCIRegularizationLoss,  # Encourage high RCI
    ARCODistanceLoss,        # Direct arcogram distance
    CombinedARCOLoss         # Combination of multiple losses
)

# RCI regularization: penalize low RCI
rci_reg = RCIRegularizationLoss(weight=0.02, target_rci=0.8)
loss = rci_reg(pred)

# Arcogram distance: minimize L1/L2 distance
arco_dist = ARCODistanceLoss(weight=0.01, distance_metric='l1')
loss = arco_dist(pred, target)

# Combined loss
combined = CombinedARCOLoss(
    rci_aux_weight=0.05,
    rci_reg_weight=0.02,
    arco_dist_weight=0.01
)
losses_dict = combined(pred, target)
total_loss = losses_dict['total']
```

---

## Parameter Tuning

### `max_q` (Maximum Denominator)

**Default**: 30

**Effect**:
- **Higher** (50-100): More frequency resolution, captures fine periodicities, but higher computation cost
- **Lower** (10-20): Faster computation, focuses on strong fundamental periodicities

**Recommendations**:
- **Audio (speech/music)**: 30-50
- **ECG/physiological signals**: 20-30
- **Protein sequences**: 10-20

### `base_sigma` (Gaussian Width)

**Default**: 0.5

**Effect**:
- **Higher** (1.0-2.0): Wider frequency bands, more robust to frequency variations
- **Lower** (0.1-0.3): Narrower bands, higher frequency precision

**Note**: Actual width scales as `σ = base_sigma / q²`, so higher-order rationals have narrower bands automatically.

**Recommendations**:
- **Stable periodic signals**: 0.3-0.5
- **Noisy or frequency-modulated signals**: 0.8-1.5

### `window_size` (Window Length)

**Default**: 500 ms (varies by application)

**Effect**:
- **Longer** (1-2 sec): Better frequency resolution, slower time resolution
- **Shorter** (100-200 ms): Faster time resolution, captures transient periodicities

**Recommendations**:
- **Speech**: 200-500 ms
- **Music**: 500-1000 ms
- **ECG**: 1-3 seconds (several heartbeats)

### `step` (Window Step)

**Default**: 100 ms (or window_size / 5)

**Effect**:
- **Smaller**: Higher temporal resolution, more computation
- **Larger**: Faster processing, may miss transient events

**Recommendations**:
- Typically set to `window_size / 4` to `window_size / 2`

### `major_max_den` (Major Arc Threshold)

**Default**: 7

**Effect**: Determines which arcs are considered "major" for RCI computation.

**Recommendations**:
- **Default (7)**: Good for most applications
- **Lower (3-5)**: Focus on strongest fundamental periodicities
- **Higher (10-15)**: Include more harmonics in RCI

---

## Applications

### 1. Audio Analysis

**Speech periodicity**:
- Detect voiced vs. unvoiced segments (high vs. low RCI)
- Pitch tracking via peak rational arc
- Speaker diarization using ARCO-print features

**Music analysis**:
- Tempo detection
- Beat tracking
- Genre classification using arcogram patterns

### 2. Physiological Signals

**ECG**:
- Heart rate variability
- Arrhythmia detection (irregular RCI patterns)
- QRS complex analysis

**EEG**:
- Sleep stage classification (different RCI in different stages)
- Seizure detection (sudden RCI changes)

### 3. Protein Structure

**Heptad repeats** (coiled-coils):
- Peak at 1/7 indicates heptad structure
- RCI quantifies repeat strength

**Leucine zippers**:
- 7-residue periodicity detection

### 4. Machine Learning Features

Use ARCO-print as input features for:
- Classification models (e.g., speech/music classifier)
- Regression (emotion recognition, quality prediction)
- Clustering (unsupervised pattern discovery)

**Example**:
```python
# Extract features for ML
features = []
labels = []

for file, label in dataset:
    fs, signal = read_audio_file(file)
    signal = stereo_to_mono(signal)

    arco = ARCOSpectrum(max_q=30)
    arcograms, _, _ = arco.compute(signal, fs, window_size=512, step=256)

    # Aggregate to fixed-length feature
    feature = arco_print_aggregate(arcograms, method='all')
    features.append(feature)
    labels.append(label)

# Train classifier
from sklearn.svm import SVC
clf = SVC()
clf.fit(features, labels)
```

---

## Examples

### Example 1: Analyze Audio File

```python
from pyAudioAnalysis.audioBasicIO import read_audio_file, stereo_to_mono
from pyAudioAnalysis_ARCO_RCI.arc_rci import ARCOSpectrum
import matplotlib.pyplot as plt

# Load audio
fs, signal = read_audio_file('speech.wav')
signal = stereo_to_mono(signal)

# Compute arcograms
arco = ARCOSpectrum(max_q=30)
arcograms, rci, positions = arco.compute(
    signal, fs,
    window_size=int(0.5*fs),
    step=int(0.1*fs)
)

# Plot RCI timeline
plt.figure(figsize=(10, 4))
plt.plot(positions/fs, rci)
plt.xlabel('Time (s)')
plt.ylabel('RCI')
plt.title('Rational Coherence Index Over Time')
plt.grid(True)
plt.savefig('rci_timeline.png')
plt.show()
```

### Example 2: Multi-Track Analysis

```python
import numpy as np

# Create multi-track signal (e.g., protein hydropathy + charge)
track1 = np.sin(2 * np.pi * 5 * np.arange(1000) / 100)
track2 = np.sin(2 * np.pi * 7 * np.arange(1000) / 100)
multi_signal = np.vstack([track1, track2])

# Compute
arco = ARCOSpectrum(max_q=20)
arcograms, rci, pos = arco.compute(
    multi_signal, fs=100,
    window_size=200, step=50
)

# arcograms shape: (n_windows, 2, M)
print(f"Multi-track arcograms shape: {arcograms.shape}")

# Aggregate across tracks and windows
from pyAudioAnalysis_ARCO_RCI.arc_rci import arco_print_aggregate
feature = arco_print_aggregate(arcograms, method='mean')
print(f"Feature shape: {feature.shape}")  # (2, M)
```

### Example 3: GPU Batch Processing

```python
import torch
from pyAudioAnalysis_ARCO_RCI.torch_arc_rci import TorchARCOSpectrum

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
arco_gpu = TorchARCOSpectrum(max_q=30, fs=16000, window_size=512, device=device)

# Load multiple files and create batch
signals_batch = []
for file in audio_files[:32]:
    fs, sig = read_audio_file(file)
    # Extract 512-sample window
    sig_window = sig[:512]
    signals_batch.append(sig_window)

# Convert to tensor
batch = torch.from_numpy(np.array(signals_batch)).float().to(device)
print(f"Batch shape: {batch.shape}")  # (32, 512)

# Compute arcograms
arcograms_batch = arco_gpu.compute_batch(batch)
rci_batch = arco_gpu.compute_rci_batch(arcograms_batch)

print(f"Arcograms: {arcograms_batch.shape}")  # (32, M)
print(f"RCI: {rci_batch.shape}")  # (32,)
```

---

## Troubleshooting

### Issue: Peak not at expected frequency

**Solution**: Adjust `base_sigma`. Try increasing to 1.0-2.0 for noisy signals.

### Issue: Low RCI for known periodic signal

**Solution**:
- Check window size (should span several periods)
- Verify `major_max_den` includes the relevant denominators
- Try higher `max_q` to capture the periodicity

### Issue: High memory usage with GPU

**Solution**:
- Process in smaller batches
- Reduce `max_q`
- Use CPU version for very long signals

### Issue: Slow computation

**Solution**:
- Use GPU variant (`torch_arc_rci`)
- Reduce `max_q`
- Increase `step` size (fewer windows)

---

## Citation

If you use ARCO-RCI in your research, please cite:

```bibtex
@article{pyaudioanalysis,
  title={pyAudioAnalysis: An Open-Source Python Library for Audio Signal Analysis},
  author={Giannakopoulos, Theodoros},
  journal={PloS one},
  volume={10},
  number={12},
  year={2015},
  publisher={Public Library of Science}
}
```

---

## License

This module is part of pyAudioAnalysis and is licensed under the Apache License 2.0.

---

## Contact

For questions, issues, or contributions, please open an issue on the [GitHub repository](https://github.com/tyiannak/pyAudioAnalysis).
