"""
Unit tests for ARCO-RCI functionality
======================================

Test cases:
1. Sine harmonic test - verify peak detection at known frequency
2. Heptad repeat test - detect periodic patterns (protein-like)
3. RCI null test - statistical significance of RCI
4. Consistency test - multi-track equals sum of single-track
5. Basic functionality tests
"""

import unittest
import numpy as np
from fractions import Fraction
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyAudioAnalysis_ARCO_RCI.arc_rci import (
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


class TestRationalArcs(unittest.TestCase):
    """Test rational arc generation."""

    def test_generate_rational_arcs(self):
        """Test that rational arcs are generated correctly."""
        arcs = generate_rational_arcs(max_q=10)

        # Should have multiple arcs
        self.assertGreater(len(arcs), 0)

        # All should be Fraction objects
        self.assertTrue(all(isinstance(a, Fraction) for a in arcs))

        # All should be in (0, 0.5)
        for arc in arcs:
            r = float(arc)
            self.assertGreater(r, 0.0)
            self.assertLess(r, 0.5)

        # Should be sorted
        values = [float(a) for a in arcs]
        self.assertEqual(values, sorted(values))

        # Should not contain 1/2
        self.assertNotIn(Fraction(1, 2), arcs)

        # Should contain some known fractions
        self.assertIn(Fraction(1, 3), arcs)
        self.assertIn(Fraction(1, 4), arcs)

    def test_rational_arcs_uniqueness(self):
        """Test that arcs are unique and reduced."""
        arcs = generate_rational_arcs(max_q=20)

        # Convert to set and check size is same
        arc_set = set(arcs)
        self.assertEqual(len(arcs), len(arc_set))

        # Check all are in lowest terms
        for arc in arcs:
            # Fraction is automatically reduced, so just check it's a Fraction
            self.assertEqual(arc, Fraction(arc.numerator, arc.denominator))


class TestSignalProcessing(unittest.TestCase):
    """Test basic signal processing functions."""

    def test_sequence_to_signal(self):
        """Test signal normalization."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = sequence_to_signal(signal, normalize=True)

        # Should have mean close to 0 and std close to 1
        self.assertAlmostEqual(normalized.mean(), 0.0, places=10)
        self.assertAlmostEqual(normalized.std(), 1.0, places=10)

        # Without normalization
        unnormalized = sequence_to_signal(signal, normalize=False)
        np.testing.assert_array_equal(unnormalized, signal)

    def test_compute_power_spectrum(self):
        """Test power spectrum computation."""
        # Create a simple sine wave
        fs = 1000
        t = np.arange(0, 1, 1/fs)
        f0 = 10  # 10 Hz sine wave
        signal = np.sin(2 * np.pi * f0 * t)

        freqs, power = compute_power_spectrum(signal, fs, use_hann=True)

        # Check frequency bins
        self.assertEqual(len(freqs), len(power))
        self.assertGreater(len(freqs), 0)
        self.assertAlmostEqual(freqs[0], 0.0)

        # Power should sum to 1 (normalized)
        self.assertAlmostEqual(power.sum(), 1.0, places=5)

        # Peak should be near f0
        peak_idx = np.argmax(power)
        peak_freq = freqs[peak_idx]
        self.assertAlmostEqual(peak_freq, f0, delta=2.0)  # Within 2 Hz


class TestSineHarmonicDetection(unittest.TestCase):
    """Test 1: Sine harmonic test - detect known frequency."""

    def test_sine_detection(self):
        """
        Generate a sine wave of known frequency and verify that the
        arcogram peak corresponds to the correct rational.
        """
        # Parameters
        fs = 1000  # Sampling rate
        f0 = 50    # Fundamental frequency (50 Hz)
        duration = 2.0  # 2 seconds

        # Generate sine wave
        t = np.arange(0, duration, 1/fs)
        signal = np.sin(2 * np.pi * f0 * t)

        # Expected rational: f0 / nyquist = 50 / 500 = 1/10
        nyquist = fs / 2.0
        expected_r = f0 / nyquist
        self.assertAlmostEqual(expected_r, 0.1)

        # Create ARCO spectrum
        arco = ARCOSpectrum(max_q=30, base_sigma=0.5)

        # Compute arcogram for entire signal
        window_size = len(signal)
        arcogram = arco.compute_window(signal, fs)

        # Find peak
        peak_idx = np.argmax(arcogram)
        peak_frac = arco.rational_arcs[peak_idx]

        # The peak should be at or very near 1/10
        peak_value = float(peak_frac)
        self.assertAlmostEqual(peak_value, expected_r, delta=0.02)

        # Print for debugging
        print(f"\nSine test: f0={f0} Hz, expected r={expected_r:.3f}")
        print(f"Peak at: {peak_frac} = {peak_value:.3f}")

        # Top 3 peaks
        top3_idx = np.argsort(arcogram)[::-1][:3]
        print("Top 3 arcs:")
        for i, idx in enumerate(top3_idx):
            frac = arco.rational_arcs[idx]
            print(f"  {i+1}. {frac} ({float(frac):.3f}): power={arcogram[idx]:.6f}")


class TestHeptadRepeat(unittest.TestCase):
    """Test 2: Heptad repeat test - detect periodic pattern."""

    def test_heptad_pattern(self):
        """
        Generate a synthetic signal with 7-residue repeating motif
        and verify peak near 1/7.
        """
        # Create a repeating pattern with period 7
        motif = np.array([1.0, -0.5, 0.8, -0.3, 0.6, -0.2, 0.4])
        n_repeats = 50
        signal = np.tile(motif, n_repeats)

        # Add small noise
        np.random.seed(42)
        signal = signal + 0.1 * np.random.randn(len(signal))

        # Treat as sampled at fs=1 (one sample per residue)
        fs = 1.0

        # Create ARCO
        arco = ARCOSpectrum(max_q=30, base_sigma=0.5)

        # Compute arcogram
        arcogram = arco.compute_window(signal, fs)

        # Find peak
        peak_idx = np.argmax(arcogram)
        peak_frac = arco.rational_arcs[peak_idx]

        # Expected: For period 7 at fs=1, fundamental freq = 1/7 Hz
        # Normalized: f / Nyquist = (1/7) / 0.5 = 2/7
        expected = Fraction(2, 7)
        expected_value = float(expected)

        print(f"\nHeptad test: period=7, expected r={expected_value:.3f}")
        print(f"Peak at: {peak_frac} = {float(peak_frac):.3f}")

        # The peak should be at or near 2/7
        self.assertAlmostEqual(float(peak_frac), expected_value, delta=0.05)

        # Also check that 2/7 is among top 3
        top3_idx = np.argsort(arcogram)[::-1][:3]
        top3_fracs = [arco.rational_arcs[i] for i in top3_idx]

        print("Top 3 arcs:")
        for i, idx in enumerate(top3_idx):
            frac = arco.rational_arcs[idx]
            print(f"  {i+1}. {frac} ({float(frac):.3f}): power={arcogram[idx]:.6f}")

        # 2/7 should be in top 3 (or very close rational)
        # Check if peak is close to expected value
        self.assertTrue(
            any(abs(float(frac) - expected_value) < 0.05 for frac in top3_fracs),
            f"Expected rational near {expected} not in top 3"
        )


class TestRCINull(unittest.TestCase):
    """Test 3: RCI null test - statistical significance."""

    def test_rci_null_significance(self):
        """
        For a periodic sequence, compute RCI. Then shuffle the sequence
        multiple times and compute RCI null distribution. The original
        RCI should have a z-score > 2.
        """
        # Create a strong periodic signal (period 5)
        motif = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        n_repeats = 40
        signal = np.tile(motif, n_repeats)

        # Add small noise
        np.random.seed(42)
        signal = signal + 0.1 * np.random.randn(len(signal))

        fs = 1.0

        # Create ARCO
        arco = ARCOSpectrum(max_q=20, base_sigma=0.5)

        # Compute RCI for original signal
        arcogram_orig = arco.compute_window(signal, fs)
        rci_orig = arco.compute_rci(arcogram_orig)

        print(f"\nRCI null test:")
        print(f"Original RCI: {rci_orig:.4f}")

        # Compute RCI for shuffled versions
        n_shuffles = 20
        rci_null = []

        np.random.seed(42)
        for i in range(n_shuffles):
            shuffled = signal.copy()
            np.random.shuffle(shuffled)

            arcogram_shuffled = arco.compute_window(shuffled, fs)
            rci_shuffled = arco.compute_rci(arcogram_shuffled)
            rci_null.append(rci_shuffled)

        rci_null = np.array(rci_null)
        null_mean = rci_null.mean()
        null_std = rci_null.std()

        # Compute z-score
        z_score = (rci_orig - null_mean) / (null_std + 1e-10)

        print(f"Null RCI: {null_mean:.4f} ± {null_std:.4f}")
        print(f"Z-score: {z_score:.2f}")

        # Original RCI should be significantly higher
        self.assertGreater(z_score, 2.0, "RCI z-score should be > 2")
        self.assertGreater(rci_orig, null_mean, "RCI should be higher than null mean")


class TestMultiTrackConsistency(unittest.TestCase):
    """Test 4: Multi-track consistency test."""

    def test_multitrack_vs_single_track(self):
        """
        Verify that multi-track arcogram computation is consistent
        with single-track computation.
        """
        # Create two different signals
        np.random.seed(42)
        fs = 100
        N = 500

        t = np.arange(N) / fs
        signal1 = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(N)
        signal2 = np.sin(2 * np.pi * 7 * t) + 0.1 * np.random.randn(N)

        # Stack as multi-track
        multi_track = np.vstack([signal1, signal2])  # Shape: (2, N)

        # Create ARCO
        arco = ARCOSpectrum(max_q=20)

        # Compute multi-track arcogram
        arcogram_multi = arco.compute_window(multi_track, fs)
        # Should be shape (2, M)
        self.assertEqual(arcogram_multi.shape[0], 2)

        # Compute single-track arcograms
        arcogram1 = arco.compute_window(signal1, fs)
        arcogram2 = arco.compute_window(signal2, fs)

        # Check consistency
        np.testing.assert_array_almost_equal(
            arcogram_multi[0], arcogram1, decimal=10
        )
        np.testing.assert_array_almost_equal(
            arcogram_multi[1], arcogram2, decimal=10
        )

        print("\nMulti-track consistency test passed!")


class TestARCOSpectrum(unittest.TestCase):
    """Test ARCOSpectrum class."""

    def test_arco_spectrum_init(self):
        """Test ARCOSpectrum initialization."""
        arco = ARCOSpectrum(max_q=20, base_sigma=0.3)

        self.assertEqual(arco.max_q, 20)
        self.assertEqual(arco.base_sigma, 0.3)
        self.assertGreater(len(arco.rational_arcs), 0)
        self.assertGreater(len(arco.major_indices), 0)

    def test_arco_spectrum_sliding_window(self):
        """Test arcogram computation with sliding windows."""
        # Create a signal with time-varying frequency
        fs = 1000
        duration = 2.0
        t = np.arange(0, duration, 1/fs)

        # First half: 10 Hz, second half: 20 Hz
        signal = np.concatenate([
            np.sin(2 * np.pi * 10 * t[:len(t)//2]),
            np.sin(2 * np.pi * 20 * t[len(t)//2:])
        ])

        # Create ARCO
        arco = ARCOSpectrum(max_q=30)

        # Compute with sliding windows
        window_size = 500
        step = 250

        arcograms, rci_windows, positions = arco.compute(
            signal, fs, window_size, step
        )

        # Check shapes
        n_windows = len(positions)
        M = len(arco.rational_arcs)

        self.assertEqual(arcograms.shape, (n_windows, M))
        self.assertEqual(rci_windows.shape, (n_windows,))
        self.assertEqual(positions.shape, (n_windows,))

        # Check that positions make sense
        self.assertGreater(n_windows, 0)
        self.assertGreater(positions[0], 0)
        self.assertLess(positions[-1], len(signal))

        print(f"\nSliding window test: {n_windows} windows computed")


class TestRCIComputation(unittest.TestCase):
    """Test RCI computation."""

    def test_rci_basic(self):
        """Test basic RCI computation."""
        # Create a simple arcogram
        arcogram = np.array([0.1, 0.2, 0.3, 0.15, 0.1, 0.05, 0.05, 0.05])

        # Major indices: first 3
        major_indices = [0, 1, 2]

        rci = compute_rci_from_arcogram(arcogram, major_indices)

        # RCI = (0.1 + 0.2 + 0.3) / 1.0 = 0.6
        expected = 0.6
        self.assertAlmostEqual(rci, expected, places=5)

        # RCI should be in [0, 1]
        self.assertGreaterEqual(rci, 0.0)
        self.assertLessEqual(rci, 1.0)

    def test_rci_edge_cases(self):
        """Test RCI edge cases."""
        # All zeros
        arcogram = np.zeros(10)
        rci = compute_rci_from_arcogram(arcogram, [0, 1, 2])
        self.assertEqual(rci, 0.0)

        # All power in major arcs
        arcogram = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
        rci = compute_rci_from_arcogram(arcogram, [0, 1, 2])
        self.assertAlmostEqual(rci, 1.0, places=5)


class TestArcoPrint(unittest.TestCase):
    """Test ARCO-print aggregation."""

    def test_arco_print_mean(self):
        """Test mean aggregation."""
        # Create fake arcograms
        arcograms = np.random.rand(100, 50)  # 100 windows, 50 arcs

        feature = arco_print_aggregate(arcograms, method='mean')

        # Should be shape (50,)
        self.assertEqual(feature.shape, (50,))

        # Should equal mean
        expected = np.mean(arcograms, axis=0)
        np.testing.assert_array_almost_equal(feature, expected)

    def test_arco_print_all(self):
        """Test 'all' aggregation method."""
        arcograms = np.random.rand(100, 50)

        feature = arco_print_aggregate(arcograms, method='all')

        # Should be shape (4*50,) = (200,)
        self.assertEqual(feature.shape, (200,))

    def test_arco_print_multitrack(self):
        """Test aggregation for multi-track."""
        # Shape: (n_windows, n_tracks, M)
        arcograms = np.random.rand(100, 3, 50)

        feature = arco_print_aggregate(arcograms, method='mean')

        # Should aggregate over windows, keeping tracks
        # Result: (3, 50) then flattened
        self.assertEqual(feature.shape, (3, 50))


class TestPickMajorArcs(unittest.TestCase):
    """Test major arc selection."""

    def test_pick_major_arcs(self):
        """Test picking major arcs."""
        arcs = generate_rational_arcs(max_q=20)
        major_idx = pick_major_arcs(arcs, max_den=5)

        # Should return indices
        self.assertGreater(len(major_idx), 0)

        # All indices should be valid
        for idx in major_idx:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(arcs))

        # All selected arcs should have denominator <= 5
        for idx in major_idx:
            frac = arcs[idx]
            self.assertLessEqual(frac.denominator, 5)


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def test_full_pipeline(self):
        """Test full pipeline from signal to ARCO-print."""
        # Generate test signal
        np.random.seed(42)
        fs = 1000
        duration = 1.0
        t = np.arange(0, duration, 1/fs)
        signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 15 * t)
        signal = sequence_to_signal(signal, normalize=True)

        # Create ARCO
        arco = ARCOSpectrum(max_q=30)

        # Compute arcograms
        window_size = 200
        step = 100
        arcograms, rci_windows, positions = arco.compute(
            signal, fs, window_size, step
        )

        # Compute ARCO-print
        arco_print = arco_print_aggregate(arcograms, method='mean')

        # Verify shapes and values
        self.assertGreater(len(arcograms), 0)
        self.assertEqual(len(rci_windows), len(arcograms))
        self.assertEqual(len(arco_print), len(arco.rational_arcs))

        # RCI should be in valid range
        self.assertTrue(np.all(rci_windows >= 0.0))
        self.assertTrue(np.all(rci_windows <= 1.0))

        # ARCO-print should be positive
        self.assertTrue(np.all(arco_print >= 0.0))

        print(f"\nFull pipeline test:")
        print(f"  Signal length: {len(signal)} samples")
        print(f"  Windows: {len(arcograms)}")
        print(f"  Mean RCI: {rci_windows.mean():.4f}")
        print(f"  ARCO-print shape: {arco_print.shape}")


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestRationalArcs))
    suite.addTests(loader.loadTestsFromTestCase(TestSignalProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestSineHarmonicDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestHeptadRepeat))
    suite.addTests(loader.loadTestsFromTestCase(TestRCINull))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiTrackConsistency))
    suite.addTests(loader.loadTestsFromTestCase(TestARCOSpectrum))
    suite.addTests(loader.loadTestsFromTestCase(TestRCIComputation))
    suite.addTests(loader.loadTestsFromTestCase(TestArcoPrint))
    suite.addTests(loader.loadTestsFromTestCase(TestPickMajorArcs))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
