import time
import numpy as np


# Matches the real EEGStream interface: .fs, .pull() → (chunk, timestamps)
class MockEEGStream:
    """
    Generates synthetic EEG noise without requiring LSL hardware.

    Channel layout mirrors the real headset (8 ch); Fp1 is index 1.
    Call inject_spike() to simulate a large blink/jaw event on Fp1.
    """

    FS = 256          # samples per second
    N_CHANNELS = 8    # must match real device so downstream indexing is unchanged
    FP1_INDEX = 1
    NOISE_AMP = 30.0  # µV background noise amplitude — visible in ±150 µV plot

    def __init__(self):
        self.fs = self.FS
        self._last_pull = time.monotonic()
        self._rng = np.random.default_rng()
        self._pending_spike = False   # set True by inject_spike()

    def inject_spike(self, amplitude: float = 200.0):
        """Queue a single large spike on Fp1 for the next pull() call."""
        self._spike_amp = amplitude
        self._pending_spike = True

    def pull(self):
        now = time.monotonic()
        elapsed = now - self._last_pull
        self._last_pull = now

        n_samples = max(1, int(elapsed * self.fs))
        chunk = self._rng.normal(0, self.NOISE_AMP, (n_samples, self.N_CHANNELS))
        timestamps = now - np.linspace(elapsed, 0, n_samples, endpoint=False)[::-1]

        if self._pending_spike:
            self._pending_spike = False
            # Place a sharp Gaussian peak in the middle of the chunk on Fp1
            mid = n_samples // 2
            width = max(1, int(0.05 * self.fs))  # ~50 ms peak
            peak = np.zeros(n_samples)
            for i in range(n_samples):
                peak[i] = self._spike_amp * np.exp(-0.5 * ((i - mid) / width) ** 2)
            chunk[:, self.FP1_INDEX] += peak

        return chunk, timestamps
