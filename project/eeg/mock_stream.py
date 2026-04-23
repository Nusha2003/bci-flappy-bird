import time
import numpy as np

# DSI-style channel layout: F3 F4 C3 C4 P3 P4 + trigger
_CH_NAMES = ["F3", "F4", "C3", "C4", "P3", "P4", "TRG"]
_F4_IDX = 1  # index of F4 in _CH_NAMES


class MockEEGStream:
    """Drop-in replacement for EEGStream that generates synthetic EEG data.

    Background: low-amplitude Gaussian noise on all channels.
    Events: periodic blink-like spikes on F4 (large positive peak) and
    jaw/hand clench spikes (broad high-amplitude burst on all channels).
    Call trigger_event() to inject an event immediately.
    """

    fs: int = 300
    ch_names: list[str] = _CH_NAMES

    def __init__(self, auto_event_interval: float = 4.0):
        """
        auto_event_interval: seconds between auto-fired simulated events (0 to disable).
        """
        self._start = time.monotonic()
        self._last_pull = self._start
        self._rng = np.random.default_rng()

        self._auto_interval = auto_event_interval
        self._next_auto_event = self._start + auto_event_interval if auto_event_interval > 0 else None

        # Manual trigger: number of blink-spike samples remaining to inject
        self._pending_spike_samples = 0

    def trigger_event(self) -> None:
        """Inject a blink-like spike into the next pull() output."""
        spike_duration_s = 0.08  # 80 ms spike
        self._pending_spike_samples = int(spike_duration_s * self.fs)

    def pull(self) -> tuple[np.ndarray, np.ndarray]:
        now = time.monotonic()
        elapsed = now - self._last_pull
        self._last_pull = now

        n_samples = max(1, int(elapsed * self.fs))
        n_ch = len(self.ch_names)

        # Background noise: ~10 µV amplitude
        data = self._rng.normal(0, 10.0, size=(n_samples, n_ch))

        # Auto-fire events
        if self._next_auto_event is not None and now >= self._next_auto_event:
            self._pending_spike_samples = int(0.08 * self.fs)
            jitter = self._rng.uniform(self._auto_interval * 0.8, self._auto_interval * 1.2)
            self._next_auto_event = now + jitter

        # Inject spike samples
        if self._pending_spike_samples > 0:
            inject = min(self._pending_spike_samples, n_samples)
            # Blink: large positive peak on F4 (and F3 slightly)
            data[:inject, _F4_IDX] += 150.0
            f3_idx = self.ch_names.index("F3") if "F3" in self.ch_names else None
            if f3_idx is not None:
                data[:inject, f3_idx] += 60.0
            self._pending_spike_samples -= inject

        timestamps = np.linspace(now - elapsed, now, n_samples, endpoint=False)
        return data, timestamps
