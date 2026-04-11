# eeg/jaw_clench_detector.py

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

try:
    import pylsl
except Exception:
    pylsl = None


@dataclass
class JawClenchDetector:
    fs: float
    lowcut: float = 20.0
    highcut: float = 45.0
    refractory: float = 1.0
    merge_tol: float = 0.05
    history_seconds: float = 2.0
    min_height: float = 0.0

    recent_clenches: List[float] = field(default_factory=list)
    last_clench_time: float = 0.0

    def bandpass(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal, dtype=float)
        if signal.size < 5:
            return signal

        nyq = 0.5 * self.fs
        low = max(self.lowcut / nyq, 1e-4)
        high = min(self.highcut / nyq, 0.9999)
        if low >= high:
            return signal

        b, a = butter(4, [low, high], btype="band")
        return filtfilt(b, a, signal)

    def envelope(self, signal: np.ndarray, window_s: float = 0.05) -> np.ndarray:
        x = np.asarray(signal, dtype=float)
        n = max(int(window_s * self.fs), 1)
        if x.size < n:
            return np.sqrt(np.mean(x**2)) * np.ones_like(x)

        kernel = np.ones(n, dtype=float) / n
        power = np.convolve(x**2, kernel, mode="same")
        return np.sqrt(np.maximum(power, 0.0))

    def features(self, signal: np.ndarray) -> dict:
        x = np.asarray(signal, dtype=float)
        return {
            "rms": float(np.sqrt(np.mean(x**2))) if x.size else 0.0,
            "ptp": float(np.ptp(x)) if x.size else 0.0,
            "std": float(np.std(x)) if x.size else 0.0,
            "mean_abs": float(np.mean(np.abs(x))) if x.size else 0.0,
        }

    def detect(
        self,
        signal: np.ndarray,
        timestamps: np.ndarray,
        thresh_min: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[float], dict]:
        x = np.asarray(signal, dtype=float)
        t = np.asarray(timestamps, dtype=float)
        if x.size != t.size:
            raise ValueError("signal and timestamps must have the same length")
        if x.size == 0:
            return np.array([], dtype=int), [], {"threshold": 0.0, "features": {}}

        filtered = self.bandpass(x)
        env = self.envelope(filtered)

        spread = np.percentile(env, 95) - np.percentile(env, 5)
        absolute_min = self.min_height if thresh_min is None else thresh_min
        thresh = max(spread * 0.8, absolute_min)

        peaks, props = find_peaks(
            env,
            height=thresh,
            distance=max(int(0.20 * self.fs), 1),
            prominence=max(thresh * 0.25, 1e-6),
        )

        new_clenches: List[float] = []
        now = time.time()
        for p in peaks:
            ts = float(t[p])
            if any(abs(ts - prev) < self.merge_tol for prev in self.recent_clenches):
                continue
            if now - self.last_clench_time < self.refractory:
                continue

            self.last_clench_time = now
            self.recent_clenches.append(ts)
            new_clenches.append(ts)

        if pylsl is not None:
            now_lsl = pylsl.local_clock()
            self.recent_clenches = [ts for ts in self.recent_clenches if now_lsl - ts < self.history_seconds]
        else:
            self.recent_clenches = self.recent_clenches[-25:]

        info = {
            "threshold": float(thresh),
            "features": self.features(env),
            "n_peaks": int(len(peaks)),
            "peak_heights": props.get("peak_heights", np.array([])).tolist() if len(peaks) else [],
        }
        return peaks, new_clenches, info
