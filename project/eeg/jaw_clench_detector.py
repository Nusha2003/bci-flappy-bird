import time
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import pylsl


class JawClenchDetector:
    def __init__(self, fs: float):
        self.fs = fs

        self.recent_clenches = []
        self.refractory = 1.0  
        self.last_clench_time = 0
        self.merge_tol = 0.5

        # calibration
        self.calibrated_thresh = None
        self.calibrated = False

    def _preprocess(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal, dtype=float)
        signal = signal - np.median(signal)
        signal = np.abs(signal)
        return signal
    def calibrate(self, signal: np.ndarray) -> bool:
        signal = self._preprocess(signal)

        if signal.size == 0:
            print("Calibration failed: empty jaw signal")
            return False

        signal_range = np.percentile(signal, 95) - np.percentile(signal, 5)

        peaks, props = find_peaks(
            signal,
            height=signal_range * 0.25,   # jaw needs higher threshold than blink
            width=(int(0.03 * self.fs), int(0.25 * self.fs)),
            distance=int(0.15 * self.fs),
        )

        if len(peaks) < 3:
            print("Calibration failed: not enough jaw clench candidates")
            return False

        peak_heights = props["peak_heights"]

        cutoff = np.percentile(peak_heights, 25)
        peak_heights = peak_heights[peak_heights > cutoff]

        if len(peak_heights) == 0:
            print("Calibration failed: jaw peaks too weak")
            return False

        self.calibrated_thresh = np.percentile(peak_heights, 40)
        self.calibrated_thresh *= 0.85  # safety margin

        self.calibrated = True

        print(f"Jaw calibration complete. Threshold = {self.calibrated_thresh:.3f}")
        return True

    def detect(self, signal: np.ndarray, timestamps: np.ndarray, thresh_min: float):
        signal = np.asarray(signal, dtype=float)
        timestamps = np.asarray(timestamps, dtype=float)

        if signal.size == 0:
            return np.array([], dtype=int), [], {
                "threshold": 0.0,
                "n_peaks": 0,
                "peak_heights": [],
            }

        signal = self._preprocess(signal)

        signal_range = np.percentile(signal, 95) - np.percentile(signal, 5)

        adaptive_thresh = max(
            signal_range * 0.6,
            float(np.percentile(signal, 85)),
            float(thresh_min),
        )

        if self.calibrated and self.calibrated_thresh is not None:
            thresh = max(self.calibrated_thresh, adaptive_thresh * 0.8)
        else:
            thresh = adaptive_thresh

        peaks, props = find_peaks(
            signal,
            height=thresh,
            width=(int(0.03 * self.fs), int(0.20 * self.fs)),
            distance=int(0.12 * self.fs),
            prominence=max(thresh * 0.05, 1e-6),
        )

        new_clenches = []
        now = time.time()

        for p in peaks:
            t = float(timestamps[p])

            # merge duplicates
            if any(abs(t - prev) < self.merge_tol for prev in self.recent_clenches):
                continue

            # refractory (prevents spam detection)
            if now - self.last_clench_time < self.refractory:
                continue

            self.last_clench_time = now
            self.recent_clenches.append(t)
            new_clenches.append(t)
        now_lsl = pylsl.local_clock()
        self.recent_clenches = [
            t for t in self.recent_clenches if now_lsl - t < 2
        ]

        info = {
            "threshold": float(thresh),
            "n_peaks": int(len(peaks)),
            "peak_heights": (
                props.get("peak_heights", np.array([])).tolist()
                if len(peaks)
                else []
            ),
        }

        return peaks, new_clenches, info