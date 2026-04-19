# eeg/jaw_clench_detector.py
import time
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import pylsl


class JawClenchDetector:
    def __init__(self, fs):
        self.fs = fs
        self.recent_clenches = []
        self.refractory = 0.35
        self.last_clench_time = 0
        self.merge_tol = 0.06

        self.baseline_window = []
        self.baseline_size = int(5 * fs)

        self.calibrated_thresh = None
        self.calibrated = False

    def _bandpass(self, signal, lowcut, highcut):
        x = np.asarray(signal, dtype=float)
        if x.size < 20:
            return x

        nyq = 0.5 * self.fs
        low = max(lowcut / nyq, 1e-4)
        high = min(highcut / nyq, 0.9999)
        if low >= high:
            return x

        b, a = butter(4, [low, high], btype="band")
        padlen = 3 * max(len(a), len(b))
        if x.size <= padlen:
            return x

        return filtfilt(b, a, x)

    def _envelope(self, signal, window_s=0.05):
        x = np.asarray(signal, dtype=float)
        n = max(int(window_s * self.fs), 1)

        if x.size < n:
            return np.sqrt(np.mean(x ** 2)) * np.ones_like(x)

        kernel = np.ones(n, dtype=float) / n
        power = np.convolve(x ** 2, kernel, mode="same")
        return np.sqrt(np.maximum(power, 0.0))

    def _robust_stats(self, x):
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return 0.0, 1.0

        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        scale = 1.4826 * mad

        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.std(x))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0

        return med, scale

    def _jaw_envelope(self, signal):
        jaw_band = self._bandpass(signal, 20.0, 45.0)
        return self._envelope(jaw_band)

    def calibrate(self, signal):
        signal = np.asarray(signal, dtype=float)
        if signal.size == 0:
            print("Calibration failed: empty jaw signal")
            return False

        jaw_env = self._jaw_envelope(signal)
        base_med, base_scale = self._robust_stats(jaw_env)

        seed_thresh = max(
            base_med + 1.2 * base_scale,
            float(np.percentile(jaw_env, 75)),
        )

        peaks, props = find_peaks(
            jaw_env,
            height=seed_thresh,
            width=(int(0.02 * self.fs), int(0.30 * self.fs)),
            distance=int(0.15 * self.fs),
            prominence=max(seed_thresh * 0.04, 1e-6),
        )

        if len(peaks) < 3:
            print("Calibration failed: not enough jaw clench candidates")
            return False

        peak_heights = props["peak_heights"]
        cutoff = np.percentile(peak_heights, 20)
        peak_heights = peak_heights[peak_heights > cutoff]

        if len(peak_heights) == 0:
            print("Calibration failed: jaw clenches too weak")
            return False

        self.calibrated_thresh = float(np.percentile(peak_heights, 35)) * 0.9
        self.calibrated = True

        self.baseline_window = jaw_env[-self.baseline_size :].tolist()

        print(f"Calibration complete. Threshold = {self.calibrated_thresh:.3f}")
        return True

    def detect(self, signal, timestamps, thresh_min):
        signal = np.asarray(signal, dtype=float)
        timestamps = np.asarray(timestamps, dtype=float)

        if signal.size == 0:
            return np.array([], dtype=int), [], {
                "threshold": 0.0,
                "n_peaks": 0,
                "peak_heights": [],
            }

        jaw_env = self._jaw_envelope(signal)

        self.baseline_window.extend(jaw_env.tolist())
        if len(self.baseline_window) > self.baseline_size:
            self.baseline_window = self.baseline_window[-self.baseline_size:]

        base_med, base_scale = self._robust_stats(self.baseline_window)

        adaptive_thresh = max(
            base_med + 1.8 * base_scale,
            float(np.percentile(jaw_env, 88)),
        )

        if self.calibrated:
            thresh = max(
                float(self.calibrated_thresh),
                adaptive_thresh * 0.85,
                float(thresh_min),
            )
        else:
            thresh = max(adaptive_thresh, float(thresh_min))

        peaks, props = find_peaks(
            jaw_env,
            height=thresh,
            width=(int(0.02 * self.fs), int(0.30 * self.fs)),
            distance=int(0.15 * self.fs),
            prominence=max(thresh * 0.06, 1e-6),
        )

        new_clenches = []
        now = time.time()

        for p in peaks:
            t = float(timestamps[p])

            if any(abs(t - prev) < self.merge_tol for prev in self.recent_clenches):
                continue

            if now - self.last_clench_time < self.refractory:
                continue

            self.last_clench_time = now
            self.recent_clenches.append(t)
            new_clenches.append(t)

        now_lsl = pylsl.local_clock()
        self.recent_clenches = [t for t in self.recent_clenches if now_lsl - t < 2]

        info = {
            "threshold": float(thresh),
            "n_peaks": int(len(peaks)),
            "peak_heights": props.get("peak_heights", np.array([])).tolist() if len(peaks) else [],
        }
        return peaks, new_clenches, info
