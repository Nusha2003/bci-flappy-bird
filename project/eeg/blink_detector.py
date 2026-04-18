import time
import numpy as np
from collections import deque
from scipy.signal import find_peaks
import pylsl


class BlinkDetector:
    def __init__(self, fs):
        self.fs = fs
        self.recent_blinks = []
        self.refractory = 0.5
        self.last_blink_time = 0
        self.merge_tol = 0.04

        # calibration
        self.calibrated_thresh = None
        self.calibrated = False

    def _preprocess(self, signal):
        signal = signal - np.median(signal)
        if abs(np.min(signal)) > abs(np.max(signal)):
            signal = -signal

        return signal

    def calibrate(self, signal):
        signal = self._preprocess(signal)

        signal_range = np.percentile(signal, 95) - np.percentile(signal, 5)

        peaks, props = find_peaks(
            signal,
            height=signal_range * 0.15,
            width=(int(0.04*self.fs), int(0.4*self.fs)),
            distance=int(0.2*self.fs)
        )

        if len(peaks) < 3:
            print("Calibration failed: not enough blink candidates")
            return False

        peak_heights = props["peak_heights"]

        low  = np.percentile(peak_heights, 5)
        high = np.percentile(peak_heights, 95)
        trimmed = peak_heights[(peak_heights >= low) & (peak_heights <= high)]

        if len(trimmed) == 0:
            print("Calibration failed: peaks too weak")
            return False

        self.calibrated_thresh = float(np.mean(trimmed))
        self.calibrated = True

        print(f"Calibration complete. Threshold = {self.calibrated_thresh:.2f}")
        return True

    def detect(self, signal, timestamps, thresh_min):
        signal = self._preprocess(signal)

        if self.calibrated:
            thresh = self.calibrated_thresh
        else:
            signal_range = np.percentile(signal, 95) - np.percentile(signal, 5)
            thresh = max(signal_range * 0.4, thresh_min)

        peaks, _ = find_peaks(
            signal,
            height=thresh,
            width=(int(0.05*self.fs), int(0.3*self.fs)),
            distance=int(0.25*self.fs)
        )

        new_blinks = []
        now = time.time()

        for p in peaks:
            t = timestamps[p]

            if any(abs(t - prev) < self.merge_tol for prev in self.recent_blinks):
                continue

            if now - self.last_blink_time > self.refractory:
                self.last_blink_time = now
                new_blinks.append(t)

            self.recent_blinks.append(t)
        now_lsl = pylsl.local_clock()
        self.recent_blinks = [t for t in self.recent_blinks if now_lsl - t < 2]

        return peaks, new_blinks