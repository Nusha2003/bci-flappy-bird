# eeg/blink_detector.py
import time
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt, find_peaks
import pylsl

class BlinkDetector:
    def __init__(self, fs):
        self.fs = fs
        self.recent_blinks = []
        self.refractory = 1.0
        self.last_blink_time = 0
        self.merge_tol = 0.04

    def detect(self, signal, timestamps, thresh_min):
        signal_range = np.percentile(signal, 95) - np.percentile(signal, 5)
        thresh = max(signal_range * 0.6, thresh_min)

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

        # cleanup
        now_lsl = pylsl.local_clock()
        self.recent_blinks = [t for t in self.recent_blinks if now_lsl - t < 2]

        return peaks, new_blinks