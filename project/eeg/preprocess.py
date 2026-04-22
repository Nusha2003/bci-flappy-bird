# eeg/processing.py
from scipy.signal import butter, filtfilt, find_peaks
import numpy as np

def highpass(x, fs):
    b, a = butter(4, 2/(fs/2), btype='high')
    return filtfilt(b, a, x)

def bandpass(x, fs):
    b, a = butter(4, [1/(fs/2), 10/(fs/2)], btype='band')
    return filtfilt(b, a, x)

def preprocess(signal, fs):
    hp = highpass(signal, fs)
    bp = bandpass(hp, fs)
    signal = bp - np.mean(bp)

    if np.max(signal) < abs(np.min(signal)):
        signal = -signal

    return signal


def smooth_signal(signal, fs, window_ms=80):
    """Light display-only smoothing to make the live trace less jittery."""
    x = np.asarray(signal, dtype=float)
    if x.size < 3:
        return x

    window_n = max(3, int(round((window_ms / 1000.0) * fs)))
    if window_n % 2 == 0:
        window_n += 1

    kernel = np.ones(window_n, dtype=float) / window_n
    return np.convolve(x, kernel, mode="same")
