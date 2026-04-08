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