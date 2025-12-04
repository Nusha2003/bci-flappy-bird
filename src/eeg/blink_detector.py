import numpy as np
import time
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, filtfilt
from collections import deque

from display_eeg import EEGWindow, init_eeg_window, update_eeg_window

def highpass_mne(x, fs):
    b, a = butter(4, 2/(fs/2), btype='high')
    return filtfilt(b, a, x)

def blink_band_mne(x, fs):
    b, a = butter(4, [1/(fs/2), 10/(fs/2)], btype='band')
    return filtfilt(b, a, x)

def main():
    print("Looking for FakeEEG...")
    streams = resolve_byprop("name", "FakeEEG", timeout=5)
    inlet = StreamInlet(streams[0])

    fs = int(inlet.info().nominal_srate())
    buffer_size = fs * 2

    buffer = deque(maxlen=buffer_size)
    refractory = 0.3
    last_blink_time = 0

    fp1_index = 21

    playback_window = init_eeg_window(
        buffer_size=buffer_size,
        ylim=(-150, 150),
        title="Real-Time Blink Detection (Fp1)",
    )

    blink_markers = deque(maxlen=10)

    print("Connected. Plotting real-time EEG with blink detection...")

    while True:
        samples, _ = inlet.pull_chunk(timeout=1.0, max_samples=64)
        if not samples:
            continue

        samples = np.array(samples)
        fp1_chunk = samples[:, fp1_index]

        for v in fp1_chunk:
            buffer.append(v)

        if len(buffer) < buffer_size:
            continue

        data = np.array(buffer)

        hp = highpass_mne(data, fs)
        bp = blink_band_mne(hp, fs)
        temp = bp - np.mean(bp)

        # Polarity
        if np.max(temp) > abs(np.min(temp)):
            signal = temp
        else:
            signal = -temp

        dynamic_threshold = (np.max(signal) - np.min(signal)) / 6
        peak_value = np.max(signal)

        blink_detected = False

        if peak_value > dynamic_threshold and (time.time() - last_blink_time > refractory):
            print("BLINK DETECTED!")
            last_blink_time = time.time()
            blink_detected = True
            blink_markers.append(len(signal) - 1)  # mark last sample

        update_eeg_window(playback_window, signal, blink_markers)

        time.sleep(0.01)


if __name__ == "__main__":
    main()