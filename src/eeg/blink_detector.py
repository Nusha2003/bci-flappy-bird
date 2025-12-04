import numpy as np
import time
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from collections import deque

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

    fs = int(inlet.info().nominal_srate())  # 160 Hz
    buffer_size = fs * 2  # 2 seconds = 320 samples

    buffer = deque(maxlen=buffer_size)
    refractory = 0.3
    last_blink_time = 0

    fp1_index = 21 

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    line, = ax.plot([], [], lw=1.5)
    ax.set_ylim(-150, 150)
    ax.set_xlim(0, buffer_size)
    ax.set_title("Real-Time Blink Detection (Fp1)")
    ax.set_xlabel("Samples (last 2 seconds)")
    ax.set_ylabel("Amplitude (µV)")

    # store blink markers
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

        dynamic_threshold = (np.max(signal) - np.min(signal)) / 4
        peak_value = np.max(signal)

        blink_detected = False

        if peak_value > dynamic_threshold and (time.time() - last_blink_time > refractory):
            print("BLINK DETECTED!")
            last_blink_time = time.time()
            blink_detected = True
            blink_markers.append(len(signal) - 1)  # mark last sample

        line.set_ydata(signal)
        line.set_xdata(np.arange(len(signal)))

        for artist in ax.lines[1:]:
            artist.remove()
            
        for blink_x in blink_markers:
            ax.axvline(blink_x, color='red', linewidth=2)

        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(0.01)


if __name__ == "__main__":
    main()
