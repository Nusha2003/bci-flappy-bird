import time
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt, find_peaks

import pylsl
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui


############################################
# FILTERING
############################################
def highpass(x, fs):
    b, a = butter(4, 2/(fs/2), btype='high')
    return filtfilt(b, a, x)

def bandpass(x, fs):
    b, a = butter(4, [1/(fs/2), 10/(fs/2)], btype='band')
    return filtfilt(b, a, x)


############################################
# MAIN
############################################
def main():
    print("Looking for FakeEEG stream...")
    streams = pylsl.resolve_byprop("name", "FakeEEG", timeout=5)
    if len(streams) == 0:
        print("No FakeEEG stream found!")
        return

    inlet = pylsl.StreamInlet(
        streams[0],
        max_buflen=10,
        processing_flags=pylsl.proc_dejitter | pylsl.proc_clocksync
    )

    fs = int(inlet.info().nominal_srate())
    fp1_index = 21
    buffer_secs = 5
    plot_duration = 5
    buffer = deque(maxlen=int(fs * buffer_secs))

    ############################################
    # PYQTGRAPH WINDOW
    ############################################
    win = pg.plot(title="Real-time EEG (Fp1) + Blink Peaks")
    plt = win.getPlotItem()
    plt.setLabel("bottom", "Time (s)")
    plt.setYRange(-150, 150)

    # yellow EEG trace
    curve = plt.plot(pen='y')

    # scatter plot for blink peaks
    blink_scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush('r'))
    plt.addItem(blink_scatter)

    ############################################
    # SCROLLING WINDOW
    ############################################
    def scroll():
        plt.setXRange(-plot_duration, 0)

    ############################################
    # UPDATE LOOP
    ############################################
    def update():
        chunk, timestamps = inlet.pull_chunk(
            max_samples=int(fs / 3), timeout=0.0
        )
        if not timestamps:
            return

        chunk = np.array(chunk)
        fp1 = chunk[:, fp1_index]

        # Add new samples to buffer
        for v in fp1:
            buffer.append(v)

        if len(buffer) < fs:
            return

        # Only last 5 seconds
        window_samples = fs * plot_duration
        data = np.array(buffer)[-window_samples:] * 1e6  # convert to µV

        ############################################
        # FILTER EEG
        ############################################
        hp = highpass(data, fs)
        bp = bandpass(hp, fs)
        temp = bp - np.mean(bp)
        signal = temp if np.max(temp) > abs(np.min(temp)) else -temp

        ############################################
        # TIME AXIS (rolling window)
        ############################################
        times = np.linspace(-plot_duration, 0, len(signal))
        curve.setData(times, signal)

        ############################################
        # BLINK PEAK DETECTION (dots)
        ############################################
        thresh = max((np.max(signal) - np.min(signal)) / 4, )

        # detect peaks above threshold
        peaks, _ = find_peaks(signal, height=thresh, distance=int(fs * 0.2))

        # assign points
        pts = [{'pos': (times[p], signal[p])} for p in peaks]
        blink_scatter.setData(pts)

    ############################################
    # TIMERS
    ############################################
    update_timer = QtCore.QTimer()
    update_timer.timeout.connect(update)
    update_timer.start(20)

    scroll_timer = QtCore.QTimer()
    scroll_timer.timeout.connect(scroll)
    scroll_timer.start(20)

    QtGui.QGuiApplication.instance().exec_()


if __name__ == "__main__":
    main()
