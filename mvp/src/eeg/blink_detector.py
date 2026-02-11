import numpy as np
import time
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, filtfilt
from collections import deque
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

def highpass(x, fs):
    b, a = butter(4, 2/(fs/2), btype='high')
    return filtfilt(b, a, x)

def bandpass(x, fs):
    b, a = butter(4, [1/(fs/2), 10/(fs/2)], btype='band')
    return filtfilt(b, a, x)

def main():
    print("Looking for FakeEEG...")
    streams = resolve_byprop("name", "FakeEEG", timeout=5)
    inlet = StreamInlet(streams[0])

    fs = int(inlet.info().nominal_srate())     # 160 Hz
    buffer_size = fs * 2                        # 2 sec = 320 samples
    buffer = deque(maxlen=buffer_size)

    fp1_index = 21
    refractory = 0.3
    last_blink_time = 0

    app = QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="Real-Time Blink Detection")
    win.resize(1000, 300)

    plot = win.addPlot(title="Filtered EEG (Fp1)")
    curve = plot.plot(pen='y')

    blink_lines = []
    plot.setYRange(-150, 150)

    print("Connected. Streaming real-time plot...")

    def update():
        nonlocal last_blink_time

        samples, _ = inlet.pull_chunk(max_samples=64)
        if not samples:
            return

        samples = np.array(samples)
        fp1_chunk = samples[:, fp1_index]

        for v in fp1_chunk:
            buffer.append(v)

        if len(buffer) < buffer_size:
            return

        data = np.array(buffer)

        hp = highpass(data, fs)
        bp = bandpass(hp, fs)
        temp = bp - np.mean(bp)

        signal = temp if np.max(temp) > abs(np.min(temp)) else -temp


        thresh = (np.max(signal) - np.min(signal)) / 4
        peak_value = np.max(signal)

        blink = peak_value > thresh and (time.time() - last_blink_time > refractory)

        curve.setData(signal)

        if blink:
            last_blink_time = time.time()
            x = len(signal) - 1
            line = pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen('r', width=2))
            plot.addItem(line)
            blink_lines.append((line, time.time()))

        for line, t in list(blink_lines):
            if time.time() - t > 1:
                plot.removeItem(line)
                blink_lines.remove((line, t))

    # Timer for updating plot
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(20)

    app.exec()


if __name__ == "__main__":
    main()
