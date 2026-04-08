import time
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt, find_peaks
import pylsl

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

def highpass(x, fs):
    b, a = butter(4, 2/(fs/2), btype='high')
    return filtfilt(b, a, x)

def bandpass(x, fs):
    b, a = butter(4, [1/(fs/2), 10/(fs/2)], btype='band')
    return filtfilt(b, a, x)

class BlinkFlappyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.used_peak_indices = set()
        self.recent_blinks = []   
        self.blink_merge_tolerance = 0.04 
        self.setWindowTitle("EEG Blink - Flappy Bird Demo")
        self.resize(1100, 700)

        print("Looking for FakeEEG stream...")
        streams = pylsl.resolve_byprop("name", "WS-default", timeout=10)
        
      

        self.inlet = pylsl.StreamInlet(
            streams[0],
            max_buflen=10,
            processing_flags=pylsl.proc_dejitter | pylsl.proc_clocksync
        )

        self.fs = int(self.inlet.info().nominal_srate())
        self.fp1_index = 1
        self.buffer_secs = 5
        self.plot_duration = 2

        self.buffer = deque(maxlen=int(self.fs * self.buffer_secs))
        self.time_buffer = deque(maxlen=int(self.fs * self.buffer_secs))

        self.blink_count = 0
        self.last_blink_time = 0.0              # wall-clock (time.time)
        self.refractory = 1.0                # seconds
        self.thresh_min = 70.0                  # µV minimum threshold
        self.bird_jump_pending = False

        self.used_blink_times = set()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)

        top_bar = QtWidgets.QHBoxLayout()
        main_layout.addLayout(top_bar)

        top_bar.addWidget(QtWidgets.QLabel("Blink Threshold (µV):"))
        self.thresh_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.thresh_slider.setRange(40, 200)
        self.thresh_slider.setValue(int(self.thresh_min))
        self.thresh_slider.valueChanged.connect(self.on_thresh_changed)
        top_bar.addWidget(self.thresh_slider)

        self.thresh_label = QtWidgets.QLabel(f"{int(self.thresh_min)} µV")
        top_bar.addWidget(self.thresh_label)
        top_bar.addStretch()
        split_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(split_layout, stretch=1)

        self.eeg_plot = pg.PlotWidget(title="Real-time EEG (Fp1) + Blink Peaks")
        self.eeg_plot.setLabel("bottom", "Time (s)")
        self.eeg_plot.setYRange(-150, 150)
        self.eeg_curve = self.eeg_plot.plot(pen='y')
        self.blink_scatter = pg.ScatterPlotItem(
            size=14,
            symbol='o',             # circle
            brush=None,             # hollow
            pen=pg.mkPen('r', width=2)   # red outline
        )
        self.eeg_plot.addItem(self.blink_scatter)
        split_layout.addWidget(self.eeg_plot, stretch=1)

        self.game_width = 360
        self.game_height = 640
        self.bird_width = 34
        self.bird_height = 24

        self.scene = QtWidgets.QGraphicsScene(0, 0, self.game_width, self.game_height)
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        split_layout.addWidget(self.view, stretch=1)

        bg_path = "/Users/anusha/bci-flappy-bird/src/game/flappybirdbg.png"
        bird_path = "/Users/anusha/bci-flappy-bird/src/game/flappybird.png"

        self.bg_pix = QtGui.QPixmap(bg_path)
        self.bird_pix = QtGui.QPixmap(bird_path).scaled(
            self.bird_width, self.bird_height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )

        self.bg_item = QtWidgets.QGraphicsPixmapItem(self.bg_pix)
        self.bg_item.setZValue(0)
        self.scene.addItem(self.bg_item)

        self.bird_item = QtWidgets.QGraphicsPixmapItem(self.bird_pix)
        self.bird_item.setZValue(1)
        self.scene.addItem(self.bird_item)

        self.bird_x = self.game_width / 8
        self.bird_y = self.game_height / 2
        self.bird_vel = 0.0
        self.gravity = 0.5
        self.jump_strength = 15.0

        self.bird_item.setPos(self.bird_x, self.bird_y)

        bottom_bar = QtWidgets.QHBoxLayout()
        main_layout.addLayout(bottom_bar)

        self.blink_label = QtWidgets.QLabel("Blinks detected: 0")
        bottom_bar.addWidget(self.blink_label)
        bottom_bar.addStretch()

        self.eeg_timer = QtCore.QTimer()
        self.eeg_timer.timeout.connect(self.update_eeg)
        self.eeg_timer.start(20)  # ms

        self.game_timer = QtCore.QTimer()
        self.game_timer.timeout.connect(self.update_bird)
        self.game_timer.start(16)  # ~60 FPS

    def on_thresh_changed(self, value):
        self.thresh_min = float(value)
        self.thresh_label.setText(f"{value} µV")

    def update_eeg(self):
        chunk, timestamps = self.inlet.pull_chunk(
            max_samples=int(self.fs / 3), timeout=0.0
        )
        if not timestamps:
            return

        chunk = np.array(chunk)
        timestamps = np.array(timestamps)
        fp1 = chunk[:, self.fp1_index]

        # Fill buffers: values + times
        for v, t in zip(fp1, timestamps):
            self.buffer.append(v)
            self.time_buffer.append(t)

        if len(self.buffer) < self.fs:
            return

        # Last N seconds
        window_samples = int(self.fs * self.plot_duration)
        data = np.array(self.buffer)[-window_samples:] * 1e6  # µV
        time_window = np.array(self.time_buffer)[-window_samples:]  # LSL timestamps

        # Filter
        hp = highpass(data, self.fs)
        bp = bandpass(hp, self.fs)
        signal = bp - np.mean(bp)

        # Polarity flip so blinks are positive
        if np.max(signal) < abs(np.min(signal)):
            signal = -signal

        rel_times = time_window - time_window[-1]

        self.eeg_curve.setData(rel_times, signal)
        self.eeg_plot.setXRange(-self.plot_duration, 0, padding=0)


        signal_range = np.percentile(signal, 95) - np.percentile(signal, 5)
        dynamic_thresh = signal_range * 0.6
        thresh = max(dynamic_thresh, self.thresh_min)

        peaks, props = find_peaks(
            signal,
            height=thresh,
            width=(int(0.05 * self.fs), int(0.30 * self.fs)),  # width in samples
            distance=int(0.25 * self.fs)
        )


        pts = [{'pos': (rel_times[p], signal[p])} for p in peaks]
        self.blink_scatter.setData(pts)

        now_wall = time.time()
        now_lsl = pylsl.local_clock()

        for p in peaks:
            peak_lsl = time_window[p]

            # Check if we've already seen a blink within ±40 ms
            if any(abs(peak_lsl - prev) < self.blink_merge_tolerance for prev in self.recent_blinks):
                continue  # SAME blink → do not retrigger

            # This is a NEW blink
            now = time.time()
            if now - self.last_blink_time > self.refractory:
                self.last_blink_time = now
                self.blink_count += 1
                self.blink_label.setText(f"Blinks detected: {self.blink_count}")
                self.bird_jump_pending = True

            # Save blink time
            self.recent_blinks.append(peak_lsl)

        # Keep last 2 seconds of blink times only
        now_lsl = pylsl.local_clock()
        self.recent_blinks = [t for t in self.recent_blinks if now_lsl - t < 2.0]


        self.used_blink_times = {
            t for t in self.used_blink_times
            if now_lsl - t < 2.0    
        }

    def update_bird(self):
        if self.bird_jump_pending:
            self.bird_vel = -self.jump_strength
            self.bird_jump_pending = False

        self.bird_vel += self.gravity
        self.bird_y += self.bird_vel

        if self.bird_y < 0:
            self.bird_y = 10
            self.bird_vel = 0
        if self.bird_y > self.game_height - self.bird_height:
            self.bird_y = self.game_height - self.bird_height
            self.bird_vel = 0

        self.bird_item.setPos(self.bird_x, self.bird_y)
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = BlinkFlappyWindow()
    win.show()
    app.exec_()
