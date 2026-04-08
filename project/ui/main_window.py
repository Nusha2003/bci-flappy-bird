# ui/main_window.py

import numpy as np
from collections import deque
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from eeg.stream import EEGStream
from eeg.preprocess import preprocess
from eeg.blink_detector import BlinkDetector
from game.flappy import Bird


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("EEG Blink Flappy Bird")
        self.resize(1100, 700)

        self.stream = EEGStream()
        self.detector = BlinkDetector(self.stream.fs)
        self.bird = Bird(y=300)

        self.fp1_index = 1
        self.thresh_min = 70.0

        self.buffer = deque(maxlen=int(self.stream.fs * 5))
        self.time_buffer = deque(maxlen=int(self.stream.fs * 5))

        self.plot_duration = 2
        self.blink_count = 0

        self._setup_ui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(20)

        self.game_timer = QtCore.QTimer()
        self.game_timer.timeout.connect(self.update_game)
        self.game_timer.start(16)
    def _setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)

        split = QtWidgets.QHBoxLayout()
        main_layout.addLayout(split)

        self.plot = pg.PlotWidget(title="EEG (Fp1)")
        self.plot.setYRange(-150, 150)
        self.plot.showGrid(x=True, y=True, alpha=0.3)

        self.curve = self.plot.plot(pen='y')
        split.addWidget(self.plot, stretch=2)

        self.scene = QtWidgets.QGraphicsScene(0, 0, 360, 640)
        self.view = QtWidgets.QGraphicsView(self.scene)

        self.view.setRenderHints(
            QtGui.QPainter.Antialiasing |
            QtGui.QPainter.SmoothPixmapTransform
        )
        self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.view.setFrameShape(QtWidgets.QFrame.NoFrame)

        split.addWidget(self.view, stretch=3)
        bg_path = "C:\\Users\winni\\bci-flappy-bird\\mvp\src\\game\\flappybirdbg.png"
        bird_path = "C:\\Users\winni\\bci-flappy-bird\\mvp\src\\game\\flappybird.png"

        self.bg_pix = QtGui.QPixmap(bg_path).scaled(
            360, 640,
            QtCore.Qt.IgnoreAspectRatio,
            QtCore.Qt.SmoothTransformation
        )

        self.bird_pix = QtGui.QPixmap(bird_path).scaled(
            34, 24,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )

        # Background
        self.bg_item = QtWidgets.QGraphicsPixmapItem(self.bg_pix)
        self.scene.addItem(self.bg_item)

        # Bird
        self.bird_item = QtWidgets.QGraphicsPixmapItem(self.bird_pix)
        self.bird_item.setOffset(-17, -12)  # center it
        self.scene.addItem(self.bird_item)

        # Initial position
        self.bird_x = 80
        self.bird_item.setPos(self.bird_x, self.bird.y)

        # ---- Bottom label ----
        self.label = QtWidgets.QLabel("Blinks: 0")
        main_layout.addWidget(self.label)

    # -------------------------
    # EEG Loop
    # -------------------------
    def update_loop(self):
        chunk, ts = self.stream.pull()
        if len(ts) == 0:
            return

        fp1 = chunk[:, self.fp1_index]

        for v, t in zip(fp1, ts):
            self.buffer.append(v)
            self.time_buffer.append(t)

        if len(self.buffer) < self.stream.fs:
            return

        window_samples = int(self.stream.fs * self.plot_duration)
        data = np.array(self.buffer)[-window_samples:]
        times = np.array(self.time_buffer)[-window_samples:]

        signal = preprocess(data, self.stream.fs)

        rel_time = times - times[-1]
        self.curve.setData(rel_time, signal)

        peaks, blinks = self.detector.detect(signal, times, self.thresh_min)

        if blinks:
            self.blink_count += len(blinks)
            self.label.setText(f"Blinks: {self.blink_count}")
            self.bird.jump()

    # -------------------------
    # Game Loop
    # -------------------------
    def update_game(self):
        self.bird.update()

        # Clamp
        if self.bird.y < 0:
            self.bird.y = 0
        if self.bird.y > 640:
            self.bird.y = 640

        # Rotation (adds game feel)
        angle = max(-30, min(60, self.bird.vel * 3))
        self.bird_item.setRotation(angle)

        self.bird_item.setPos(self.bird_x, self.bird.y)