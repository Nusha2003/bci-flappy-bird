import time
import numpy as np
from collections import deque
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from eeg.stream import EEGStream
from eeg.preprocess import preprocess
from eeg.blink_detector import BlinkDetector
from eeg.jaw_clench_detector import JawClenchDetector
from game.flappy import Bird


class ModeSelectScreen(QtWidgets.QWidget):
    mode_selected = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        layout.setSpacing(24)

        title = QtWidgets.QLabel("BCI Flappy Bird")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel("Choose your control method")
        subtitle.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(16)

        btn_blink = QtWidgets.QPushButton("Eye Blink")
        btn_blink.setFixedSize(220, 64)
        btn_blink.clicked.connect(lambda: self.mode_selected.emit(1))
        layout.addWidget(btn_blink, alignment=QtCore.Qt.AlignCenter)

        btn_jaw = QtWidgets.QPushButton("Jaw Clench")
        btn_jaw.setFixedSize(220, 64)
        btn_jaw.clicked.connect(lambda: self.mode_selected.emit(2))
        layout.addWidget(btn_jaw, alignment=QtCore.Qt.AlignCenter)

        hint = QtWidgets.QLabel("Blink to flap  ·  Clench to flap")
        hint.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(hint)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI Flappy Bird")
        self.resize(1100, 700)

        self.mode = None

        self._select_screen = ModeSelectScreen()
        self._select_screen.mode_selected.connect(self._on_mode_selected)
        self.setCentralWidget(self._select_screen)

    def _on_mode_selected(self, mode: int):
        self.mode = mode
        self._init_game()

    def _init_game(self):
        self.stream = EEGStream()
        self.bird = Bird(y=300)

        self.fp1_index = 1
        self.thresh_min = 70.0 if self.mode == 1 else 0.1

        self.buffer = deque(maxlen=int(self.stream.fs * 5))
        self.time_buffer = deque(maxlen=int(self.stream.fs * 5))
        self.plot_duration = 2

        self.event_count = 0

        # Calibration
        self.calibrating = True
        self.calibration_data = []
        self.calibration_duration = 5
        self.calibration_start_time = None
        self.calib_dot_count = 0
        self.calib_detected = 0

        # Jaw control timing
        self.clench_hold_until = 0.0
        self.last_jump_time = 0.0
        self.jump_interval = 0.18
        self.hold_seconds = 0.35

        # ---- KEY FIX: Blink veto system ----
        if self.mode == 1:
            self.detector = BlinkDetector(self.stream.fs)
            self.blink_veto_detector = None
            self.setWindowTitle("EEG Eye Blink Flappy Bird")
        else:
            self.detector = JawClenchDetector(self.stream.fs)
            self.blink_veto_detector = BlinkDetector(self.stream.fs)
            self.setWindowTitle("EEG Jaw Clench Flappy Bird")

        self.blink_veto_thresh = 70.0
        self.jaw_block_until = 0.0
        self.jaw_block_seconds = 0.45

        self._setup_game_ui()
        self._update_calib_label()

        self.eeg_timer = QtCore.QTimer()
        self.eeg_timer.timeout.connect(self.update_loop)
        self.eeg_timer.start(20)

        self.dot_timer = QtCore.QTimer()
        self.dot_timer.timeout.connect(self._tick_dots)
        self.dot_timer.start(500)

        self.game_timer = QtCore.QTimer()
        self.game_timer.timeout.connect(self.update_game)
        self.game_timer.start(16)

    # ---------------- UI ----------------

    def _setup_game_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)
        split = QtWidgets.QHBoxLayout()
        main_layout.addLayout(split)

        self.plot = pg.PlotWidget(title="EEG (Fp1)")
        self.plot.setYRange(-150, 150)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot(pen="y")
        split.addWidget(self.plot, stretch=2)

        self.scene = QtWidgets.QGraphicsScene(0, 0, 360, 640)
        self.view = QtWidgets.QGraphicsView(self.scene)
        split.addWidget(self.view, stretch=3)

        self.bird_item = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.bird_item)

        self.bird_x = 80

        self.label = QtWidgets.QLabel("")
        main_layout.addWidget(self.label)

    # ---------------- Calibration UI ----------------

    def _tick_dots(self):
        if self.calibrating:
            self.calib_dot_count = (self.calib_dot_count + 1) % 4
            self._update_calib_label()

    def _update_calib_label(self):
        dots = "." * self.calib_dot_count
        if self.mode == 1:
            self.label.setText(f"Calibrating blinks{dots} — {self.calib_detected}")
        else:
            self.label.setText(f"Calibrating clenches{dots} — {self.calib_detected}")

    # ---------------- EEG LOOP ----------------

    def update_loop(self):
        chunk, ts = self.stream.pull()
        if len(ts) == 0:
            return

        fp1 = chunk[:, self.fp1_index]

        for v, t in zip(fp1, ts):
            self.buffer.append(v)
            self.time_buffer.append(t)

        if len(self.buffer) < int(self.stream.fs):
            return

        window_samples = int(self.stream.fs * self.plot_duration)
        data = np.array(self.buffer)[-window_samples:]
        times = np.array(self.time_buffer)[-window_samples:]

        signal = preprocess(data, self.stream.fs)
        self.curve.setData(times - times[-1], signal)

        # ---- Blink veto detection ----
        blink_events = []
        if self.mode == 2 and self.blink_veto_detector:
            _, blink_events = self.blink_veto_detector.detect(
                signal, times, self.blink_veto_thresh
            )

        # ---- Calibration ----
        if self.calibrating:
            if self.mode == 1 or not blink_events:
                self.calibration_data.extend(fp1.tolist())

            self._run_calibration(signal, times, blink_events)
            return

        # ---- Detection ----
        if self.mode == 1:
            self._handle_blink(signal, times)
        else:
            self._handle_jaw(signal, times, blink_events)

    # ---------------- Calibration Logic ----------------

    def _run_calibration(self, signal, times, blink_events=None):
        if self.calibration_start_time is None:
            self.calibration_start_time = times[-1]

        if self.mode == 1:
            _, events = self.detector.detect(signal, times, self.thresh_min)
        else:
            if blink_events:
                events = []
            else:
                _, events, _ = self.detector.detect(signal, times, self.thresh_min)

        if events:
            self.calib_detected += len(events)
            self._update_calib_label()

        elapsed = times[-1] - self.calibration_start_time
        if elapsed < self.calibration_duration:
            return

        calib_signal = preprocess(np.array(self.calibration_data), self.stream.fs)

        success = self.detector.calibrate(calib_signal) if hasattr(self.detector, "calibrate") else True

        if success:
            self.calibrating = False
            self.dot_timer.stop()
            self.label.setText("Calibration complete!")
        else:
            self.calibration_data = []
            self.calibration_start_time = times[-1]
            self.calib_detected = 0

    # ---------------- Detection ----------------

    def _handle_blink(self, signal, times):
        _, blinks = self.detector.detect(signal, times, self.thresh_min)
        if blinks:
            self.event_count += len(blinks)
            self.bird.jump()

    def _handle_jaw(self, signal, times, blink_events=None):
        now = time.monotonic()

        # ---- HARD BLOCK ON BLINK ----
        if blink_events:
            self.jaw_block_until = now + self.jaw_block_seconds
            return

        if now < self.jaw_block_until:
            return

        _, clenches, _ = self.detector.detect(signal, times, self.thresh_min)
        if clenches:
            self.event_count += len(clenches)
            self.clench_hold_until = now + self.hold_seconds

    # ---------------- Game Loop ----------------

    def update_game(self):
        now = time.monotonic()

        if self.mode == 2 and not self.calibrating:
            if now < self.clench_hold_until and (now - self.last_jump_time) >= self.jump_interval:
                self.bird.jump()
                self.last_jump_time = now

        self.bird.update()
        self.bird.y = max(0, min(640, self.bird.y))
        self.bird_item.setPos(self.bird_x, self.bird.y)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())