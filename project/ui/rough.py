"""
import time
import numpy as np
from collections import deque
from pathlib import Path
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from eeg.stream import EEGStream
from eeg.preprocess import preprocess
from eeg.blink_detector import BlinkDetector
from game.flappy import Bird


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("EEG Jaw Clenches Flappy Bird")
        self.resize(1100, 700)

        self.stream = EEGStream()
        self.detector = JawClenchDetector(self.stream.fs)
        self.bird = Bird(y=300)

        self.fp1_index = 1
        self.thresh_min = 0.1

        self.buffer = deque(maxlen=int(self.stream.fs * 5))
        self.time_buffer = deque(maxlen=int(self.stream.fs * 5))

        self.plot_duration = 2
        self.jaw_count = 0

        # Hold-to-jump state
        self.clench_hold_until = 0.0
        self.last_jump_time = 0.0
        self.jump_interval = 0.18   # smaller = faster repeated jumps while held
        self.hold_seconds = 0.35    # how long a detected clench stays "active"

        self._setup_ui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(20)

        self.game_timer = QtCore.QTimer()
        self.game_timer.timeout.connect(self.update_game)
        self.game_timer.start(16)

    def _asset_path(self, filename: str) -> Path:
        root = Path(__file__).resolve().parents[2]
        candidates = [
            root / "project" / "game" / filename,
            root / "mvp" / "src" / "game" / filename,
        ]
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    def _load_pixmap(self, filename: str, fallback_size: tuple[int, int], fallback_color: QtGui.QColor) -> QtGui.QPixmap:
        path = self._asset_path(filename)
        pix = QtGui.QPixmap(str(path))
        if not pix.isNull():
            return pix

        fallback = QtGui.QPixmap(fallback_size[0], fallback_size[1])
        fallback.fill(fallback_color)
        return fallback

    def _setup_ui(self):
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
        self.scene.setSceneRect(0, 0, 360, 640)

        self.view = QtWidgets.QGraphicsView(self.scene)

        try:
            self.view.setRenderHint(QtGui.QPainter.Antialiasing, True)
            self.view.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        except Exception:
            pass

        try:
            self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        except Exception:
            try:
                self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            except Exception:
                pass

        try:
            self.view.setFrameShape(QtWidgets.QFrame.NoFrame)
        except Exception:
            try:
                self.view.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            except Exception:
                pass

        split.addWidget(self.view, stretch=3)

        self.bg_pix = self._load_pixmap(
            "flappybirdbg.png",
            (360, 640),
            QtGui.QColor(120, 200, 255),
        ).scaled(
            360, 640,
            QtCore.Qt.AspectRatioMode.IgnoreAspectRatio
            if hasattr(QtCore.Qt, "AspectRatioMode")
            else QtCore.Qt.IgnoreAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
            if hasattr(QtCore.Qt, "TransformationMode")
            else QtCore.Qt.SmoothTransformation,
        )

        self.bird_pix = self._load_pixmap(
            "flappybird.png",
            (34, 24),
            QtGui.QColor(255, 220, 0),
        ).scaled(
            34, 24,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio
            if hasattr(QtCore.Qt, "AspectRatioMode")
            else QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
            if hasattr(QtCore.Qt, "TransformationMode")
            else QtCore.Qt.SmoothTransformation,
        )

        self.bg_item = QtWidgets.QGraphicsPixmapItem(self.bg_pix)
        self.bg_item.setZValue(-10)
        self.scene.addItem(self.bg_item)

        self.bird_item = QtWidgets.QGraphicsPixmapItem(self.bird_pix)
        self.bird_item.setOffset(-17, -12)
        self.bird_item.setZValue(10)
        self.scene.addItem(self.bird_item)

        self.bird_x = 80
        self.bird_item.setPos(self.bird_x, self.bird.y)

        self.label = QtWidgets.QLabel("Jaw Clenches: 0")
        main_layout.addWidget(self.label)

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

        rel_time = times - times[-1]
        self.curve.setData(rel_time, signal)

        peaks, clenches, info = self.detector.detect(signal, times, self.thresh_min)
        #print(info["jaw_threshold"], info["score_threshold"], len(clenches))

        if clenches:
            self.jaw_count += len(clenches)
            self.label.setText(f"Jaw Clenches: {self.jaw_count}")

            # Hold jump mode while the clench is active
            self.clench_hold_until = time.monotonic() + self.hold_seconds

    def update_game(self):
        now = time.monotonic()

        # Keep jumping while the clench is being held
        if now < self.clench_hold_until and (now - self.last_jump_time) >= self.jump_interval:
            self.bird.jump()
            self.last_jump_time = now

        self.bird.update()

        if self.bird.y < 0:
            self.bird.y = 0
        if self.bird.y > 640:
            self.bird.y = 640

        angle = max(-30, min(60, self.bird.vel * 3))
        self.bird_item.setRotation(angle)
        self.bird_item.setPos(self.bird_x, self.bird.y)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

    ------------------------------------------------------------------

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

        self.setWindowTitle("EEG Blinks Flappy Bird")
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
        self.calibrating = True
        self.calibration_data = []
        self.calibration_duration = 5  # seconds
        self.calibration_start_time = None
        self.calibration_start_time = None
        self.label.setText("Calibration: Blink 5–10 times...")
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

            # collect calibration data
            if self.calibrating:
                self.calibration_data.append(v)

        if len(self.buffer) < self.stream.fs:
            return

        window_samples = int(self.stream.fs * self.plot_duration)
        data = np.array(self.buffer)[-window_samples:]
        times = np.array(self.time_buffer)[-window_samples:]

        signal = preprocess(data, self.stream.fs)

        rel_time = times - times[-1]
        self.curve.setData(rel_time, signal)

        if self.calibrating:
            if self.calibration_start_time is None:
                self.calibration_start_time = ts[-1]

            elapsed = ts[-1] - self.calibration_start_time

            if elapsed >= self.calibration_duration:
                print("Running calibration...")

                calib_signal = preprocess(
                    np.array(self.calibration_data),
                    self.stream.fs
                )

                success = self.detector.calibrate(calib_signal)

                if success:
                    self.calibrating = False
                    self.label.setText("Calibration complete! Start blinking to play")
                else:
                    self.label.setText("Calibration failed, try again")
                    self.calibration_data = []
                    self.calibration_start_time = ts[-1]

            return 

        peaks, blinks = self.detector.detect(signal, times, self.thresh_min)

        if blinks:
            self.blink_count += len(blinks)
            self.label.setText(f"Blinks: {self.blink_count}")
            self.bird.jump()

    def update_game(self):
        self.bird.update()

        if self.bird.y < 0:
            self.bird.y = 0
        if self.bird.y > 640:
            self.bird.y = 640

        angle = max(-30, min(60, self.bird.vel * 3))
        self.bird_item.setRotation(angle)

        self.bird_item.setPos(self.bird_x, self.bird.y) 
"""