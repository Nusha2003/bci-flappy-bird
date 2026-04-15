# ui/test_window.py

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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, mode: int = 1):
        super().__init__()

        self.mode = mode

        title_map = {1: "EEG Eye Blink Flappy Bird", 2: "EEG Jaw Clench Flappy Bird"}
        self.setWindowTitle(title_map.get(mode, "EEG Flappy Bird"))
        self.resize(1100, 700)

        self.stream = EEGStream()
        self.bird = Bird(y=300)

        self.fp1_index = 1
        self.thresh_min = 70.0 if self.mode == 1 else 0.1

        self.buffer = deque(maxlen=int(self.stream.fs * 5))
        self.time_buffer = deque(maxlen=int(self.stream.fs * 5))
        self.plot_duration = 2

        self.blink_count = 0
        self.jaw_count = 0

        # --- Unified calibration state ---
        self.calibrating = True          # always calibrate, regardless of mode
        self.calibration_data = []
        self.calibration_duration = 5    # seconds
        self.calibration_start_time = None

        # Jaw clench hold-to-jump state
        self.clench_hold_until = 0.0
        self.last_jump_time = 0.0
        self.jump_interval = 0.18
        self.hold_seconds = 0.35

        if self.mode == 1:
            self.detector = BlinkDetector(self.stream.fs)
        else:
            self.detector = JawClenchDetector(self.stream.fs)

        self._setup_ui()

        calib_prompt = {
            1: "Calibration: Blink 5–10 times naturally...",
            2: "Calibration: Clench your jaw 3–5 times...",
        }
        self.label.setText(calib_prompt.get(self.mode, "Calibrating..."))

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(20)

        self.game_timer = QtCore.QTimer()
        self.game_timer.timeout.connect(self.update_game)
        self.game_timer.start(16)

    # ------------------------------------------------------------------
    # Asset helpers
    # ------------------------------------------------------------------

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

    def _load_pixmap(self, filename, fallback_size, fallback_color):
        path = self._asset_path(filename)
        pix = QtGui.QPixmap(str(path))
        if not pix.isNull():
            return pix
        fallback = QtGui.QPixmap(fallback_size[0], fallback_size[1])
        fallback.fill(fallback_color)
        return fallback

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

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

        for policy_off in [
            lambda: self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff),
            lambda: self.view.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            ),
        ]:
            try:
                policy_off()
                break
            except Exception:
                pass

        for policy_off in [
            lambda: self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff),
            lambda: self.view.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            ),
        ]:
            try:
                policy_off()
                break
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

        def _qt(attr, *fallbacks):
            for name in (attr, *fallbacks):
                v = getattr(QtCore.Qt, name, None)
                if v is not None:
                    return v
            return None

        ign = _qt("IgnoreAspectRatio", "AspectRatioMode.IgnoreAspectRatio")
        keep = _qt("KeepAspectRatio", "AspectRatioMode.KeepAspectRatio")
        smooth = _qt("SmoothTransformation", "TransformationMode.SmoothTransformation")

        self.bg_pix = self._load_pixmap(
            "flappybirdbg.png", (360, 640), QtGui.QColor(120, 200, 255)
        ).scaled(360, 640, ign, smooth)

        self.bird_pix = self._load_pixmap(
            "flappybird.png", (34, 24), QtGui.QColor(255, 220, 0)
        ).scaled(34, 24, keep, smooth)

        self.bg_item = QtWidgets.QGraphicsPixmapItem(self.bg_pix)
        self.bg_item.setZValue(-10)
        self.scene.addItem(self.bg_item)

        self.bird_item = QtWidgets.QGraphicsPixmapItem(self.bird_pix)
        self.bird_item.setOffset(-17, -12)
        self.bird_item.setZValue(10)
        self.scene.addItem(self.bird_item)

        self.bird_x = 80
        self.bird_item.setPos(self.bird_x, self.bird.y)

        self.label = QtWidgets.QLabel("Calibrating...")
        main_layout.addWidget(self.label)

    # ------------------------------------------------------------------
    # EEG update loop
    # ------------------------------------------------------------------

    def update_loop(self):
        chunk, ts = self.stream.pull()
        if len(ts) == 0:
            return

        fp1 = chunk[:, self.fp1_index]

        for v, t in zip(fp1, ts):
            self.buffer.append(v)
            self.time_buffer.append(t)
            if self.calibrating:
                self.calibration_data.append(v)

        if len(self.buffer) < int(self.stream.fs):
            return

        window_samples = int(self.stream.fs * self.plot_duration)
        data = np.array(self.buffer)[-window_samples:]
        times = np.array(self.time_buffer)[-window_samples:]

        signal = preprocess(data, self.stream.fs)
        rel_time = times - times[-1]
        self.curve.setData(rel_time, signal)

        if self.calibrating:
            self._run_calibration(signal, times)
            return

        if self.mode == 1:
            self._handle_blink_mode(signal, times)
        else:
            self._handle_jaw_mode(signal, times)

    # ------------------------------------------------------------------
    # Shared calibration logic
    # ------------------------------------------------------------------

    def _run_calibration(self, signal, times):
        if self.calibration_start_time is None:
            self.calibration_start_time = times[-1]

        elapsed = times[-1] - self.calibration_start_time

        # Update countdown in label
        remaining = max(0, int(self.calibration_duration - elapsed))
        mode_hint = {
            1: f"Calibrating: blink naturally... ({remaining}s)",
            2: f"Calibrating: clench jaw 3–5 times... ({remaining}s)",
        }
        self.label.setText(mode_hint.get(self.mode, f"Calibrating... ({remaining}s)"))

        if elapsed < self.calibration_duration:
            return

        # Time's up — run calibration
        calib_signal = preprocess(np.array(self.calibration_data), self.stream.fs)

        success = False
        if hasattr(self.detector, "calibrate"):
            success = self.detector.calibrate(calib_signal)
        else:
            # Detector has no calibrate() — skip gracefully
            success = True

        if success:
            self.calibrating = False
            ready_msg = {
                1: "Calibration complete! Blink to flap.",
                2: "Calibration complete! Clench to flap.",
            }
            self.label.setText(ready_msg.get(self.mode, "Calibration complete!"))
        else:
            # Reset and retry
            self.calibration_data = []
            self.calibration_start_time = times[-1]
            retry_msg = {
                1: "Calibration failed — blink more clearly and try again.",
                2: "Calibration failed — clench more firmly and try again.",
            }
            self.label.setText(retry_msg.get(self.mode, "Calibration failed, retrying..."))

    # ------------------------------------------------------------------
    # Detection handlers
    # ------------------------------------------------------------------

    def _handle_blink_mode(self, signal, times):
        peaks, blinks = self.detector.detect(signal, times, self.thresh_min)
        if blinks:
            self.blink_count += len(blinks)
            self.label.setText(f"Blinks: {self.blink_count}")
            self.bird.jump()

    def _handle_jaw_mode(self, signal, times):
        peaks, clenches, info = self.detector.detect(signal, times, self.thresh_min)
        if clenches:
            self.jaw_count += len(clenches)
            self.label.setText(f"Jaw clenches: {self.jaw_count}")
            self.clench_hold_until = time.monotonic() + self.hold_seconds

    # ------------------------------------------------------------------
    # Game update loop
    # ------------------------------------------------------------------

    def update_game(self):
        now = time.monotonic()

        if self.mode == 2:
            if now < self.clench_hold_until and (now - self.last_jump_time) >= self.jump_interval:
                self.bird.jump()
                self.last_jump_time = now

        self.bird.update()
        self.bird.y = max(0, min(640, self.bird.y))

        angle = max(-30, min(60, self.bird.vel * 3))
        self.bird_item.setRotation(angle)
        self.bird_item.setPos(self.bird_x, self.bird.y)


if __name__ == "__main__":
    import sys

    print("Choose control mode:")
    print("1 = Eye Blink")
    print("2 = Jaw Clench")

    try:
        mode = int(input("Enter 1 or 2: ").strip())
        if mode not in (1, 2):
            mode = 1
    except Exception:
        mode = 1

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(mode=mode)
    window.show()
    sys.exit(app.exec())