# ui/main_window.py

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
    """Splash screen shown before the game. Emits mode_selected(int) on choice."""

    mode_selected = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #1a1a2e;")

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

        self.mode = None  # set after mode select screen

        # Show mode select first
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

        self.event_count = 0  # blinks or clenches detected total

        # Calibration state
        self.calibrating = True
        self.calibration_data = []
        self.calibration_duration = 5
        self.calibration_start_time = None
        self.calib_dot_count = 0       # dots shown in label (0–5 cycling)
        self.calib_detected = 0        # events detected during calibration
        self.last_dot_time = 0.0

        # Jaw hold-to-jump state (mode 2 only)
        self.clench_hold_until = 0.0
        self.last_jump_time = 0.0
        self.jump_interval = 0.18
        self.hold_seconds = 0.35

        if self.mode == 1:
            self.detector = BlinkDetector(self.stream.fs)
            self.setWindowTitle("EEG Eye Blink Flappy Bird")
        else:
            self.detector = JawClenchDetector(self.stream.fs)
            self.setWindowTitle("EEG Jaw Clench Flappy Bird")

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
    # Game UI
    # ------------------------------------------------------------------

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
        self.scene.setSceneRect(0, 0, 360, 640)
        self.view = QtWidgets.QGraphicsView(self.scene)

        for fn in [
            lambda: self.view.setRenderHint(QtGui.QPainter.Antialiasing, True),
        ]:
            try:
                fn()
            except Exception:
                pass

        for fn in [
            lambda: self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff),
            lambda: self.view.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            ),
        ]:
            try:
                fn(); break
            except Exception:
                pass

        for fn in [
            lambda: self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff),
            lambda: self.view.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            ),
        ]:
            try:
                fn(); break
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

        def _qt(*names):
            for n in names:
                v = getattr(QtCore.Qt, n, None)
                if v is not None:
                    return v

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

        self.label = QtWidgets.QLabel("")
        self.label.setStyleSheet("font-size: 14px; padding: 4px;")
        main_layout.addWidget(self.label)

    # ------------------------------------------------------------------
    # Calibration label helpers
    # ------------------------------------------------------------------

    def _tick_dots(self):
        """Called every 500 ms to animate the '...' in the calibration label."""
        if self.calibrating:
            self.calib_dot_count = (self.calib_dot_count + 1) % 4
            self._update_calib_label()

    def _update_calib_label(self):
        dots = "." * self.calib_dot_count
        pad = " " * (3 - self.calib_dot_count)  # keep label width stable
        if self.mode == 1:
            self.label.setText(
                f"Calibrating blinks{dots}{pad}  — blinks detected: {self.calib_detected}"
            )
        else:
            self.label.setText(
                f"Calibrating jaw clenches{dots}{pad}  — clenches detected: {self.calib_detected}"
            )

    # ------------------------------------------------------------------
    # EEG loop
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
        self.curve.setData(times - times[-1], signal)

        if self.calibrating:
            self._run_calibration(signal, times)
            return

        # Post-calibration detection — only one motor drives the jump
        if self.mode == 1:
            self._handle_blink(signal, times)
        else:
            self._handle_jaw(signal, times)

    # ------------------------------------------------------------------
    # Shared calibration
    # ------------------------------------------------------------------

    def _run_calibration(self, signal, times):
        if self.calibration_start_time is None:
            self.calibration_start_time = times[-1]

        # Count events live during calibration so user gets feedback
        if self.mode == 1:
            _, events = self.detector.detect(signal, times, self.thresh_min)
        else:
            _, events, _ = self.detector.detect(signal, times, self.thresh_min)

        if events:
            self.calib_detected += len(events)
            self._update_calib_label()

        elapsed = times[-1] - self.calibration_start_time
        if elapsed < self.calibration_duration:
            return

        # Time's up — run the appropriate calibration
        calib_signal = preprocess(np.array(self.calibration_data), self.stream.fs)

        success = False
        if hasattr(self.detector, "calibrate"):
            success = self.detector.calibrate(calib_signal)
        else:
            success = True  # detector has no calibrate(); skip gracefully

        if success:
            self.calibrating = False
            self.dot_timer.stop()
            self.calib_detected = 0
            if self.mode == 1:
                self.label.setText("Calibration complete! Blink to flap.")
            else:
                self.label.setText("Calibration complete! Clench to flap.")
        else:
            # Reset and retry
            self.calibration_data = []
            self.calibration_start_time = times[-1]
            self.calib_detected = 0
            self.calib_dot_count = 0
            if self.mode == 1:
                self.label.setText("Calibration failed — blink more clearly, retrying...")
            else:
                self.label.setText("Calibration failed — clench more firmly, retrying...")

    # ------------------------------------------------------------------
    # Detection handlers (post-calibration)
    # ------------------------------------------------------------------

    def _handle_blink(self, signal, times):
        _, blinks = self.detector.detect(signal, times, self.thresh_min)
        if blinks:
            self.event_count += len(blinks)
            self.label.setText(f"Blinks: {self.event_count}")
            self.bird.jump()

    def _handle_jaw(self, signal, times):
        _, clenches, _ = self.detector.detect(signal, times, self.thresh_min)
        if clenches:
            self.event_count += len(clenches)
            self.label.setText(f"Jaw clenches: {self.event_count}")
            self.clench_hold_until = time.monotonic() + self.hold_seconds

    # ------------------------------------------------------------------
    # Game loop
    # ------------------------------------------------------------------

    def update_game(self):
        now = time.monotonic()

        if self.mode == 2 and not self.calibrating:
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

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())