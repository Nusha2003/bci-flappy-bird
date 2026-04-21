from pathlib import Path

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from controller import EEGController
from game.flappy import Bird


class ModeSelectScreen(QtWidgets.QWidget):
    """Splash screen shown before the game. Emits mode_selected(int) on choice."""

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

        btn_hand = QtWidgets.QPushButton("Hand Clench")
        btn_hand.setFixedSize(220, 64)
        btn_hand.clicked.connect(lambda: self.mode_selected.emit(3))
        layout.addWidget(btn_hand, alignment=QtCore.Qt.AlignCenter)

        hint = QtWidgets.QLabel(
            "Blink to flap  ·  Jaw clench to flap  ·  Hand clench to flap"
        )
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
        self.controller = EEGController(self.mode)
        self.bird = Bird(y=300)
        self.setWindowTitle(self.controller.window_title)

        self._setup_game_ui()
        self.label.setText(self.controller.status_text)

        self.eeg_timer = QtCore.QTimer()
        self.eeg_timer.timeout.connect(self.update_loop)
        self.eeg_timer.start(20)

        self.dot_timer = QtCore.QTimer()
        self.dot_timer.timeout.connect(self._tick_dots)
        self.dot_timer.start(500)

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

    def _load_pixmap(self, filename, fallback_size, fallback_color):
        path = self._asset_path(filename)
        pix = QtGui.QPixmap(str(path))
        if not pix.isNull():
            return pix
        fallback = QtGui.QPixmap(fallback_size[0], fallback_size[1])
        fallback.fill(fallback_color)
        return fallback

    def _setup_game_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)
        split = QtWidgets.QHBoxLayout()
        main_layout.addLayout(split)

        self.plot = pg.PlotWidget(title=f"EEG ({self.controller.plot_channel_name})")
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
                fn()
                break
            except Exception:
                pass

        for fn in [
            lambda: self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff),
            lambda: self.view.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            ),
        ]:
            try:
                fn()
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

    def _tick_dots(self):
        status_text = self.controller.tick_calibration_indicator()
        if status_text is not None:
            self.label.setText(status_text)

    def update_loop(self):
        update = self.controller.process_eeg()
        if update is None:
            return

        if not self.controller.calibrating and self.dot_timer.isActive():
            self.dot_timer.stop()

        self.curve.setData(update.rel_times, update.signal)

        if update.status_text:
            self.label.setText(update.status_text)

        if update.jump_now:
            self.bird.jump()

    def update_game(self):
        if self.controller.consume_held_jump():
            self.bird.jump()

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
