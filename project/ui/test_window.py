import json
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from controller.game_controller import GameController
from game.flappy import PIPE_WIDTH

# ── constants ─────────────────────────────────────────────────────────────────

ORANGE = "#F2A007"
_BG_CACHE: QtGui.QPixmap | None = None

_SCORE_FILE = Path(__file__).resolve().parents[1] / "high_score.json"


def _load_high_score() -> int:
    try:
        return int(json.loads(_SCORE_FILE.read_text()).get("high_score", 0))
    except Exception:
        return 0


def _save_high_score(score: int) -> None:
    try:
        _SCORE_FILE.write_text(json.dumps({"high_score": score}))
    except Exception:
        pass


# ── asset helpers ─────────────────────────────────────────────────────────────

def _asset_path(filename: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    for candidate in [
        root / "project" / "game" / filename,
        root / "mvp" / "src" / "game" / filename,
    ]:
        if candidate.exists():
            return candidate
    return root / "mvp" / "src" / "game" / filename


def _load_pixmap(filename: str, fallback_size=(360, 640), fallback_color=None) -> QtGui.QPixmap:
    pix = QtGui.QPixmap(str(_asset_path(filename)))
    if not pix.isNull():
        return pix
    fb = QtGui.QPixmap(*fallback_size)
    fb.fill(fallback_color or QtGui.QColor(120, 200, 255))
    return fb


def _bg_pixmap() -> QtGui.QPixmap:
    global _BG_CACHE
    if _BG_CACHE is None:
        _BG_CACHE = _load_pixmap("flappybirdbg.png")
    return _BG_CACHE


# ── fonts ─────────────────────────────────────────────────────────────────────

def _font_title(size: int = 52) -> QtGui.QFont:
    return QtGui.QFont("Comic Sans MS", size, QtGui.QFont.Bold)


def _font_body(size: int = 18, bold: bool = True) -> QtGui.QFont:
    f = QtGui.QFont("Comic Sans MS", size)
    f.setBold(bold)
    return f


# ── stroked label (fill color + white outline) ────────────────────────────────

class _StrokedLabel(QtWidgets.QWidget):
    """Draws multi-line text with a white stroke using QPainterPath."""

    def __init__(self, text: str, font: QtGui.QFont, fill_color: str,
                 stroke_width: int = 3, parent=None):
        super().__init__(parent)
        self._text = text
        self._font = font
        self._fill = QtGui.QColor(fill_color)
        self._stroke_width = stroke_width
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )

    def sizeHint(self):
        fm = QtGui.QFontMetrics(self._font)
        lines = self._text.split("\n")
        w = max(fm.horizontalAdvance(l) for l in lines) + self._stroke_width * 2 + 4
        h = fm.height() * len(lines) + fm.leading() * (len(lines) - 1) + self._stroke_width * 2 + 4
        return QtCore.QSize(w, h)

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.TextAntialiasing)

        fm = QtGui.QFontMetrics(self._font)
        lines = self._text.split("\n")
        line_h = fm.height()
        total_h = line_h * len(lines)
        y_start = (self.height() - total_h) // 2 + fm.ascent()

        for i, line in enumerate(lines):
            text_w = fm.horizontalAdvance(line)
            x = (self.width() - text_w) // 2
            y = y_start + i * line_h

            path = QtGui.QPainterPath()
            path.addText(x, y, self._font, line)

            pen = QtGui.QPen(QtGui.QColor("white"), self._stroke_width * 2,
                             QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
            p.setPen(pen)
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawPath(path)

            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(self._fill)
            p.drawPath(path)

        p.end()


# ── shared background base ────────────────────────────────────────────────────

class _BgWidget(QtWidgets.QWidget):
    """Stretches flappybirdbg.png to fill itself as a background."""

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.drawPixmap(self.rect(), _bg_pixmap())
        p.end()


# ── reusable menu item ────────────────────────────────────────────────────────

class _MenuItem(QtWidgets.QWidget):
    """Single menu row: a retro ▶ triangle (always occupying space) + label text."""

    clicked = QtCore.Signal()

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.setAttribute(QtCore.Qt.WA_Hover, True)

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 4, 0, 4)
        row.setSpacing(10)

        self._triangle = QtWidgets.QLabel("▶")
        self._triangle.setFont(_font_body(20))
        # Transparent by default — keeps width reserved so text never shifts.
        self._triangle.setStyleSheet("color: transparent; background: transparent;")
        row.addWidget(self._triangle)

        self._text = QtWidgets.QLabel(text)
        self._text.setFont(_font_body(20))
        self._text.setStyleSheet("color: #1A1A1A; background: transparent;")
        row.addWidget(self._text)

    def enterEvent(self, event):
        self._triangle.setStyleSheet("color: #1A1A1A; background: transparent;")
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._triangle.setStyleSheet("color: transparent; background: transparent;")
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


# ── screens ───────────────────────────────────────────────────────────────────

class HomeMenu(_BgWidget):
    mode_selected = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scroll_x = 0.0
        self._scroll_timer = QtCore.QTimer(self)
        self._scroll_timer.timeout.connect(self._tick_scroll)
        self._build_ui()

    def showEvent(self, event):
        self._scroll_timer.start(16)
        super().showEvent(event)

    def hideEvent(self, event):
        self._scroll_timer.stop()
        super().hideEvent(event)

    def _tick_scroll(self):
        w = self.width()
        if w > 0:
            self._scroll_x = (self._scroll_x + 0.5) % w
            self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        bg = _bg_pixmap()
        w, h = self.width(), self.height()
        if w == 0 or h == 0:
            return
        scaled = bg.scaled(w, h, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
        offset = int(self._scroll_x)
        p.drawPixmap(-offset, 0, scaled)
        p.drawPixmap(w - offset, 0, scaled)
        p.end()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignCenter)

        layout.addStretch(2)

        title = _StrokedLabel("BCI-Controlled\nFlappy Bird", _font_title(52), ORANGE, stroke_width=3)
        layout.addWidget(title, alignment=QtCore.Qt.AlignCenter)

        layout.addStretch(1)

        prompt = QtWidgets.QLabel("Choose Your BCI Detection Mode:")
        prompt.setAlignment(QtCore.Qt.AlignCenter)
        prompt.setFont(_font_body(18))
        prompt.setStyleSheet("color: #1A1A1A; background: transparent;")
        layout.addWidget(prompt)

        layout.addSpacing(8)

        for label, mode in [("Jaw Clench", 2), ("Fist Clench", 3), ("Eye Blink", 1)]:
            item = _MenuItem(label)
            item.clicked.connect(lambda m=mode: self.mode_selected.emit(m))
            layout.addWidget(item, alignment=QtCore.Qt.AlignCenter)

        layout.addStretch(2)


class CalibrationScreen(_BgWidget):
    def __init__(self, controller: GameController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._total = GameController.CALIBRATION_DURATION
        self._build_ui()
        self._ctrl.calibration_progress.connect(self._on_progress)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        layout.setSpacing(16)

        title = QtWidgets.QLabel("Calibrating now.")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setFont(_font_title(44))
        title.setStyleSheet(f"color: {ORANGE}; background: transparent;")
        layout.addWidget(title)

        mode_word = "blink" if self._ctrl.mode == 1 else "jaw clench"
        instr = QtWidgets.QLabel(
            f"{mode_word.capitalize()} for the next {int(self._total)} seconds."
        )
        instr.setAlignment(QtCore.Qt.AlignCenter)
        instr.setFont(_font_body(20))
        instr.setStyleSheet("color: white; background: transparent;")
        layout.addWidget(instr)

        self._countdown = QtWidgets.QLabel(f"{int(self._total)}s remaining")
        self._countdown.setAlignment(QtCore.Qt.AlignCenter)
        self._countdown.setFont(_font_body(28, bold=True))
        self._countdown.setStyleSheet("color: white; background: transparent;")
        layout.addWidget(self._countdown)

    def _on_progress(self, detected: int, elapsed: float, total: float):
        remaining = max(0, int(total - elapsed) + 1)
        self._countdown.setText(f"{remaining}s remaining")


_GAME_W, _GAME_H = 440, 871   # canonical game canvas dimensions


class PlayScreen(QtWidgets.QWidget):
    """Portrait game canvas only — EEG plot lives in MainWindow."""

    def __init__(self, controller: GameController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._event_count = 0
        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._scene = QtWidgets.QGraphicsScene(0, 0, _GAME_W, _GAME_H)
        self._scene.setSceneRect(0, 0, _GAME_W, _GAME_H)
        self._view = QtWidgets.QGraphicsView(self._scene)
        self._view.setFixedSize(_GAME_W, _GAME_H)
        self._view.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(120, 200, 255)))

        try:
            self._view.setRenderHint(QtGui.QPainter.Antialiasing, True)
        except Exception:
            pass

        for setter, attr in [
            (self._view.setHorizontalScrollBarPolicy, "ScrollBarAlwaysOff"),
            (self._view.setVerticalScrollBarPolicy, "ScrollBarAlwaysOff"),
        ]:
            v = getattr(QtCore.Qt, attr, None)
            if v is not None:
                try:
                    setter(v)
                except Exception:
                    pass

        try:
            self._view.setFrameShape(QtWidgets.QFrame.NoFrame)
        except Exception:
            try:
                self._view.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            except Exception:
                pass

        layout.addWidget(self._view)

        def _qt(*names):
            for n in names:
                v = getattr(QtCore.Qt, n, None)
                if v is not None:
                    return v

        ign = _qt("IgnoreAspectRatio", "AspectRatioMode.IgnoreAspectRatio")
        keep = _qt("KeepAspectRatio", "AspectRatioMode.KeepAspectRatio")
        smooth = _qt("SmoothTransformation", "TransformationMode.SmoothTransformation")

        bg_pix = _load_pixmap(
            "flappybirdbg.png", (_GAME_W, _GAME_H), QtGui.QColor(120, 200, 255)
        ).scaled(_GAME_W, _GAME_H, ign, smooth)
        bird_pix = _load_pixmap(
            "flappybird.png", (34, 24), QtGui.QColor(255, 220, 0)
        ).scaled(34, 24, keep, smooth)

        bg_item = QtWidgets.QGraphicsPixmapItem(bg_pix)
        bg_item.setZValue(-10)
        self._scene.addItem(bg_item)

        self._bird_item = QtWidgets.QGraphicsPixmapItem(bird_pix)
        self._bird_item.setOffset(-17, -12)
        self._bird_item.setZValue(10)
        self._scene.addItem(self._bird_item)
        self._bird_x = 80
        self._bird_item.setPos(self._bird_x, 300)

        # Pipe image pool — 3 pairs covers max simultaneous pipes on screen
        top_src = _load_pixmap("toppipe.png",    (PIPE_WIDTH, 400), QtGui.QColor(0, 150, 0))
        bot_src = _load_pixmap("bottompipe.png", (PIPE_WIDTH, 400), QtGui.QColor(0, 150, 0))
        self._top_src = top_src
        self._bot_src = bot_src
        self._pipe_pool: list[tuple] = []
        for _ in range(3):
            t = QtWidgets.QGraphicsPixmapItem(top_src)
            t.setZValue(5)
            t.setVisible(False)
            self._scene.addItem(t)
            b = QtWidgets.QGraphicsPixmapItem(bot_src)
            b.setZValue(5)
            b.setVisible(False)
            self._scene.addItem(b)
            self._pipe_pool.append((t, b))

    def _connect_signals(self):
        self._ctrl.detection_event.connect(self._on_detection_event)
        self._ctrl.bird_updated.connect(self._on_bird_updated)
        self._ctrl.pipes_updated.connect(self._on_pipes_updated)

    def _on_detection_event(self, count: int):
        self._event_count += count

    def _on_bird_updated(self, y: float, vel: float):
        angle = max(-30, min(60, vel * 3))
        self._bird_item.setRotation(angle)
        self._bird_item.setPos(self._bird_x, y)

    def _on_pipes_updated(self, pipes: list):
        src_w_top = self._top_src.width()
        src_h_top = self._top_src.height()
        src_w_bot = self._bot_src.width()
        src_h_bot = self._bot_src.height()

        for i, (top_item, bot_item) in enumerate(self._pipe_pool):
            if i < len(pipes):
                x, top_h, bot_y, bot_h = pipes[i]
                if top_h > 0 and src_h_top > 0:
                    top_item.setTransform(QtGui.QTransform.fromScale(
                        PIPE_WIDTH / src_w_top, top_h / src_h_top
                    ))
                    top_item.setPos(x, 0)
                    top_item.setVisible(True)
                else:
                    top_item.setVisible(False)
                if bot_h > 0 and src_h_bot > 0:
                    bot_item.setTransform(QtGui.QTransform.fromScale(
                        PIPE_WIDTH / src_w_bot, bot_h / src_h_bot
                    ))
                    bot_item.setPos(x, bot_y)
                    bot_item.setVisible(True)
                else:
                    bot_item.setVisible(False)
            else:
                top_item.setVisible(False)
                bot_item.setVisible(False)


class GameOverScreen(_BgWidget):
    restart = QtCore.Signal()
    go_home = QtCore.Signal()

    def __init__(self, score: int, high_score: int, parent=None):
        super().__init__(parent)
        self._score = score
        self._high_score = high_score
        self._build_ui()

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)

        # Score row: top-left aligned
        score_row = QtWidgets.QHBoxLayout()
        score_label = QtWidgets.QLabel(f"SCORE: {self._score}")
        score_label.setFont(_font_body(22))
        score_label.setStyleSheet("color: #1A1A1A; background: transparent;")
        score_row.addWidget(score_label)
        score_row.addStretch()
        outer.addLayout(score_row)

        outer.addStretch(1)

        high_label = QtWidgets.QLabel(f"HIGH SCORE: {self._high_score}")
        high_label.setAlignment(QtCore.Qt.AlignCenter)
        high_label.setFont(_font_body(22))
        high_label.setStyleSheet("color: #1A1A1A; background: transparent;")
        outer.addWidget(high_label)

        outer.addSpacing(12)

        game_over_label = _StrokedLabel("GAME OVER", _font_title(56), ORANGE, stroke_width=3)
        outer.addWidget(game_over_label, alignment=QtCore.Qt.AlignCenter)

        outer.addSpacing(24)

        play_again = QtWidgets.QLabel("Play Again?")
        play_again.setAlignment(QtCore.Qt.AlignCenter)
        play_again.setFont(_font_body(22))
        play_again.setStyleSheet("color: #1A1A1A; background: transparent;")
        outer.addWidget(play_again)

        outer.addSpacing(12)

        btn_col = QtWidgets.QVBoxLayout()
        btn_col.setAlignment(QtCore.Qt.AlignCenter)
        btn_col.setSpacing(4)

        for text, signal in [("Yes", self.restart), ("No", self.go_home)]:
            item = _MenuItem(text)
            item.clicked.connect(signal.emit)
            btn_col.addWidget(item, alignment=QtCore.Qt.AlignCenter)

        outer.addLayout(btn_col)
        outer.addStretch(2)


# ── main window ───────────────────────────────────────────────────────────────

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI Flappy Bird")

        self._high_score = _load_high_score()
        self._mode: int | None = None
        self._controller: GameController | None = None

        # Root layout: permanent EEG plot (left) | game area (right, fixed _GAME_W px)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._eeg_plot = pg.PlotWidget(title="EEG — Fp1")
        self._eeg_plot.setYRange(-150, 150)
        self._eeg_plot.showGrid(x=True, y=True, alpha=0.3)
        self._eeg_curve = self._eeg_plot.plot(pen="y")
        root.addWidget(self._eeg_plot, stretch=1)

        self._stack = QtWidgets.QStackedWidget()
        self._stack.setFixedSize(_GAME_W, _GAME_H)
        root.addWidget(self._stack)

        home = HomeMenu()
        home.mode_selected.connect(self._on_mode_selected)
        self._stack.addWidget(home)  # always index 0

        # Size the window so the content area exactly fits the game height.
        # resize() sets the outer frame; Qt subtracts title bar (~28 px on macOS).
        self.resize(820, _GAME_H + 40)
        self.setMinimumSize(500, _GAME_H + 40)

    def _make_stream(self):
        """Return a stream object, or None to let GameController use real EEGStream."""
        return None

    def _on_eeg_updated(self, times, signal):
        self._eeg_curve.setData(times, signal)

    def _cleanup_game_screens(self):
        """Stop the controller and remove all screens after HomeMenu."""
        if self._controller is not None:
            self._controller.stop()
            self._controller = None
        while self._stack.count() > 1:
            w = self._stack.widget(1)
            self._stack.removeWidget(w)
            w.deleteLater()

    def _on_mode_selected(self, mode: int):
        self._mode = mode
        self._cleanup_game_screens()

        self._controller = GameController(mode, parent=self, stream=self._make_stream())
        self._controller.eeg_updated.connect(self._on_eeg_updated)
        self._controller.calibration_done.connect(self._on_calibration_done)
        self._controller.game_over.connect(self._on_game_over)

        calib = CalibrationScreen(self._controller)
        self._stack.addWidget(calib)
        self._stack.setCurrentWidget(calib)
        self._controller.start()

    def _on_calibration_done(self):
        play = PlayScreen(self._controller)
        self._stack.addWidget(play)
        self._stack.setCurrentWidget(play)

    def _on_game_over(self, score: int):
        if score > self._high_score:
            self._high_score = score
            _save_high_score(score)
        over = GameOverScreen(score, self._high_score)
        over.restart.connect(self._on_restart)
        over.go_home.connect(self._on_go_home)
        self._stack.addWidget(over)
        self._stack.setCurrentWidget(over)

    def _on_restart(self):
        self._on_mode_selected(self._mode)

    def _on_go_home(self):
        self._cleanup_game_screens()
        self._stack.setCurrentIndex(0)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
