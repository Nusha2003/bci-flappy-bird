import json
import random
import time
from pathlib import Path

import sip
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from controller import EEGController
from game.flappy import Bird


ORANGE = "#F2A007"
PIPE_WIDTH = 60
_BG_CACHE: QtGui.QPixmap | None = None

_SCORE_FILE = Path(__file__).resolve().parents[1] / "high_score.json"

_GAME_W, _GAME_H = 360, 640


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


def _asset_path(filename: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    for candidate in [
        root / "project" / "game" / filename,
        root / "mvp" / "src" / "game" / filename,
    ]:
        if candidate.exists():
            return candidate
    return root / "mvp" / "src" / "game" / filename


def _load_pixmap(
    filename: str,
    fallback_size=(360, 640),
    fallback_color=None,
) -> QtGui.QPixmap:
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


def _font_title(size: int = 52) -> QtGui.QFont:
    return QtGui.QFont("Comic Sans MS", size, QtGui.QFont.Bold)


def _font_body(size: int = 18, bold: bool = True) -> QtGui.QFont:
    font = QtGui.QFont("Comic Sans MS", size)
    font.setBold(bold)
    return font


class _StrokedLabel(QtWidgets.QWidget):
    """Draw multi-line text with a white outline for the retro screens."""

    def __init__(
        self,
        text: str,
        font: QtGui.QFont,
        fill_color: str,
        stroke_width: int = 3,
        parent=None,
    ):
        super().__init__(parent)
        self._text = text
        self._font = font
        self._fill = QtGui.QColor(fill_color)
        self._stroke_width = stroke_width
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Preferred,
        )

    def sizeHint(self):
        fm = QtGui.QFontMetrics(self._font)
        lines = self._text.split("\n")
        width = max(fm.horizontalAdvance(line) for line in lines)
        width += self._stroke_width * 2 + 4
        height = fm.height() * len(lines)
        height += fm.leading() * (len(lines) - 1) + self._stroke_width * 2 + 4
        return QtCore.QSize(width, height)

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)

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

            pen = QtGui.QPen(
                QtGui.QColor("white"),
                self._stroke_width * 2,
                QtCore.Qt.SolidLine,
                QtCore.Qt.RoundCap,
                QtCore.Qt.RoundJoin,
            )
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawPath(path)

            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(self._fill)
            painter.drawPath(path)

        painter.end()


class _BgWidget(QtWidgets.QWidget):
    """Fill the widget using the Flappy background cropped to preserve aspect."""

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        bg = _bg_pixmap()
        w, h = self.width(), self.height()
        scaled = bg.scaled(
            w,
            h,
            QtCore.Qt.KeepAspectRatioByExpanding,
            QtCore.Qt.SmoothTransformation,
        )
        x_off = (scaled.width() - w) // 2
        y_off = (scaled.height() - h) // 2
        painter.drawPixmap(-x_off, -y_off, scaled)
        painter.end()


class _MenuItem(QtWidgets.QWidget):
    """Single retro menu row with a hover triangle."""

    clicked = QtCore.pyqtSignal()

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.setAttribute(QtCore.Qt.WA_Hover, True)

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 4, 0, 4)
        row.setSpacing(10)

        self._triangle = QtWidgets.QLabel("▶")
        self._triangle.setFont(_font_body(20))
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


class ModeSelectScreen(QtWidgets.QWidget):
    """Splash screen shown before the game. Emits mode_selected(int) on choice."""

    mode_selected = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scroll_x = 0.0
        self._cached_sw = 1
        self._scroll_timer = QtCore.QTimer(self)
        self._scroll_timer.timeout.connect(self._tick_scroll)
        self._build_ui()

    def showEvent(self, event):
        self._scroll_timer.start(16)
        super().showEvent(event)

    def hideEvent(self, event):
        self._scroll_timer.stop()
        super().hideEvent(event)

    def resizeEvent(self, event):
        w, h = self.width(), self.height()
        if w > 0 and h > 0:
            self._cached_sw = _bg_pixmap().scaled(
                w,
                h,
                QtCore.Qt.KeepAspectRatioByExpanding,
                QtCore.Qt.SmoothTransformation,
            ).width()
        super().resizeEvent(event)

    def _tick_scroll(self):
        if self._cached_sw > 1:
            self._scroll_x = (self._scroll_x + 0.5) % self._cached_sw
            self.update()

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        bg = _bg_pixmap()
        w, h = self.width(), self.height()
        if w == 0 or h == 0:
            return
        scaled = bg.scaled(
            w,
            h,
            QtCore.Qt.KeepAspectRatioByExpanding,
            QtCore.Qt.SmoothTransformation,
        )
        sw = scaled.width()
        offset = int(self._scroll_x)
        painter.drawPixmap(-offset, 0, scaled)
        painter.drawPixmap(sw - offset, 0, scaled)
        painter.end()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignCenter)

        layout.addStretch(2)

        title = _StrokedLabel(
            "BCI-Controlled\nFlappy Bird",
            _font_title(52),
            ORANGE,
            stroke_width=3,
        )
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
            item.clicked.connect(lambda _checked=False, m=mode: self.mode_selected.emit(m))
            layout.addWidget(item, alignment=QtCore.Qt.AlignCenter)

        layout.addStretch(2)


class CalibrationScreen(_BgWidget):
    def __init__(self, controller: EEGController, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self._ctrl = controller
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QtGui.QColor("#10161f"))
        self.setPalette(palette)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(18)

        badge = QtWidgets.QLabel("Calibration")
        badge.setAlignment(QtCore.Qt.AlignCenter)
        badge.setFont(_font_body(12, bold=True))
        badge.setStyleSheet(
            f"color: {ORANGE}; background: rgba(242, 160, 7, 28);"
            "padding: 6px 12px; border-radius: 12px;"
        )
        badge.setMaximumWidth(140)
        layout.addWidget(badge, alignment=QtCore.Qt.AlignCenter)

        if self._ctrl.mode == 1:
            headline = "Eye Blink Calibration"
            detail = "Blink naturally when prompted and keep your face relaxed."
        elif self._ctrl.mode == 2:
            headline = "Jaw Clench Calibration"
            detail = (
                "Clench only when prompted. Relax your jaw and avoid blinking "
                "or extra facial movement between trials."
            )
        else:
            headline = "Hand Clench Calibration"
            detail = "Follow the REST and CLENCH prompts and stay still otherwise."

        headline_label = QtWidgets.QLabel(headline)
        headline_label.setAlignment(QtCore.Qt.AlignCenter)
        headline_label.setWordWrap(True)
        headline_label.setFont(_font_body(19, bold=True))
        headline_label.setStyleSheet("color: white; background: transparent;")
        layout.addWidget(headline_label)

        instr = QtWidgets.QLabel(detail)
        instr.setAlignment(QtCore.Qt.AlignCenter)
        instr.setWordWrap(True)
        instr.setFont(_font_body(13))
        instr.setStyleSheet(
            "color: #d5dde8; background: rgba(255, 255, 255, 18); "
            "padding: 14px; border-radius: 14px;"
        )
        layout.addWidget(instr)

        self._status = QtWidgets.QLabel(self._ctrl.status_text)
        self._status.setAlignment(QtCore.Qt.AlignCenter)
        self._status.setWordWrap(True)
        self._status.setFont(_font_body(14, bold=True))
        self._status.setStyleSheet(
            "color: white; background: rgba(255, 255, 255, 24); "
            "padding: 16px; border-radius: 16px;"
        )
        layout.addWidget(self._status)

        self._countdown = QtWidgets.QLabel("")
        self._countdown.setAlignment(QtCore.Qt.AlignCenter)
        self._countdown.setFont(_font_body(16, bold=True))
        self._countdown.setStyleSheet(
            f"color: {ORANGE}; background: rgba(242, 160, 7, 22); "
            "padding: 10px 14px; border-radius: 12px;"
        )
        layout.addWidget(self._countdown, alignment=QtCore.Qt.AlignCenter)
        layout.addStretch(1)

    def paintEvent(self, event):
        del event
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#10161f"))
        painter.end()

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_remaining(self, remaining: int | None) -> None:
        if remaining is None:
            self._countdown.setText("")
        else:
            self._countdown.setText(f"{remaining}s remaining")


class PlayScreen(QtWidgets.QWidget):
    """Portrait game canvas only — EEG plot lives in MainWindow."""

    def __init__(self, controller: EEGController, parent=None):
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
            value = getattr(QtCore.Qt, attr, None)
            if value is not None:
                try:
                    setter(value)
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
            for name in names:
                value = getattr(QtCore.Qt, name, None)
                if value is not None:
                    return value

        ign = _qt("IgnoreAspectRatio", "AspectRatioMode.IgnoreAspectRatio")
        keep = _qt("KeepAspectRatio", "AspectRatioMode.KeepAspectRatio")
        smooth = _qt("SmoothTransformation", "TransformationMode.SmoothTransformation")

        bg_pix = _load_pixmap(
            "flappybirdbg.png",
            (_GAME_W, _GAME_H),
            QtGui.QColor(120, 200, 255),
        ).scaled(_GAME_W, _GAME_H, ign, smooth)
        bird_pix = _load_pixmap(
            "flappybird.png",
            (34, 24),
            QtGui.QColor(255, 220, 0),
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

        top_src = _load_pixmap(
            "toppipe.png",
            (PIPE_WIDTH, 400),
            QtGui.QColor(0, 150, 0),
        )
        bot_src = _load_pixmap(
            "bottompipe.png",
            (PIPE_WIDTH, 400),
            QtGui.QColor(0, 150, 0),
        )
        self._top_src = top_src
        self._bot_src = bot_src
        self._pipe_pool: list[tuple] = []
        for _ in range(3):
            top_item = QtWidgets.QGraphicsPixmapItem(top_src)
            top_item.setZValue(5)
            top_item.setVisible(False)
            self._scene.addItem(top_item)

            bot_item = QtWidgets.QGraphicsPixmapItem(bot_src)
            bot_item.setZValue(5)
            bot_item.setVisible(False)
            self._scene.addItem(bot_item)

            self._pipe_pool.append((top_item, bot_item))

    def _connect_signals(self):
        try:
            self._ctrl.detection_event.connect(self._on_detection_event)
            self._ctrl.bird_updated.connect(self._on_bird_updated)
            self._ctrl.pipes_updated.connect(self._on_pipes_updated)
        except AttributeError:
            pass

    def update_bird(self, y: float, vel: float) -> None:
        angle = max(-30, min(60, vel * 3))
        self._bird_item.setRotation(angle)
        self._bird_item.setPos(self._bird_x, y)

    def update_pipes(self, pipes: list) -> None:
        self._on_pipes_updated(pipes)

    def _on_detection_event(self, count: int):
        self._event_count += count

    def _on_bird_updated(self, y: float, vel: float):
        self.update_bird(y, vel)

    def _on_pipes_updated(self, pipes: list):
        src_w_top = self._top_src.width()
        src_h_top = self._top_src.height()
        src_w_bot = self._bot_src.width()
        src_h_bot = self._bot_src.height()

        for i, (top_item, bot_item) in enumerate(self._pipe_pool):
            if i < len(pipes):
                x, top_h, bot_y, bot_h = pipes[i]

                if top_h > 0 and src_h_top > 0:
                    top_item.setTransform(
                        QtGui.QTransform.fromScale(
                            PIPE_WIDTH / src_w_top,
                            top_h / src_h_top,
                        )
                    )
                    top_item.setPos(x, 0)
                    top_item.setVisible(True)
                else:
                    top_item.setVisible(False)

                if bot_h > 0 and src_h_bot > 0:
                    bot_item.setTransform(
                        QtGui.QTransform.fromScale(
                            PIPE_WIDTH / src_w_bot,
                            bot_h / src_h_bot,
                        )
                    )
                    bot_item.setPos(x, bot_y)
                    bot_item.setVisible(True)
                else:
                    bot_item.setVisible(False)
            else:
                top_item.setVisible(False)
                bot_item.setVisible(False)


class GameOverScreen(_BgWidget):
    restart = QtCore.pyqtSignal()
    go_home = QtCore.pyqtSignal()

    def __init__(self, score: int, high_score: int, parent=None):
        super().__init__(parent)
        self._score = score
        self._high_score = high_score
        self._build_ui()

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)

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

        game_over_label = _StrokedLabel(
            "GAME OVER",
            _font_title(56),
            ORANGE,
            stroke_width=3,
        )
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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI Flappy Bird")
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._high_score = _load_high_score()

        self._mode = None
        self._paused = False
        self._bird_tuning: dict[int, dict[str, float]] = {}
        self._eeg_timer = QtCore.QTimer(self)
        self._game_timer = QtCore.QTimer(self)
        self._dot_timer = QtCore.QTimer(self)
        self._dot_timer.timeout.connect(self._dot_tick)

        self._show_home()

    def _show_home(self):
        self._stop_timers()
        self._paused = False
        self._status_label = None
        self._pause_label = None
        self._calib_screen = None
        self._calib_status_label = None
        self._calib_countdown_label = None
        self.resize(_GAME_W, _GAME_H + 40)
        screen = ModeSelectScreen()
        screen.setFocusPolicy(QtCore.Qt.NoFocus)
        screen.mode_selected.connect(self._on_mode_selected)
        self.setCentralWidget(screen)
        self.setFocus()

    def _on_mode_selected(self, mode: int):
        self._mode = mode
        if mode not in self._bird_tuning:
            self._bird_tuning[mode] = self._default_bird_physics(mode)
        self._show_calibration()

    def _default_bird_physics(self, mode: int) -> dict[str, float]:
        if mode == 1:
            return {"gravity": 0.46, "jump": 9.5}
        if mode == 2:
            return {"gravity": 0.52, "jump": 12.0}
        return {"gravity": 0.5, "jump": 11.0}

    def _bird_physics(self) -> dict[str, float]:
        if self._mode not in self._bird_tuning:
            self._bird_tuning[self._mode] = self._default_bird_physics(self._mode)
        return dict(self._bird_tuning[self._mode])

    def _build_physics_controls(self) -> QtWidgets.QWidget:
        physics = self._bird_physics()

        panel = QtWidgets.QFrame()
        panel.setStyleSheet(
            "QFrame { background: rgba(16, 22, 31, 235); border-radius: 14px; }"
            "QLabel { color: white; }"
        )
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("Bird Tuning")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 15px; font-weight: bold;")
        layout.addWidget(title)

        self._gravity_value_label = QtWidgets.QLabel("")
        self._jump_value_label = QtWidgets.QLabel("")

        gravity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        gravity_slider.setRange(20, 100)
        gravity_slider.setValue(int(round(physics["gravity"] * 100)))
        gravity_slider.valueChanged.connect(self._on_gravity_changed)

        jump_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        jump_slider.setRange(50, 180)
        jump_slider.setValue(int(round(physics["jump"] * 10)))
        jump_slider.valueChanged.connect(self._on_jump_changed)

        layout.addWidget(self._gravity_value_label)
        layout.addWidget(gravity_slider)
        layout.addWidget(self._jump_value_label)
        layout.addWidget(jump_slider)

        self._gravity_slider = gravity_slider
        self._jump_slider = jump_slider
        self._refresh_physics_labels()
        return panel

    def _refresh_physics_labels(self):
        if not hasattr(self, "_gravity_value_label") or self._gravity_value_label is None:
            return
        physics = self._bird_physics()
        self._gravity_value_label.setText(f"Gravity: {physics['gravity']:.2f}")
        self._jump_value_label.setText(f"Jump: {physics['jump']:.1f}")

    def _apply_bird_physics_to_live_birds(self):
        physics = self._bird_physics()
        if hasattr(self, "_bird") and self._bird is not None:
            self._bird.gravity = physics["gravity"]
            self._bird.jump_strength = physics["jump"]
        if hasattr(self, "_calib_bird") and self._calib_bird is not None:
            self._calib_bird.gravity = physics["gravity"]
            self._calib_bird.jump_strength = physics["jump"]

    def _on_gravity_changed(self, value: int):
        self._bird_tuning[self._mode]["gravity"] = value / 100.0
        self._refresh_physics_labels()
        self._apply_bird_physics_to_live_birds()

    def _on_jump_changed(self, value: int):
        self._bird_tuning[self._mode]["jump"] = value / 10.0
        self._refresh_physics_labels()
        self._apply_bird_physics_to_live_birds()

    def _show_calibration(self):
        self._stop_timers()
        self._paused = False
        self._status_label = None
        self._pause_label = None
        self.controller = EEGController(self._mode)
        self.setWindowTitle(self.controller.window_title)
        self.resize(_GAME_W * 2 + 20, _GAME_H + 40)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(18)

        self._eeg_plot = pg.PlotWidget(
            title=f"EEG ({self.controller.plot_channel_name})"
        )
        self._eeg_plot.setYRange(-150, 150)
        self._eeg_plot.showGrid(x=True, y=True, alpha=0.3)
        self._eeg_curve = self._eeg_plot.plot(pen="y")
        self._eeg_plot.setMinimumWidth(420)
        layout.addWidget(self._eeg_plot)

        self._calib_status_label = None
        self._calib_countdown_label = None
        self._calib_bird = None
        self._calib_practice_screen = None
        self._gravity_slider = None
        self._jump_slider = None
        self._gravity_value_label = None
        self._jump_value_label = None

        if self.controller.mode == 1:
            self._last_calib_detected = int(getattr(self.controller, "calib_detected", 0))
            self._calib_bird = Bird(y=_GAME_H // 2, **self._bird_physics())
            self._calib_practice_screen = PlayScreen(self.controller)
            self._calib_practice_screen.setMinimumWidth(420)
            self._calib_practice_screen.setMaximumWidth(520)

            right_panel = QtWidgets.QWidget()
            right_layout = QtWidgets.QVBoxLayout(right_panel)
            right_layout.setContentsMargins(0, 0, 0, 0)
            right_layout.setSpacing(14)
            right_layout.addWidget(self._calib_practice_screen)

            hint = QtWidgets.QLabel("Eye Blink Calibration")
            hint.setAlignment(QtCore.Qt.AlignCenter)
            hint.setStyleSheet("font-size: 18px; font-weight: bold; padding: 2px; color: white;")
            right_layout.addWidget(hint, alignment=QtCore.Qt.AlignCenter)

            detail = QtWidgets.QLabel(
                "Practice blinking to flap while calibration runs. "
                "Keep your jaw relaxed between blinks."
            )
            detail.setAlignment(QtCore.Qt.AlignCenter)
            detail.setWordWrap(True)
            detail.setStyleSheet(
                "font-size: 13px; color: #d5dde8; background: rgba(16, 22, 31, 235); "
                "padding: 14px; border-radius: 14px;"
            )
            right_layout.addWidget(detail)

            self._calib_status_label = QtWidgets.QLabel(self.controller.status_text)
            self._calib_status_label.setAlignment(QtCore.Qt.AlignCenter)
            self._calib_status_label.setWordWrap(True)
            self._calib_status_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: white; "
                "background: rgba(16, 22, 31, 245); padding: 16px; border-radius: 16px;"
            )
            right_layout.addWidget(self._calib_status_label)

            self._calib_countdown_label = QtWidgets.QLabel("")
            self._calib_countdown_label.setAlignment(QtCore.Qt.AlignCenter)
            self._calib_countdown_label.setStyleSheet(
                f"font-size: 16px; font-weight: bold; color: {ORANGE}; "
                "background: rgba(242, 160, 7, 22); padding: 10px 14px; border-radius: 12px;"
            )
            right_layout.addWidget(self._calib_countdown_label, alignment=QtCore.Qt.AlignCenter)
            right_layout.addWidget(self._build_physics_controls())

            layout.addWidget(right_panel)
        else:
            self._calib_screen = CalibrationScreen(self.controller)
            self._calib_screen.setMinimumWidth(420)
            layout.addWidget(self._calib_screen)
            layout.itemAt(1).widget().layout().addWidget(self._build_physics_controls())

        layout.setStretch(0, 1)
        layout.setStretch(1, 1)

        self.setCentralWidget(container)
        self.setFocus()

        self._calib_start = time.monotonic()
        self._eeg_timer = QtCore.QTimer(self)
        self._eeg_timer.timeout.connect(self._calibration_tick)
        self._eeg_timer.start(20)
        self._dot_timer.start(500)

        if self.controller.mode == 1:
            self._game_timer = QtCore.QTimer(self)
            self._game_timer.timeout.connect(self._calibration_game_tick)
            self._game_timer.start(16)

    def _show_play(self):
        self._stop_timers()
        self._paused = False
        self._calib_screen = None
        self._calib_status_label = None
        self._calib_countdown_label = None
        self.resize(_GAME_W * 2 + 20, _GAME_H + 40)
        self._bird = Bird(y=_GAME_H // 2, **self._bird_physics())
        self._score = 0

        self._play_screen = PlayScreen(self.controller)
        self._pipes: list[dict] = []
        self._pipe_timer = 0
        self._first_pipe_spawn_interval = 18
        self._pipe_spawn_interval = 90
        self._pipe_speed = 3.0
        self._pipe_gap = 200

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        top_row = QtWidgets.QHBoxLayout()

        self._eeg_plot = pg.PlotWidget(
            title=f"EEG ({self.controller.plot_channel_name})"
        )
        self._eeg_plot.setYRange(-150, 150)
        self._eeg_plot.showGrid(x=True, y=True, alpha=0.3)
        self._eeg_curve = self._eeg_plot.plot(pen="y")
        top_row.addWidget(self._eeg_plot)
        top_row.addWidget(self._play_screen)
        layout.addLayout(top_row)

        self._status_label = QtWidgets.QLabel("")
        self._status_label.setStyleSheet("font-size: 14px; padding: 4px;")
        layout.addWidget(self._status_label)

        self._pause_label = QtWidgets.QLabel("Paused  |  Press SPACE to resume")
        self._pause_label.setAlignment(QtCore.Qt.AlignCenter)
        self._pause_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: white; "
            "background: rgba(16, 22, 31, 235); padding: 10px; border-radius: 12px;"
        )
        self._pause_label.hide()
        layout.addWidget(self._pause_label)

        self._refresh_status_label()
        layout.addWidget(self._build_physics_controls())

        self.setCentralWidget(container)
        self.setFocus()

        self._eeg_timer = QtCore.QTimer(self)
        self._eeg_timer.timeout.connect(self._eeg_tick)
        self._eeg_timer.start(20)

        self._game_timer = QtCore.QTimer(self)
        self._game_timer.timeout.connect(self._game_tick)
        self._game_timer.start(16)

    def _show_game_over(self, score: int):
        self._stop_timers()
        self._paused = False
        self._status_label = None
        self._pause_label = None
        self._calib_screen = None
        self._calib_status_label = None
        self._calib_countdown_label = None
        self.resize(_GAME_W, _GAME_H + 40)
        if score > self._high_score:
            self._high_score = score
            _save_high_score(score)
        screen = GameOverScreen(score, self._high_score)
        screen.restart.connect(self._play_again_same_player)
        screen.go_home.connect(self._show_home)
        self.setCentralWidget(screen)
        self.setFocus()

    def _play_again_same_player(self):
        if hasattr(self, "controller") and self.controller is not None:
            self._show_play()
            return
        self._show_calibration()

    def _calibration_tick(self):
        update = self.controller.process_eeg()

        if update is not None:
            self._eeg_curve.setData(update.rel_times, update.signal)
            if update.status_text and self.controller.mode == 1 and self._calib_status_label is not None:
                self._calib_status_label.setText(update.status_text)
            elif update.status_text:
                self._calib_screen.set_status(update.status_text)

        if self.controller.mode == 1:
            elapsed = time.monotonic() - self._calib_start
            remaining = max(0, int(self.controller.calibration_duration - elapsed))
            if self._calib_countdown_label is not None:
                self._calib_countdown_label.setText(f"{remaining}s remaining")

            detected = int(getattr(self.controller, "calib_detected", 0))
            if detected > getattr(self, "_last_calib_detected", 0) and self._calib_bird is not None:
                self._calib_bird.jump()
            self._last_calib_detected = detected
        else:
            self._calib_screen.set_remaining(None)

        if not self.controller.calibrating:
            self._show_play()

    def _calibration_game_tick(self):
        if self.controller.mode != 1 or self._calib_bird is None:
            return

        self._calib_bird.update()
        self._calib_bird.y = max(0, min(_GAME_H, self._calib_bird.y))

        if self._calib_practice_screen is not None:
            self._calib_practice_screen.update_bird(
                self._calib_bird.y,
                self._calib_bird.vel,
            )

    def _eeg_tick(self):
        update = self.controller.process_eeg()
        if update is None:
            return

        self._eeg_curve.setData(update.rel_times, update.signal)
        self._refresh_status_label(update.status_text)

        if update.jump_now and not self._paused:
            self._bird.jump()

    def _game_tick(self):
        if self._paused:
            return

        if self.controller.consume_held_jump():
            self._bird.jump()

        self._bird.update()
        self._bird.y = max(0, min(_GAME_H, self._bird.y))
        self._play_screen.update_bird(self._bird.y, self._bird.vel)

        self._pipe_timer += 1
        spawn_interval = (
            self._first_pipe_spawn_interval if not self._pipes else self._pipe_spawn_interval
        )
        if self._pipe_timer >= spawn_interval:
            self._pipe_timer = 0
            gap_top = random.randint(80, _GAME_H - 80 - self._pipe_gap)
            self._pipes.append({
                "x": float(_GAME_W),
                "top_h": gap_top,
                "bot_y": gap_top + self._pipe_gap,
                "scored": False,
            })

        for pipe in self._pipes:
            pipe["x"] -= self._pipe_speed

        for pipe in self._pipes:
            if not pipe["scored"] and pipe["x"] + PIPE_WIDTH < 80:
                pipe["scored"] = True
                self._score += 1
                self._refresh_status_label()

        bx, by = 80, self._bird.y
        for pipe in self._pipes:
            if bx + 17 > pipe["x"] and bx - 17 < pipe["x"] + PIPE_WIDTH:
                if by - 12 < pipe["top_h"] or by + 12 > pipe["bot_y"]:
                    self._show_game_over(self._score)
                    return

        self._pipes = [pipe for pipe in self._pipes if pipe["x"] > -PIPE_WIDTH]

        pipe_data = [
            (pipe["x"], pipe["top_h"], pipe["bot_y"], _GAME_H - pipe["bot_y"])
            for pipe in self._pipes
        ]
        self._play_screen.update_pipes(pipe_data)

        if self._bird.y >= _GAME_H:
            self._show_game_over(self._score)

    def _dot_tick(self):
        if not hasattr(self, "controller") or not self.controller.calibrating:
            return

        status_text = self.controller.tick_calibration_indicator()
        if not status_text:
            return

        if getattr(self, "_calib_screen", None) is not None:
            self._calib_screen.set_status(status_text)
        elif getattr(self, "_calib_status_label", None) is not None:
            self._calib_status_label.setText(status_text)
        if hasattr(self, "_status_label"):
            self._refresh_status_label(status_text)

    def _refresh_status_label(self, controller_text: str | None = None):
        if not hasattr(self, "_status_label") or self._status_label is None:
            return
        if sip.isdeleted(self._status_label):
            self._status_label = None
            return

        parts = [f"Score: {getattr(self, '_score', 0)}"]
        if self._paused:
            parts.append("PAUSED")
        if controller_text is None and hasattr(self, "controller"):
            controller_text = self.controller.status_text
        if controller_text:
            parts.append(controller_text)
        try:
            self._status_label.setText("  |  ".join(parts))
        except RuntimeError:
            self._status_label = None

    def _toggle_pause(self):
        if not hasattr(self, "controller") or self.controller is None:
            return
        if self.controller.calibrating or not hasattr(self, "_bird"):
            return

        self._paused = not self._paused
        if getattr(self, "_pause_label", None) is not None:
            self._pause_label.setVisible(self._paused)
        self._refresh_status_label()

    def _stop_timers(self):
        for timer in (self._eeg_timer, self._game_timer, self._dot_timer):
            if timer.isActive():
                timer.stop()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            self._toggle_pause()
            event.accept()
            return
        super().keyPressEvent(event)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
