import json
import random
import time
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# PySide2/PySide6 use Signal; PyQt5/PyQt6 use pyqtSignal
try:
    _Signal = QtCore.pyqtSignal
except AttributeError:
    _Signal = QtCore.Signal

from controller import EEGController
from game.flappy import Bird


ORANGE = "#F2A007"
PIPE_WIDTH = 60
_BG_CACHE: QtGui.QPixmap | None = None

_SCORE_FILE = Path(__file__).resolve().parents[1] / "high_score.json"

_GAME_W, _GAME_H = 420, 640
_EEG_W = _GAME_W      # EEG plot matches game canvas width (1:1)


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

    clicked = _Signal()

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

    mode_selected = _Signal(int)

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
        super().__init__(parent)
        self._ctrl = controller
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        layout.setSpacing(16)

        title = _StrokedLabel("Calibrating now.", _font_title(44), ORANGE, stroke_width=3)
        layout.addWidget(title, alignment=QtCore.Qt.AlignCenter)

        if self._ctrl.mode == 1:
            mode_word = "blink"
        elif self._ctrl.mode == 2:
            mode_word = "jaw clench"
        else:
            mode_word = "hand clench"

        instr = QtWidgets.QLabel(
            f"Follow the prompt for {mode_word} calibration."
        )
        instr.setAlignment(QtCore.Qt.AlignCenter)
        instr.setFont(_font_body(17))
        instr.setStyleSheet("color: white; background: transparent;")
        layout.addWidget(instr)

        self._status = QtWidgets.QLabel(self._ctrl.status_text)
        self._status.setAlignment(QtCore.Qt.AlignCenter)
        self._status.setWordWrap(True)
        self._status.setFont(_font_body(18, bold=True))
        self._status.setStyleSheet("color: white; background: transparent;")
        layout.addWidget(self._status)

        self._countdown = QtWidgets.QLabel("")
        self._countdown.setAlignment(QtCore.Qt.AlignCenter)
        self._countdown.setFont(_font_body(24, bold=True))
        self._countdown.setStyleSheet("color: white; background: transparent;")
        layout.addWidget(self._countdown)

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
    restart = _Signal()
    go_home = _Signal()

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
        self._high_score = _load_high_score()
        self._mode = None

        self._eeg_timer = QtCore.QTimer(self)
        self._game_timer = QtCore.QTimer(self)
        self._dot_timer = QtCore.QTimer(self)
        self._dot_timer.timeout.connect(self._dot_tick)

        self.resize(_EEG_W + _GAME_W + 20, _GAME_H + 40)

        root = QtWidgets.QWidget()
        root_layout = QtWidgets.QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._eeg_plot = pg.PlotWidget(title="EEG")
        self._eeg_plot.setFixedWidth(_EEG_W)
        self._eeg_plot.setYRange(-150, 150)
        self._eeg_plot.showGrid(x=True, y=True, alpha=0.3)
        self._eeg_curve = self._eeg_plot.plot(pen="y")
        root_layout.addWidget(self._eeg_plot)

        self._right_stack = QtWidgets.QStackedWidget()
        self._right_stack.setFixedWidth(_GAME_W)
        root_layout.addWidget(self._right_stack)

        self.setCentralWidget(root)
        self._show_home()

    def _set_right_widget(self, widget):
        idx = self._right_stack.addWidget(widget)
        self._right_stack.setCurrentIndex(idx)
        while self._right_stack.count() > 1:
            old = self._right_stack.widget(0)
            self._right_stack.removeWidget(old)
            old.setParent(None)

    def _show_home(self):
        self._stop_timers()
        self._eeg_plot.setTitle("EEG")
        screen = ModeSelectScreen()
        screen.mode_selected.connect(self._on_mode_selected)
        self._set_right_widget(screen)

    def _on_mode_selected(self, mode: int):
        self._mode = mode
        self._show_calibration()

    def _show_calibration(self):
        self._stop_timers()
        self.controller = EEGController(self._mode)
        self.setWindowTitle(self.controller.window_title)
        self._eeg_plot.setTitle(f"EEG ({self.controller.plot_channel_name})")

        self._calib_screen = CalibrationScreen(self.controller)
        self._set_right_widget(self._calib_screen)

        self._calib_start = time.monotonic()
        self._eeg_timer = QtCore.QTimer(self)
        self._eeg_timer.timeout.connect(self._calibration_tick)
        self._eeg_timer.start(20)
        self._dot_timer.start(500)

    def _show_play(self):
        self._stop_timers()
        self._bird = Bird(y=_GAME_H // 2)
        self._score = 0

        self._play_screen = PlayScreen(self.controller)
        self._pipes: list[dict] = []
        self._pipe_timer = 0
        self._pipe_spawn_interval = 90
        self._pipe_speed = 3.0
        self._pipe_gap = 200
        self._game_start = time.monotonic()

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._play_screen)

        self._status_label = QtWidgets.QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("font-size: 14px; padding: 4px;")
        layout.addWidget(self._status_label)

        self._set_right_widget(container)
        self._refresh_status_label()

        self._eeg_timer = QtCore.QTimer(self)
        self._eeg_timer.timeout.connect(self._eeg_tick)
        self._eeg_timer.start(20)

        self._game_timer = QtCore.QTimer(self)
        self._game_timer.timeout.connect(self._game_tick)
        self._game_timer.start(16)

    def _show_game_over(self, score: int):
        self._stop_timers()
        if score > self._high_score:
            self._high_score = score
            _save_high_score(score)
        screen = GameOverScreen(score, self._high_score)
        screen.restart.connect(self._show_play)
        screen.go_home.connect(self._show_home)
        self._set_right_widget(screen)

    def _calibration_tick(self):
        update = self.controller.process_eeg()

        if update is not None:
            self._eeg_curve.setData(update.rel_times, update.signal)
            if update.status_text:
                self._calib_screen.set_status(update.status_text)

        if self.controller.mode == 1:
            elapsed = time.monotonic() - self._calib_start
            remaining = max(0, int(self.controller.calibration_duration - elapsed))
            self._calib_screen.set_remaining(remaining)
        else:
            self._calib_screen.set_remaining(None)

        if not self.controller.calibrating:
            self._show_play()

    def _eeg_tick(self):
        update = self.controller.process_eeg()
        if update is None:
            return

        self._eeg_curve.setData(update.rel_times, update.signal)
        self._refresh_status_label(update.status_text)

        if update.jump_now:
            self._bird.jump()

    def _game_tick(self):
        if self.controller.consume_held_jump():
            self._bird.jump()

        self._bird.update()
        self._bird.y = max(0, min(_GAME_H, self._bird.y))
        self._play_screen.update_bird(self._bird.y, self._bird.vel)

        if time.monotonic() - self._game_start >= 3.0:
            self._pipe_timer += 1
        if self._pipe_timer >= self._pipe_spawn_interval:
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

        if hasattr(self, "_calib_screen"):
            self._calib_screen.set_status(status_text)
        if hasattr(self, "_status_label"):
            self._refresh_status_label(status_text)

    def _refresh_status_label(self, controller_text: str | None = None):
        if not hasattr(self, "_status_label"):
            return

        parts = [f"Score: {getattr(self, '_score', 0)}"]
        if controller_text is None and hasattr(self, "controller"):
            controller_text = self.controller.status_text
        if controller_text:
            parts.append(controller_text)
        self._status_label.setText("  |  ".join(parts))

    def _stop_timers(self):
        for timer in (self._eeg_timer, self._game_timer, self._dot_timer):
            if timer.isActive():
                timer.stop()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
