import json
import random
import time
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from controller import EEGController
from game.flappy import Bird

# ── constants ─────────────────────────────────────────────────────────────────

ORANGE = "#F2A007"
PIPE_WIDTH = 60
_BG_CACHE: QtGui.QPixmap | None = None

_SCORE_FILE = Path(__file__).resolve().parents[1] / "high_score.json"

_GAME_W, _GAME_H = 360, 640   # canonical game canvas dimensions


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

    def paintEvent(self, _event):
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
    """Fills itself with flappybirdbg.png, cropping to preserve aspect ratio."""

    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        bg = _bg_pixmap()
        w, h = self.width(), self.height()
        scaled = bg.scaled(w, h, QtCore.Qt.KeepAspectRatioByExpanding,
                           QtCore.Qt.SmoothTransformation)
        x_off = (scaled.width() - w) // 2
        y_off = (scaled.height() - h) // 2
        p.drawPixmap(-x_off, -y_off, scaled)
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

class ModeSelectScreen(QtWidgets.QWidget):
    """Splash screen shown before the game. Emits mode_selected(int) on choice."""

    mode_selected = QtCore.Signal(int)

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
                w, h, QtCore.Qt.KeepAspectRatioByExpanding,
                QtCore.Qt.SmoothTransformation
            ).width()
        super().resizeEvent(event)

    def _tick_scroll(self):
        if self._cached_sw > 1:
            self._scroll_x = (self._scroll_x + 0.5) % self._cached_sw
            self.update()

    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        bg = _bg_pixmap()
        w, h = self.width(), self.height()
        if w == 0 or h == 0:
            return
        scaled = bg.scaled(w, h, QtCore.Qt.KeepAspectRatioByExpanding,
                           QtCore.Qt.SmoothTransformation)
        sw = scaled.width()
        offset = int(self._scroll_x)
        p.drawPixmap(-offset, 0, scaled)
        p.drawPixmap(sw - offset, 0, scaled)
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
    def __init__(self, controller: EEGController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._total = controller.calibration_duration
        self._build_ui()
        try:
            self._ctrl.calibration_progress.connect(self._on_progress)
        except AttributeError:
            pass

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

    def set_remaining(self, remaining: int) -> None:
        self._countdown.setText(f"{remaining}s remaining")

    def _on_progress(self, detected: int, elapsed: float, total: float):
        self.set_remaining(max(0, int(total - elapsed) + 1))


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
        # EEGController doesn't emit these signals yet — deferred
        try:
            self._ctrl.detection_event.connect(self._on_detection_event)
            self._ctrl.bird_updated.connect(self._on_bird_updated)
            self._ctrl.pipes_updated.connect(self._on_pipes_updated)
        except AttributeError:
            pass

    def update_bird(self, y: float, vel: float) -> None:
        """Called by MainWindow each game tick until signal-based updates land."""
        angle = max(-30, min(60, vel * 3))
        self._bird_item.setRotation(angle)
        self._bird_item.setPos(self._bird_x, y)

    def update_pipes(self, pipes: list) -> None:
        """Called by MainWindow each game tick with current pipe positions."""
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
        self._eeg_timer = QtCore.QTimer()
        self._game_timer = QtCore.QTimer()
        self._show_home()

    # ── screen transitions ────────────────────────────────────────────────────

    def _show_home(self):
        self._stop_timers()
        self.resize(_GAME_W, _GAME_H + 40)
        screen = ModeSelectScreen()
        screen.mode_selected.connect(self._on_mode_selected)
        self.setCentralWidget(screen)

    def _on_mode_selected(self, mode: int):
        self._mode = mode
        self._show_calibration()

    def _show_calibration(self):
        self._stop_timers()
        self.controller = EEGController(self._mode)
        self.setWindowTitle(self.controller.window_title)
        self.resize(_GAME_W * 2 + 20, _GAME_H + 40)

        self._calib_screen = CalibrationScreen(self.controller)
        self._calib_screen.setFixedSize(_GAME_W, _GAME_H)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self._eeg_plot = pg.PlotWidget(title="EEG (F4)")
        self._eeg_plot.setYRange(-150, 150)
        self._eeg_plot.showGrid(x=True, y=True, alpha=0.3)
        self._eeg_curve = self._eeg_plot.plot(pen="y")
        layout.addWidget(self._eeg_plot)
        layout.addWidget(self._calib_screen)

        self.setCentralWidget(container)

        self._calib_start = time.monotonic()
        self._eeg_timer = QtCore.QTimer()
        self._eeg_timer.timeout.connect(self._calibration_tick)
        self._eeg_timer.start(20)

    def _show_play(self):
        self._stop_timers()
        self.resize(_GAME_W * 2 + 20, _GAME_H + 40)
        self._bird = Bird(y=_GAME_H // 2)
        self._score = 0

        self._play_screen = PlayScreen(self.controller)
        self._pipes: list[dict] = []
        self._pipe_timer = 0
        self._pipe_spawn_interval = 90  # frames (~1.5 s at 60 fps)
        self._pipe_speed = 3.0
        self._pipe_gap = 200

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        top_row = QtWidgets.QHBoxLayout()

        self._eeg_plot = pg.PlotWidget(title="EEG (F4)")
        self._eeg_plot.setYRange(-150, 150)
        self._eeg_plot.showGrid(x=True, y=True, alpha=0.3)
        self._eeg_curve = self._eeg_plot.plot(pen="y")
        top_row.addWidget(self._eeg_plot)
        top_row.addWidget(self._play_screen)
        layout.addLayout(top_row)

        self._status_label = QtWidgets.QLabel(self.controller.status_text)
        self._status_label.setStyleSheet("font-size: 14px; padding: 4px;")
        layout.addWidget(self._status_label)

        self.setCentralWidget(container)

        self._eeg_timer = QtCore.QTimer()
        self._eeg_timer.timeout.connect(self._eeg_tick)
        self._eeg_timer.start(20)

        self._game_timer = QtCore.QTimer()
        self._game_timer.timeout.connect(self._game_tick)
        self._game_timer.start(16)

    def _show_game_over(self, score: int):
        self._stop_timers()
        self.resize(_GAME_W, _GAME_H + 40)
        if score > self._high_score:
            self._high_score = score
            _save_high_score(score)
        screen = GameOverScreen(score, self._high_score)
        screen.restart.connect(self._show_calibration)
        screen.go_home.connect(self._show_home)
        self.setCentralWidget(screen)

    # ── timer callbacks ───────────────────────────────────────────────────────

    def _calibration_tick(self):
        elapsed = time.monotonic() - self._calib_start
        remaining = max(0, int(self.controller.calibration_duration - elapsed))
        self._calib_screen.set_remaining(remaining)

        update = self.controller.process_eeg()
        if update is not None:
            self._eeg_curve.setData(update.rel_times, update.signal)
        if not self.controller.calibrating:
            self._show_play()

    def _eeg_tick(self):
        update = self.controller.process_eeg()
        if update is None:
            return
        self._eeg_curve.setData(update.rel_times, update.signal)
        if update.status_text:
            self._status_label.setText(update.status_text)
        if update.jump_now:
            self._bird.jump()

    def _game_tick(self):
        if self.controller.consume_held_jump():
            self._bird.jump()

        self._bird.update()
        self._bird.y = max(0, min(_GAME_H, self._bird.y))
        self._play_screen.update_bird(self._bird.y, self._bird.vel)

        # ── spawn pipes ───────────────────────────────────────────────────────
        self._pipe_timer += 1
        if self._pipe_timer >= self._pipe_spawn_interval:
            self._pipe_timer = 0
            gap_top = random.randint(80, _GAME_H - 80 - self._pipe_gap)
            self._pipes.append({
                'x': float(_GAME_W),
                'top_h': gap_top,
                'bot_y': gap_top + self._pipe_gap,
                'scored': False,
            })

        # ── move pipes ────────────────────────────────────────────────────────
        for pipe in self._pipes:
            pipe['x'] -= self._pipe_speed

        # ── score: bird passed a pipe ─────────────────────────────────────────
        for pipe in self._pipes:
            if not pipe['scored'] and pipe['x'] + PIPE_WIDTH < 80:
                pipe['scored'] = True
                self._score += 1
                self._status_label.setText(f"Score: {self._score}")

        # ── collision detection ───────────────────────────────────────────────
        bx, by = 80, self._bird.y
        for pipe in self._pipes:
            if bx + 17 > pipe['x'] and bx - 17 < pipe['x'] + PIPE_WIDTH:
                if by - 12 < pipe['top_h'] or by + 12 > pipe['bot_y']:
                    self._show_game_over(self._score)
                    return

        # ── remove off-screen pipes ───────────────────────────────────────────
        self._pipes = [p for p in self._pipes if p['x'] > -PIPE_WIDTH]

        # ── push positions to PlayScreen ──────────────────────────────────────
        pipe_data = [(p['x'], p['top_h'], p['bot_y'], _GAME_H - p['bot_y'])
                     for p in self._pipes]
        self._play_screen.update_pipes(pipe_data)

        if self._bird.y >= _GAME_H:
            self._show_game_over(self._score)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _stop_timers(self):
        for timer in (self._eeg_timer, self._game_timer):
            if timer.isActive():
                timer.stop()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
