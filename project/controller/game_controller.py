import time
import numpy as np
from collections import deque

from pyqtgraph.Qt import QtCore

from eeg.stream import EEGStream
from eeg.preprocess import preprocess
from eeg.blink_detector import BlinkDetector
from eeg.jaw_clench_detector import JawClenchDetector
from game.flappy import Game


class GameController(QtCore.QObject):
    """
    Bridges EEG detection → Game logic → Qt signals.

    The Game object owns all pure game state (bird, pipes, score).
    This class owns the EEG pipeline and fires Qt signals for the UI.
    """

    eeg_updated        = QtCore.Signal(object, object)  # (times, signal) arrays
    calibration_progress = QtCore.Signal(int, float, float)  # (detected, elapsed, total)
    calibration_done   = QtCore.Signal()
    jump_triggered     = QtCore.Signal()
    detection_event    = QtCore.Signal(int)   # raw event count per EEG chunk
    bird_updated       = QtCore.Signal(float, float)   # (y, vel)
    pipes_updated      = QtCore.Signal(object)          # list of pipe tuples
    game_over          = QtCore.Signal(int)             # final score

    CALIBRATION_DURATION = 5.0
    BLINK_THRESH_MIN     = 70.0
    JAW_THRESH_MIN       = 0.2
    JUMP_INTERVAL        = 0.18   # s between repeated jumps during jaw hold
    HOLD_SECONDS         = 0.35   # how long a jaw clench keeps the jump window open
    JAW_BLOCK_SECONDS    = 0.25   # blink-veto window in jaw mode

    def __init__(self, mode: int, parent=None, stream=None):
        super().__init__(parent)
        self.mode   = mode
        self.stream = stream if stream is not None else EEGStream()
        self.game   = Game()

        self._fp1_index    = 1
        self._plot_duration = 2
        self._buffer      = deque(maxlen=int(self.stream.fs * 5))
        self._time_buffer = deque(maxlen=int(self.stream.fs * 5))

        # Calibration
        self._calibrating             = True
        self._calibration_data: list  = []
        self._calibration_start_time: float | None = None
        self._calib_detected          = 0

        # Jaw / fist timing
        self._clench_hold_until = 0.0
        self._last_jump_time    = 0.0
        self._jaw_block_until   = 0.0

        if mode == 1:
            self.detector = BlinkDetector(self.stream.fs)
            self._blink_veto: BlinkDetector | None = None
        else:
            # mode 2 = jaw clench; mode 3 = fist clench (placeholder)
            self.detector    = JawClenchDetector(self.stream.fs)
            self._blink_veto = BlinkDetector(self.stream.fs)

        self._eeg_timer  = QtCore.QTimer(self)
        self._eeg_timer.timeout.connect(self._eeg_tick)
        self._game_timer = QtCore.QTimer(self)
        self._game_timer.timeout.connect(self._game_tick)

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        self._eeg_timer.start(20)
        self._game_timer.start(16)

    def stop(self):
        self._eeg_timer.stop()
        self._game_timer.stop()

    @property
    def is_calibrating(self) -> bool:
        return self._calibrating

    # ── EEG tick ──────────────────────────────────────────────────────────────

    def _eeg_tick(self):
        chunk, ts = self.stream.pull()
        if len(ts) == 0:
            return

        fp1 = chunk[:, self._fp1_index]
        for v, t in zip(fp1, ts):
            self._buffer.append(v)
            self._time_buffer.append(t)

        if len(self._buffer) < int(self.stream.fs):
            return

        window = int(self.stream.fs * self._plot_duration)
        data   = np.array(self._buffer)[-window:]
        times  = np.array(self._time_buffer)[-window:]

        signal = preprocess(data, self.stream.fs)
        self.eeg_updated.emit(times - times[-1], signal)

        blink_events = []
        if self.mode == 2 and self._blink_veto is not None:
            _, blink_events = self._blink_veto.detect(signal, times, self.BLINK_THRESH_MIN)

        if self._calibrating:
            self._calibration_data.extend(fp1.tolist())
            self._run_calibration(signal, times, blink_events)
            return

        if self.mode == 1:
            self._handle_blink(signal, times)
        else:
            self._handle_jaw(signal, times, blink_events)

    # ── calibration ───────────────────────────────────────────────────────────

    def _run_calibration(self, signal, times, blink_events):
        if self._calibration_start_time is None:
            self._calibration_start_time = times[-1]

        if self.mode == 1:
            _, events = self.detector.detect(signal, times, self.BLINK_THRESH_MIN)
        else:
            events = [] if blink_events else self.detector.detect(signal, times, self.JAW_THRESH_MIN)[1]

        if events:
            self._calib_detected += len(events)

        elapsed = times[-1] - self._calibration_start_time
        self.calibration_progress.emit(self._calib_detected, elapsed, self.CALIBRATION_DURATION)

        if elapsed < self.CALIBRATION_DURATION:
            return

        calib_signal = preprocess(np.array(self._calibration_data), self.stream.fs)
        if hasattr(self.detector, "calibrate"):
            try:
                self.detector.calibrate(calib_signal)
            except Exception:
                pass

        self._calibrating = False
        self.calibration_done.emit()

    # ── detection handlers ────────────────────────────────────────────────────

    def _handle_blink(self, signal, times):
        _, blinks = self.detector.detect(signal, times, self.BLINK_THRESH_MIN)
        if blinks:
            self.detection_event.emit(len(blinks))
            self.game.jump()
            self.jump_triggered.emit()

    def _handle_jaw(self, signal, times, blink_events):
        now = time.monotonic()

        if blink_events:
            self._jaw_block_until = now + self.JAW_BLOCK_SECONDS
            return
        if now < self._jaw_block_until:
            return

        _, clenches, _ = self.detector.detect(signal, times, self.JAW_THRESH_MIN)
        if clenches:
            self.detection_event.emit(len(clenches))
            self._clench_hold_until = now + self.HOLD_SECONDS

    # ── game tick ─────────────────────────────────────────────────────────────

    def _game_tick(self):
        if self._calibrating:
            return   # game is paused while calibrating

        now = time.monotonic()

        # Jaw / fist hold-jump
        if self.mode != 1:
            if now < self._clench_hold_until and (now - self._last_jump_time) >= self.JUMP_INTERVAL:
                self.game.jump()
                self._last_jump_time = now
                self.jump_triggered.emit()

        alive = self.game.update()

        self.pipes_updated.emit(self.game.pipe_data())
        self.bird_updated.emit(self.game.bird.y, self.game.bird.vel)

        if not alive:
            self._eeg_timer.stop()
            self._game_timer.stop()
            self.pipes_updated.emit([])
            self.game_over.emit(self.game.score)
