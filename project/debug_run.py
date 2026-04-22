"""
Debug runner — exercises the full UI flow without a live LSL stream.

Run from project/:
    python debug_run.py

What it does:
  - Replaces EEGController with MockEEGController before test_window imports it
  - Mock calibration completes after 3 seconds
  - Mock jumps fire automatically every 2 seconds so you can watch the bird move
  - Game over triggers when the bird hits the floor, then GameOverScreen appears
  - High score file is written to the normal location
"""

import sys
import time
import numpy as np
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ── ControllerUpdate (mirrors the real dataclass in controller.py) ────────────

@dataclass
class ControllerUpdate:
    rel_times: np.ndarray
    signal: np.ndarray
    status_text: str | None = None
    jump_now: bool = False


# ── Mock controller ───────────────────────────────────────────────────────────

class MockEEGController:
    """
    Stands in for EEGController without touching LSL or the headset.
    Generates a synthetic sine-wave EEG signal and fires a jump every 2 s.
    """

    def __init__(self, mode: int):
        self.mode = mode
        self.calibrating = True
        self.calibration_duration = 3          # seconds until calibration finishes
        self.window_title = f"[DEBUG] BCI Flappy Bird  (mode {mode})"
        self.status_text = "Mock calibration starting..."
        self._start = time.monotonic()
        self._signal_t = 0.0                   # running phase for fake EEG
        self._fs = 300
        self._last_jump = time.monotonic()
        self._jump_every = 0.7                # auto-jump period (seconds)
        self._hold_until = 0.0
        self._last_held = 0.0
        print(f"[mock] EEGController created  mode={mode}")

    def process_eeg(self) -> ControllerUpdate:
        elapsed = time.monotonic() - self._start

        # Synthetic EEG: 8 Hz sine + small Gaussian noise
        n = 6
        t = np.linspace(self._signal_t, self._signal_t + n / self._fs, n, endpoint=False)
        self._signal_t += n / self._fs
        signal = 60 * np.sin(2 * np.pi * 8 * t) + np.random.randn(n) * 8
        rel = t - t[-1]

        # --- calibration phase ---
        if self.calibrating:
            remaining = max(0, int(self.calibration_duration - elapsed) + 1)
            self.status_text = f"Mock calibrating...  {remaining}s remaining"
            if elapsed >= self.calibration_duration:
                self.calibrating = False
                self.status_text = "Mock calibration done — auto-jump every 2 s"
                print("[mock] calibration complete")
            return ControllerUpdate(rel_times=rel, signal=signal,
                                    status_text=self.status_text)

        # --- play phase: auto-jump on interval ---
        now = time.monotonic()
        jump_now = False
        if now - self._last_jump >= self._jump_every:
            self._last_jump = now
            jump_now = True
            self._hold_until = now + 0.35
            self.status_text = f"[mock] jump  (mode {self.mode})"
            print(f"[mock] jump fired  t={now:.2f}")

        return ControllerUpdate(rel_times=rel, signal=signal,
                                status_text=self.status_text,
                                jump_now=jump_now)

    def consume_held_jump(self) -> bool:
        """Replays held-jump ticks for jaw-clench mode (mode 2)."""
        if self.mode != 2 or self.calibrating:
            return False
        now = time.monotonic()
        if now >= self._hold_until:
            return False
        if now - self._last_held < 0.18:
            return False
        self._last_held = now
        return True

    def tick_calibration_indicator(self) -> str | None:
        return self.status_text if self.calibrating else None


# ── Patch controller module *before* test_window imports it ──────────────────
# test_window does `from controller import EEGController`, so we must set the
# attribute on the already-imported module object before that import runs.

import controller as _ctrl_mod
_ctrl_mod.EEGController = MockEEGController  # type: ignore[attr-defined]

from pyqtgraph.Qt import QtWidgets
from ui.test_window import MainWindow


def main():
    print("[debug] starting UI")
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()