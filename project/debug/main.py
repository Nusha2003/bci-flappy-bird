"""
Debug entry point — runs the full game UI without EEG hardware.

Usage:
    cd project
    python debug/main.py

Controls:
    Space   — inject a synthetic spike (triggers blink or jaw detection)
    Q / Esc — quit
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyqtgraph.Qt import QtCore, QtWidgets

from ui.test_window import MainWindow
from debug.mock_stream import MockEEGStream


class DebugMainWindow(MainWindow):
    """MainWindow that injects MockEEGStream and adds keyboard shortcuts."""

    def _make_stream(self):
        self._mock_stream = MockEEGStream()
        return self._mock_stream

    def keyPressEvent(self, event: QtCore.QEvent):
        key = event.key()

        if key == QtCore.Qt.Key_Space:
            stream = getattr(self, "_mock_stream", None)
            if stream is not None:
                stream.inject_spike()
            return

        if key in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            self.close()
            return

        super().keyPressEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = DebugMainWindow()
    window.setWindowTitle("BCI Flappy Bird [DEBUG — no hardware]")
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
