import sys
import subprocess
from PyQt5 import QtWidgets


class PygameWrapper(QtWidgets.QWidget):
    """
    A simple wrapper that launches flappy_bird.py in a fully separate process.
    Does NOT block the Qt event loop.
    """

    def __init__(self, script_name):
        super().__init__()
        self.script_name = script_name
        self.process = None
        self.setMinimumSize(360, 640)

    def start_pygame(self):
        """Launch the pygame script asynchronously."""
        if self.process is None:
            self.process = subprocess.Popen(
                [sys.executable, self.script_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

    def closeEvent(self, event):
        """Stop pygame when closing."""
        if self.process:
            self.process.kill()
        event.accept()
