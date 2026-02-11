import sys
import subprocess
from PyQt5 import QtWidgets, QtCore

from pygame_wrapper import PygameWrapper

# IMPORTANT: Do NOT import BlinkFlappyWindow at the top
# because importing it will NOT connect—but constructing it will.
# We only construct it after stream is running.

BlinkFlappyWindow = None


class Frontend(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("BCI Flappy Bird Frontend")
        self.resize(1200, 800)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # ------------------------------------------------------------------
        # STREAM BUTTON
        # ------------------------------------------------------------------
        start_btn = QtWidgets.QPushButton("Start FakeEEG Stream")
        start_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        start_btn.clicked.connect(self.start_stream)
        layout.addWidget(start_btn)

        # ------------------------------------------------------------------
        # TABS
        # ------------------------------------------------------------------
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, stretch=1)

        # --- TAB 1 placeholder ---
        self.eeg_placeholder = QtWidgets.QLabel(
            "Start the EEG stream first, then click this tab again."
        )
        self.eeg_placeholder.setAlignment(QtCore.Qt.AlignCenter)

        self.tabs.addTab(self.eeg_placeholder, "EEG Visualizer + Mini Bird")

        # --- TAB 2: Full Game ---
        pygame_tab = PygameWrapper("flappy_bird.py")
        pygame_tab.start_pygame()
        self.tabs.addTab(pygame_tab, "Flappy Bird (Full Game)")

        # Detect when user clicks Tab 0 (EEG)
        self.tabs.currentChanged.connect(self.load_eeg_tab)

        # Track whether EEG tab is loaded
        self.eeg_loaded = False

    # ----------------------------------------------------------------------
    def start_stream(self):
        subprocess.Popen([sys.executable, "stream_simulator.py"])
        QtWidgets.QMessageBox.information(
            self,
            "Stream Started",
            "FakeEEG stream_simulator.py is now streaming EEG!"
        )

    # ----------------------------------------------------------------------
    def load_eeg_tab(self, index):
        """Load EEG tab only after stream exists."""
        global BlinkFlappyWindow

        # Only load if EEG tab is selected (tab index 0)
        if index != 0:
            return

        # Load only once
        if self.eeg_loaded:
            return

        try:
            if BlinkFlappyWindow is None:
                from both import BlinkFlappyWindow

            eeg_widget = BlinkFlappyWindow()
        except RuntimeError:
            # Stream not running yet
            QtWidgets.QMessageBox.warning(
                self,
                "Stream Not Found",
                "No FakeEEG stream detected.\n\n"
                "Please click 'Start FakeEEG Stream' first."
            )
            # Return to the game tab automatically
            self.tabs.setCurrentIndex(1)
            return

        # Replace placeholder with real EEG widget
        self.tabs.removeTab(0)
        self.tabs.insertTab(0, eeg_widget, "EEG Visualizer + Mini Bird")
        self.tabs.setCurrentIndex(0)

        self.eeg_loaded = True


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = Frontend()
    win.show()
    sys.exit(app.exec_())
