import os
import sys
from pathlib import Path


def _configure_qt_plugin_paths() -> None:
    """Point Qt at the conda plugin directory before importing Qt."""
    prefix = Path(sys.prefix)
    plugins_dir = prefix / "plugins"
    platforms_dir = plugins_dir / "platforms"

    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

    if platforms_dir.exists():
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(platforms_dir))
    if plugins_dir.exists():
        os.environ.setdefault("QT_PLUGIN_PATH", str(plugins_dir))


_configure_qt_plugin_paths()

from pyqtgraph.Qt import QtWidgets

#from ui.main_window import MainWindow
from ui.test_window import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
