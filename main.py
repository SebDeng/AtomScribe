#!/usr/bin/env python
"""
AI Lab Scribe - Direct launcher
Run with: python main.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Now import and run
from atomscribe.app import AtomScribeApp
from atomscribe.ui.main_window import MainWindow


def main():
    app = AtomScribeApp()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
