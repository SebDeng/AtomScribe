"""Application entry point"""

import sys
from loguru import logger

from .app import AtomScribeApp
from .ui.main_window import MainWindow


def main() -> int:
    """Main entry point for the application"""
    try:
        # Create application
        app = AtomScribeApp()

        # Create and show main window
        window = MainWindow()
        window.show()

        logger.info("Application window displayed")

        # Run event loop
        return app.exec()

    except Exception as e:
        logger.exception(f"Application error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
