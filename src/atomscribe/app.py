"""Application configuration and setup"""

import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from loguru import logger

from .styles.theme import ThemeManager


class AtomScribeApp(QApplication):
    """
    Main application class with custom configuration.
    """

    def __init__(self, argv: list[str] | None = None):
        if argv is None:
            argv = sys.argv
        super().__init__(argv)

        self._setup_application()
        self._setup_logging()
        self._apply_theme()

    def _setup_application(self) -> None:
        """Configure application metadata and settings"""
        self.setApplicationName("AI Lab Scribe")
        self.setApplicationVersion("0.1.0")
        self.setOrganizationName("AtomSTEM")
        self.setOrganizationDomain("atomstem.com")

        # Enable high DPI scaling
        self.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    def _setup_logging(self) -> None:
        """Configure loguru logging"""
        logger.remove()  # Remove default handler
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            level="DEBUG",
        )
        logger.info("AI Lab Scribe starting...")

    def _apply_theme(self) -> None:
        """Apply the UI theme"""
        ThemeManager.apply_light_theme(self)
