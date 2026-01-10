"""Theme manager for loading and applying UI themes"""

from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontDatabase
from loguru import logger


class ThemeManager:
    """Manages loading and applying UI themes"""

    STYLES_DIR = Path(__file__).parent

    @classmethod
    def apply_light_theme(cls, app: QApplication) -> None:
        """Apply the Notion-style light theme to the application"""
        # Load custom fonts if available
        cls._load_fonts()

        # Set default font
        font = QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        font.setStyleStrategy(QFont.PreferAntialias)
        app.setFont(font)

        # Load and apply QSS stylesheet
        qss_path = cls.STYLES_DIR / "notion_light.qss"
        if qss_path.exists():
            try:
                with open(qss_path, "r", encoding="utf-8") as f:
                    stylesheet = f.read()
                app.setStyleSheet(stylesheet)
                logger.info(f"Loaded theme from {qss_path}")
            except Exception as e:
                logger.error(f"Failed to load theme: {e}")
        else:
            logger.warning(f"Theme file not found: {qss_path}")

    @classmethod
    def _load_fonts(cls) -> None:
        """Load custom fonts from resources"""
        fonts_dir = cls.STYLES_DIR.parent / "ui" / "resources" / "fonts"
        if fonts_dir.exists():
            for font_file in fonts_dir.glob("*.ttf"):
                font_id = QFontDatabase.addApplicationFont(str(font_file))
                if font_id != -1:
                    logger.debug(f"Loaded font: {font_file.name}")
