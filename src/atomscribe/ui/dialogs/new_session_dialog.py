"""New session dialog"""

from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QCheckBox,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from ...core.config import get_config_manager


class NewSessionDialog(QDialog):
    """Dialog for creating a new recording session"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Session")
        self.setMinimumWidth(450)
        self.setModal(True)

        self._config = get_config_manager()
        self._custom_path: str = ""

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        # Title
        title = QLabel("Create New Session")
        title.setFont(QFont("Segoe UI", 14, QFont.DemiBold))
        title.setStyleSheet("color: #37352F;")
        layout.addWidget(title)

        layout.addSpacing(8)

        # Session name
        name_label = QLabel("Session Name (optional)")
        name_label.setFont(QFont("Segoe UI", 11))
        name_label.setStyleSheet("color: #787774;")
        layout.addWidget(name_label)

        self.name_edit = QLineEdit()
        default_name = f"Session_{datetime.now().strftime('%Y-%m-%d')}"
        self.name_edit.setPlaceholderText(default_name)
        self.name_edit.setMinimumHeight(36)
        self.name_edit.setStyleSheet("""
            QLineEdit {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 13px;
                color: #37352F;
            }
            QLineEdit:focus {
                border-color: #2383E2;
            }
        """)
        layout.addWidget(self.name_edit)

        layout.addSpacing(12)

        # Save location
        loc_label = QLabel("Save Location")
        loc_label.setFont(QFont("Segoe UI", 11))
        loc_label.setStyleSheet("color: #787774;")
        layout.addWidget(loc_label)

        # Default location info
        default_path = self._config.get_default_save_directory()
        default_text = str(default_path) if default_path else "Not set"

        self.default_radio = QCheckBox(f"Default: {default_text}")
        self.default_radio.setChecked(True)
        self.default_radio.setStyleSheet("""
            QCheckBox {
                font-size: 12px;
                color: #37352F;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)
        self.default_radio.toggled.connect(self._on_default_toggled)
        layout.addWidget(self.default_radio)

        # Custom location
        custom_layout = QHBoxLayout()
        custom_layout.setSpacing(8)

        self.custom_radio = QCheckBox("Custom location:")
        self.custom_radio.setStyleSheet("""
            QCheckBox {
                font-size: 12px;
                color: #37352F;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)
        self.custom_radio.toggled.connect(self._on_custom_toggled)
        custom_layout.addWidget(self.custom_radio)

        self.custom_path_edit = QLineEdit()
        self.custom_path_edit.setEnabled(False)
        self.custom_path_edit.setPlaceholderText("Select folder...")
        self.custom_path_edit.setMinimumHeight(32)
        self.custom_path_edit.setStyleSheet("""
            QLineEdit {
                background-color: #F5F5F5;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 4px 10px;
                font-size: 12px;
                color: #37352F;
            }
            QLineEdit:enabled {
                background-color: #FFFFFF;
            }
            QLineEdit:focus {
                border-color: #2383E2;
            }
        """)
        custom_layout.addWidget(self.custom_path_edit, stretch=1)

        self.browse_btn = QPushButton("...")
        self.browse_btn.setEnabled(False)
        self.browse_btn.setFixedSize(32, 32)
        self.browse_btn.clicked.connect(self._browse_folder)
        self.browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #F7F7F5;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                font-size: 12px;
                color: #37352F;
            }
            QPushButton:hover:enabled {
                background-color: #EEEEEE;
            }
            QPushButton:disabled {
                background-color: #F5F5F5;
                color: #BDBDBD;
            }
        """)
        custom_layout.addWidget(self.browse_btn)

        layout.addLayout(custom_layout)

        layout.addSpacing(20)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumHeight(36)
        cancel_btn.setMinimumWidth(80)
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #F7F7F5;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                color: #37352F;
            }
            QPushButton:hover {
                background-color: #EEEEEE;
            }
        """)
        btn_layout.addWidget(cancel_btn)

        btn_layout.addSpacing(8)

        self.create_btn = QPushButton("Start Recording")
        self.create_btn.setMinimumHeight(36)
        self.create_btn.setMinimumWidth(120)
        self.create_btn.clicked.connect(self.accept)
        self.create_btn.setStyleSheet("""
            QPushButton {
                background-color: #DC2626;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: 600;
                color: white;
            }
            QPushButton:hover {
                background-color: #B91C1C;
            }
        """)
        btn_layout.addWidget(self.create_btn)

        layout.addLayout(btn_layout)

        # Dialog style
        self.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF;
            }
        """)

    def _on_default_toggled(self, checked: bool):
        """Handle default location toggle"""
        if checked:
            self.custom_radio.setChecked(False)

    def _on_custom_toggled(self, checked: bool):
        """Handle custom location toggle"""
        if checked:
            self.default_radio.setChecked(False)
        self.custom_path_edit.setEnabled(checked)
        self.browse_btn.setEnabled(checked)

    def _browse_folder(self):
        """Open folder browser"""
        default_dir = self._config.get_default_save_directory()
        start_dir = str(default_dir) if default_dir else str(Path.home() / "Documents")

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Save Location",
            start_dir,
            QFileDialog.ShowDirsOnly,
        )

        if folder:
            self._custom_path = folder
            self.custom_path_edit.setText(folder)

    def get_session_name(self) -> str:
        """Get the session name (or generate default)"""
        name = self.name_edit.text().strip()
        if not name:
            name = f"Session_{datetime.now().strftime('%Y-%m-%d')}"
        return name

    def get_save_directory(self) -> Path:
        """Get the save directory"""
        if self.custom_radio.isChecked() and self._custom_path:
            return Path(self._custom_path)
        else:
            return self._config.get_default_save_directory()

    def use_default_location(self) -> bool:
        """Check if using default location"""
        return self.default_radio.isChecked()
