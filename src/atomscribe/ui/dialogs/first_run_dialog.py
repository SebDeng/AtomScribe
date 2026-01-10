"""First run setup dialog"""

from pathlib import Path
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QSpacerItem,
    QSizePolicy,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from ...core.config import get_config_manager


class FirstRunDialog(QDialog):
    """Dialog shown on first run to set up default save directory"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to AtomScribe")
        self.setMinimumWidth(500)
        self.setModal(True)

        self._selected_path: str = ""
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(32, 32, 32, 32)

        # Welcome title
        title = QLabel("Welcome to AI Lab Scribe")
        title.setFont(QFont("Segoe UI", 18, QFont.DemiBold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #37352F;")
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Let's set up where to save your recordings")
        subtitle.setFont(QFont("Segoe UI", 12))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #787774;")
        layout.addWidget(subtitle)

        layout.addSpacing(24)

        # Description
        desc = QLabel(
            "Choose a folder where AtomScribe will save your recording sessions.\n"
            "Each session will be saved in its own subfolder with audio files,\n"
            "transcripts, and summaries."
        )
        desc.setFont(QFont("Segoe UI", 11))
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("color: #6B6B6B;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addSpacing(16)

        # Path selection
        path_layout = QHBoxLayout()
        path_layout.setSpacing(8)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select a folder...")
        self.path_edit.setReadOnly(True)
        self.path_edit.setMinimumHeight(40)
        self.path_edit.setStyleSheet("""
            QLineEdit {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 13px;
                color: #37352F;
            }
            QLineEdit:focus {
                border-color: #2383E2;
            }
        """)
        path_layout.addWidget(self.path_edit, stretch=1)

        browse_btn = QPushButton("Browse...")
        browse_btn.setMinimumHeight(40)
        browse_btn.setMinimumWidth(100)
        browse_btn.clicked.connect(self._browse_folder)
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #F7F7F5;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 13px;
                color: #37352F;
            }
            QPushButton:hover {
                background-color: #EEEEEE;
                border-color: #BDBDBD;
            }
        """)
        path_layout.addWidget(browse_btn)

        layout.addLayout(path_layout)

        # Suggestion
        suggest_label = QLabel("Tip: You can choose a folder on an encrypted drive for privacy.")
        suggest_label.setFont(QFont("Segoe UI", 10))
        suggest_label.setStyleSheet("color: #9E9E9E;")
        layout.addWidget(suggest_label)

        layout.addSpacing(24)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.continue_btn = QPushButton("Continue")
        self.continue_btn.setEnabled(False)
        self.continue_btn.setMinimumHeight(44)
        self.continue_btn.setMinimumWidth(120)
        self.continue_btn.clicked.connect(self._on_continue)
        self.continue_btn.setStyleSheet("""
            QPushButton {
                background-color: #2383E2;
                border: none;
                border-radius: 8px;
                padding: 10px 24px;
                font-size: 14px;
                font-weight: 600;
                color: white;
            }
            QPushButton:hover {
                background-color: #1A73D1;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        btn_layout.addWidget(self.continue_btn)

        layout.addLayout(btn_layout)

        # Set dialog style
        self.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF;
            }
        """)

    def _browse_folder(self):
        """Open folder browser dialog"""
        # Start from Documents folder
        start_dir = str(Path.home() / "Documents")

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Save Directory",
            start_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )

        if folder:
            self._selected_path = folder
            self.path_edit.setText(folder)
            self.continue_btn.setEnabled(True)

    def _on_continue(self):
        """Handle continue button click"""
        if self._selected_path:
            # Save the configuration
            config = get_config_manager()
            config.set_default_save_directory(self._selected_path)
            self.accept()

    def get_selected_path(self) -> str:
        """Get the selected path"""
        return self._selected_path
