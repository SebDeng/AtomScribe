"""Generation mode selection dialog - choose document generation mode after recording."""

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QCheckBox,
    QFrame,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from ...core.doc_generator import GenerationMode
from ...core.config import get_config_manager


class GenerationModeDialog(QDialog):
    """Dialog for selecting document generation mode."""

    # Signal emitted when user confirms generation
    mode_selected = Signal(GenerationMode, bool)  # (mode, remember_choice)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config = get_config_manager()

        self.setWindowTitle("Generate Document")
        self.setFixedWidth(400)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Title
        title = QLabel("Generate Document")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Choose the document type to generate:")
        subtitle.setStyleSheet("color: #787774;")
        layout.addWidget(subtitle)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #E0E0E0;")
        layout.addWidget(line)

        # Mode selection
        self._mode_group = QButtonGroup(self)

        # Training mode option
        training_container = QFrame()
        training_layout = QVBoxLayout(training_container)
        training_layout.setContentsMargins(12, 12, 12, 12)
        training_layout.setSpacing(4)

        self._training_radio = QRadioButton("Training Tutorial")
        training_font = QFont()
        training_font.setBold(True)
        self._training_radio.setFont(training_font)
        training_layout.addWidget(self._training_radio)

        training_desc = QLabel("Step-by-step instructions with screenshots and before/after comparisons")
        training_desc.setStyleSheet("color: #787774; margin-left: 20px;")
        training_desc.setWordWrap(True)
        training_layout.addWidget(training_desc)

        self._mode_group.addButton(self._training_radio, 0)
        layout.addWidget(training_container)

        # Experiment log option
        experiment_container = QFrame()
        experiment_layout = QVBoxLayout(experiment_container)
        experiment_layout.setContentsMargins(12, 12, 12, 12)
        experiment_layout.setSpacing(4)

        self._experiment_radio = QRadioButton("Experiment Log")
        experiment_font = QFont()
        experiment_font.setBold(True)
        self._experiment_radio.setFont(experiment_font)
        experiment_layout.addWidget(self._experiment_radio)

        experiment_desc = QLabel("Focus on key findings and discoveries, organized chronologically")
        experiment_desc.setStyleSheet("color: #787774; margin-left: 20px;")
        experiment_desc.setWordWrap(True)
        experiment_layout.addWidget(experiment_desc)

        self._mode_group.addButton(self._experiment_radio, 1)
        layout.addWidget(experiment_container)

        # Set default based on config
        default_mode = self._config.config.doc_generation_default_mode
        if default_mode == "experiment_log":
            self._experiment_radio.setChecked(True)
        else:
            self._training_radio.setChecked(True)

        # Remember choice checkbox
        layout.addSpacing(8)
        self._remember_checkbox = QCheckBox("Remember my choice and don't ask again")
        self._remember_checkbox.setStyleSheet("color: #787774;")
        layout.addWidget(self._remember_checkbox)

        # Buttons
        layout.addSpacing(16)
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        self._skip_btn = QPushButton("Skip")
        self._skip_btn.setFixedWidth(100)
        self._skip_btn.clicked.connect(self.reject)
        button_layout.addWidget(self._skip_btn)

        button_layout.addStretch()

        self._generate_btn = QPushButton("Generate")
        self._generate_btn.setFixedWidth(120)
        self._generate_btn.setDefault(True)
        self._generate_btn.clicked.connect(self._on_generate)
        button_layout.addWidget(self._generate_btn)

        layout.addLayout(button_layout)

    def _apply_styles(self):
        """Apply styles to the dialog."""
        self.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF;
            }
            QRadioButton {
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
            QFrame {
                background-color: #F7F7F5;
                border-radius: 8px;
            }
            QFrame:hover {
                background-color: #EEEEEC;
            }
            QPushButton {
                background-color: #F7F7F5;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 8px 16px;
                color: #37352F;
            }
            QPushButton:hover {
                background-color: #EEEEEC;
            }
            QPushButton:default {
                background-color: #2383E2;
                border: none;
                color: white;
            }
            QPushButton:default:hover {
                background-color: #1A73D8;
            }
        """)

    def _on_generate(self):
        """Handle generate button click."""
        # Determine selected mode
        if self._experiment_radio.isChecked():
            mode = GenerationMode.EXPERIMENT_LOG
        else:
            mode = GenerationMode.TRAINING

        # Save preference if remember is checked
        remember = self._remember_checkbox.isChecked()
        if remember:
            self._config.config.doc_generation_default_mode = mode.value
            self._config.config.doc_generation_show_dialog = False
            self._config.save()

        self.mode_selected.emit(mode, remember)
        self.accept()

    def get_selected_mode(self) -> GenerationMode:
        """Get the selected generation mode."""
        if self._experiment_radio.isChecked():
            return GenerationMode.EXPERIMENT_LOG
        return GenerationMode.TRAINING
