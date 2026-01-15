"""Settings dialog for API configuration"""

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QWidget,
    QFormLayout,
    QMessageBox,
    QCheckBox,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from ...core.config import get_config_manager


class SettingsDialog(QDialog):
    """Settings dialog for configuring API keys and other settings"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(550)
        self.setMinimumHeight(350)
        self.setModal(True)

        self._config = get_config_manager()
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Title
        title = QLabel("Settings")
        title.setFont(QFont("Segoe UI", 18, QFont.DemiBold))
        title.setStyleSheet("color: #37352F;")
        layout.addWidget(title)

        layout.addSpacing(8)

        # Tab widget for different settings sections
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                background-color: #FFFFFF;
                padding: 16px;
            }
            QTabBar::tab {
                background-color: #F7F7F5;
                border: 1px solid #E0E0E0;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 8px 16px;
                margin-right: 4px;
                color: #787774;
                font-size: 12px;
            }
            QTabBar::tab:selected {
                background-color: #FFFFFF;
                color: #37352F;
                font-weight: 600;
            }
            QTabBar::tab:hover:!selected {
                background-color: #EEEEEE;
            }
        """)

        # API Settings tab
        api_tab = self._create_api_tab()
        self.tab_widget.addTab(api_tab, "API Keys")

        layout.addWidget(self.tab_widget)

        layout.addSpacing(16)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumHeight(40)
        cancel_btn.setMinimumWidth(100)
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet("""
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
        btn_layout.addWidget(cancel_btn)

        btn_layout.addSpacing(8)

        save_btn = QPushButton("Save")
        save_btn.setMinimumHeight(40)
        save_btn.setMinimumWidth(100)
        save_btn.clicked.connect(self._on_save)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #2383E2;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: 600;
                color: white;
            }
            QPushButton:hover {
                background-color: #1A73D1;
            }
        """)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

        # Dialog style
        self.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF;
            }
        """)

    def _create_api_tab(self) -> QWidget:
        """Create the API settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)
        layout.setContentsMargins(8, 8, 8, 8)

        # Claude API section
        section_label = QLabel("Claude API (Anthropic)")
        section_label.setFont(QFont("Segoe UI", 13, QFont.DemiBold))
        section_label.setStyleSheet("color: #37352F;")
        layout.addWidget(section_label)

        desc = QLabel(
            "Enter your Anthropic API key to enable AI-powered document generation.\n"
            "Get your API key from: console.anthropic.com"
        )
        desc.setFont(QFont("Segoe UI", 11))
        desc.setStyleSheet("color: #787774;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addSpacing(8)

        # API key input
        key_layout = QHBoxLayout()
        key_layout.setSpacing(8)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("sk-ant-api03-...")
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setMinimumHeight(40)
        self.api_key_edit.setStyleSheet("""
            QLineEdit {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 13px;
                font-family: Consolas, monospace;
                color: #37352F;
            }
            QLineEdit:focus {
                border-color: #2383E2;
            }
        """)
        key_layout.addWidget(self.api_key_edit, stretch=1)

        # Show/hide toggle button
        self.show_key_btn = QPushButton("Show")
        self.show_key_btn.setMinimumHeight(40)
        self.show_key_btn.setMinimumWidth(70)
        self.show_key_btn.setCheckable(True)
        self.show_key_btn.clicked.connect(self._toggle_key_visibility)
        self.show_key_btn.setStyleSheet("""
            QPushButton {
                background-color: #F7F7F5;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 12px;
                color: #37352F;
            }
            QPushButton:hover {
                background-color: #EEEEEE;
            }
            QPushButton:checked {
                background-color: #E0E0E0;
            }
        """)
        key_layout.addWidget(self.show_key_btn)

        layout.addLayout(key_layout)

        # Status indicator
        self.status_label = QLabel("")
        self.status_label.setFont(QFont("Segoe UI", 10))
        self.status_label.setStyleSheet("color: #787774;")
        layout.addWidget(self.status_label)

        # Test button
        test_btn = QPushButton("Test Connection")
        test_btn.setMinimumHeight(36)
        test_btn.setMaximumWidth(150)
        test_btn.clicked.connect(self._test_api_key)
        test_btn.setStyleSheet("""
            QPushButton {
                background-color: #F7F7F5;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 12px;
                color: #37352F;
            }
            QPushButton:hover {
                background-color: #EEEEEE;
                border-color: #BDBDBD;
            }
        """)
        layout.addWidget(test_btn)

        layout.addSpacing(16)

        # Use Claude checkbox
        self.use_claude_checkbox = QCheckBox("Use Claude for document generation")
        self.use_claude_checkbox.setFont(QFont("Segoe UI", 11))
        self.use_claude_checkbox.setStyleSheet("""
            QCheckBox {
                color: #37352F;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                background-color: #FFFFFF;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #2383E2;
                border-radius: 4px;
                background-color: #2383E2;
            }
        """)
        layout.addWidget(self.use_claude_checkbox)

        use_claude_hint = QLabel(
            "When enabled, Claude API will be used for transcript analysis,\n"
            "document generation, and smart screenshot cropping (vision)."
        )
        use_claude_hint.setFont(QFont("Segoe UI", 10))
        use_claude_hint.setStyleSheet("color: #9E9E9E;")
        layout.addWidget(use_claude_hint)

        layout.addSpacing(16)

        # VLM smart crop checkbox (fallback when Claude disabled)
        self.use_vlm_checkbox = QCheckBox("Use local VLM for smart cropping (when Claude disabled)")
        self.use_vlm_checkbox.setFont(QFont("Segoe UI", 11))
        self.use_vlm_checkbox.setStyleSheet("""
            QCheckBox {
                color: #37352F;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                background-color: #FFFFFF;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #2383E2;
                border-radius: 4px;
                background-color: #2383E2;
            }
        """)
        layout.addWidget(self.use_vlm_checkbox)

        use_vlm_hint = QLabel(
            "If Claude is disabled, uses local Qwen3-VL-8B for smart cropping.\n"
            "Disable to skip all smart cropping (faster, uses full frames)."
        )
        use_vlm_hint.setFont(QFont("Segoe UI", 10))
        use_vlm_hint.setStyleSheet("color: #9E9E9E;")
        layout.addWidget(use_vlm_hint)

        layout.addStretch()

        return tab

    def _toggle_key_visibility(self):
        """Toggle API key visibility"""
        if self.show_key_btn.isChecked():
            self.api_key_edit.setEchoMode(QLineEdit.Normal)
            self.show_key_btn.setText("Hide")
        else:
            self.api_key_edit.setEchoMode(QLineEdit.Password)
            self.show_key_btn.setText("Show")

    def _load_settings(self):
        """Load current settings into the UI"""
        api_key = self._config.get_anthropic_api_key()
        if api_key:
            self.api_key_edit.setText(api_key)
            self.status_label.setText("API key configured")
            self.status_label.setStyleSheet("color: #22C55E;")  # Green
        else:
            self.status_label.setText("No API key configured")
            self.status_label.setStyleSheet("color: #787774;")

        # Load use_claude_for_docs preference
        self.use_claude_checkbox.setChecked(self._config.config.use_claude_for_docs)

        # Load use_vlm_for_smart_crop preference
        self.use_vlm_checkbox.setChecked(self._config.config.use_vlm_for_smart_crop)

    def _test_api_key(self):
        """Test the API key by making a simple request"""
        api_key = self.api_key_edit.text().strip()

        if not api_key:
            self.status_label.setText("Please enter an API key first")
            self.status_label.setStyleSheet("color: #EF4444;")  # Red
            return

        self.status_label.setText("Testing connection...")
        self.status_label.setStyleSheet("color: #787774;")

        # Force UI update
        from PySide6.QtCore import QCoreApplication
        QCoreApplication.processEvents()

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)

            # Make a minimal API call to test the key
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )

            self.status_label.setText("Connection successful!")
            self.status_label.setStyleSheet("color: #22C55E;")  # Green

        except anthropic.AuthenticationError:
            self.status_label.setText("Invalid API key")
            self.status_label.setStyleSheet("color: #EF4444;")  # Red
        except anthropic.APIConnectionError:
            self.status_label.setText("Connection failed - check network")
            self.status_label.setStyleSheet("color: #EF4444;")  # Red
        except ImportError:
            self.status_label.setText("anthropic package not installed")
            self.status_label.setStyleSheet("color: #EF4444;")  # Red
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)[:50]}")
            self.status_label.setStyleSheet("color: #EF4444;")  # Red

    def _on_save(self):
        """Save settings and close dialog"""
        api_key = self.api_key_edit.text().strip()

        # Save API key (empty string becomes None)
        self._config.set_anthropic_api_key(api_key if api_key else None)

        # Save preferences
        self._config.config.use_claude_for_docs = self.use_claude_checkbox.isChecked()
        self._config.config.use_vlm_for_smart_crop = self.use_vlm_checkbox.isChecked()
        self._config.save()

        # Reset document generator so new settings take effect
        try:
            from ...core.doc_generator import reset_document_generator
            reset_document_generator()
        except ImportError:
            pass

        self.accept()
