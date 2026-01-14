"""Generation progress dialog - shows document generation progress."""

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QFrame,
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont

from ...signals import get_app_signals


class GenerationProgressDialog(QDialog):
    """Dialog showing document generation progress."""

    # Signal emitted when user cancels generation
    cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._signals = get_app_signals()
        self._completed = False
        self._result_path: str = ""

        self.setWindowTitle("Generating Document")
        self.setFixedWidth(450)
        self.setWindowFlags(
            self.windowFlags() &
            ~Qt.WindowContextHelpButtonHint &
            ~Qt.WindowCloseButtonHint
        )

        self._setup_ui()
        self._apply_styles()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Title (changes based on state)
        self._title = QLabel("Generating Document...")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        self._title.setFont(title_font)
        layout.addWidget(self._title)

        # Progress bar container
        progress_container = QFrame()
        progress_layout = QVBoxLayout(progress_container)
        progress_layout.setContentsMargins(16, 16, 16, 16)
        progress_layout.setSpacing(12)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%p%")
        progress_layout.addWidget(self._progress_bar)

        # Status label
        self._status_label = QLabel("Initializing...")
        self._status_label.setStyleSheet("color: #787774;")
        self._status_label.setWordWrap(True)
        progress_layout.addWidget(self._status_label)

        layout.addWidget(progress_container)

        # Result info (hidden initially)
        self._result_container = QFrame()
        self._result_container.setVisible(False)
        result_layout = QVBoxLayout(self._result_container)
        result_layout.setContentsMargins(16, 16, 16, 16)
        result_layout.setSpacing(8)

        result_icon = QLabel("\u2713")  # Checkmark
        result_icon.setStyleSheet("color: #22C55E; font-size: 24px;")
        result_icon.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(result_icon)

        self._result_path_label = QLabel()
        self._result_path_label.setStyleSheet("color: #787774;")
        self._result_path_label.setWordWrap(True)
        self._result_path_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self._result_path_label)

        layout.addWidget(self._result_container)

        # Buttons
        layout.addSpacing(8)
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        button_layout.addStretch()

        # Cancel button (visible during generation)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedWidth(100)
        self._cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(self._cancel_btn)

        # Open button (visible after completion)
        self._open_btn = QPushButton("Open Document")
        self._open_btn.setFixedWidth(130)
        self._open_btn.setVisible(False)
        self._open_btn.clicked.connect(self._on_open)
        button_layout.addWidget(self._open_btn)

        # Close button (visible after completion)
        self._close_btn = QPushButton("Close")
        self._close_btn.setFixedWidth(100)
        self._close_btn.setVisible(False)
        self._close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self._close_btn)

        layout.addLayout(button_layout)

    def _apply_styles(self):
        """Apply styles to the dialog."""
        self.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF;
            }
            QFrame {
                background-color: #F7F7F5;
                border-radius: 8px;
            }
            QProgressBar {
                border: none;
                border-radius: 4px;
                background-color: #E0E0E0;
                height: 8px;
                text-align: center;
            }
            QProgressBar::chunk {
                border-radius: 4px;
                background-color: #2383E2;
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
        """)

    def _connect_signals(self):
        """Connect to app signals."""
        self._signals.doc_generation_progress.connect(self._on_progress)
        self._signals.doc_generation_completed.connect(self._on_completed)
        self._signals.doc_generation_cancelled.connect(self._on_cancelled)
        self._signals.doc_generation_error.connect(self._on_error)

    @Slot(int, int, str)
    def _on_progress(self, current: int, total: int, description: str):
        """Handle progress update."""
        if total > 0:
            percent = int((current / total) * 100)
            self._progress_bar.setValue(percent)
        self._status_label.setText(description)

    @Slot(str)
    def _on_completed(self, path: str):
        """Handle generation completed."""
        self._completed = True
        self._result_path = path

        # Update UI
        self._title.setText("Document Generated!")
        self._progress_bar.setValue(100)

        # Show result info
        self._result_container.setVisible(True)
        try:
            rel_path = Path(path).name
            self._result_path_label.setText(f"Saved to: {rel_path}")
        except Exception:
            self._result_path_label.setText(f"Saved to: {path}")

        # Update buttons
        self._cancel_btn.setVisible(False)
        self._open_btn.setVisible(True)
        self._close_btn.setVisible(True)

        # Allow closing
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowCloseButtonHint
        )
        self.show()

    @Slot()
    def _on_cancelled(self):
        """Handle generation cancelled."""
        self._title.setText("Generation Cancelled")
        self._status_label.setText("Document generation was cancelled.")
        self._cancel_btn.setVisible(False)
        self._close_btn.setVisible(True)

        # Allow closing
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowCloseButtonHint
        )
        self.show()

    @Slot(str)
    def _on_error(self, error: str):
        """Handle generation error."""
        self._title.setText("Generation Failed")
        self._status_label.setText(f"Error: {error}")
        self._status_label.setStyleSheet("color: #DC2626;")
        self._cancel_btn.setVisible(False)
        self._close_btn.setVisible(True)

        # Allow closing
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowCloseButtonHint
        )
        self.show()

    def _on_cancel(self):
        """Handle cancel button click."""
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setText("Cancelling...")
        self.cancel_requested.emit()

    def _on_open(self):
        """Open the generated document."""
        if not self._result_path:
            return

        try:
            path = Path(self._result_path)
            if not path.exists():
                return

            # Open with default application
            if sys.platform == "win32":
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=False)
            else:
                subprocess.run(["xdg-open", str(path)], check=False)

        except Exception as e:
            self._status_label.setText(f"Failed to open: {e}")

    def closeEvent(self, event):
        """Handle close event."""
        if not self._completed and self._cancel_btn.isVisible():
            # Generation in progress, ignore close
            event.ignore()
        else:
            # Disconnect signals before closing
            try:
                self._signals.doc_generation_progress.disconnect(self._on_progress)
                self._signals.doc_generation_completed.disconnect(self._on_completed)
                self._signals.doc_generation_cancelled.disconnect(self._on_cancelled)
                self._signals.doc_generation_error.disconnect(self._on_error)
            except Exception:
                pass
            event.accept()
