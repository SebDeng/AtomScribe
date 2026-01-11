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
    QScrollArea,
    QWidget,
    QFrame,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap

from ...core.config import get_config_manager
from ...core.screen_recorder import get_screen_recorder, ScreenRecorder, MonitorInfo
from ...core import window_manager


class MonitorPreviewWidget(QFrame):
    """Widget showing a monitor thumbnail with selection state"""

    clicked = Signal()

    def __init__(self, monitor: MonitorInfo, parent=None):
        super().__init__(parent)
        self.monitor = monitor
        self._selected = False

        self.setFixedSize(300, 220)
        self.setCursor(Qt.PointingHandCursor)
        self._update_style()

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 6)
        layout.setSpacing(4)

        # Thumbnail container - larger size
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(284, 160)
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border-radius: 6px;
            }
        """)
        layout.addWidget(self.thumbnail_label)

        # Monitor name and resolution
        name_text = self.monitor.name
        if self.monitor.is_primary:
            name_text += " [Primary]"

        self.name_label = QLabel(name_text)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setFixedHeight(18)
        self.name_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #37352F;")
        layout.addWidget(self.name_label)

        res_text = f"{self.monitor.width} x {self.monitor.height}"
        self.res_label = QLabel(res_text)
        self.res_label.setAlignment(Qt.AlignCenter)
        self.res_label.setFixedHeight(16)
        self.res_label.setStyleSheet("font-size: 10px; color: #787774;")
        layout.addWidget(self.res_label)

    def set_thumbnail(self, png_data: bytes):
        """Set the thumbnail image from PNG data"""
        if png_data:
            pixmap = QPixmap()
            pixmap.loadFromData(png_data)
            # Scale to fit
            scaled = pixmap.scaled(
                284, 160,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.thumbnail_label.setPixmap(scaled)
        else:
            self.thumbnail_label.setText("No preview")
            self.thumbnail_label.setStyleSheet("""
                QLabel {
                    background-color: #1a1a1a;
                    border-radius: 6px;
                    color: #666;
                    font-size: 12px;
                }
            """)

    def set_selected(self, selected: bool):
        """Set the selection state"""
        self._selected = selected
        self._update_style()

    def is_selected(self) -> bool:
        """Check if this widget is selected"""
        return self._selected

    def _update_style(self):
        """Update the widget style based on selection state"""
        if self._selected:
            self.setStyleSheet("""
                MonitorPreviewWidget {
                    background-color: #E3F2FD;
                    border: 3px solid #2383E2;
                    border-radius: 12px;
                }
            """)
        else:
            self.setStyleSheet("""
                MonitorPreviewWidget {
                    background-color: #FFFFFF;
                    border: 2px solid #E0E0E0;
                    border-radius: 12px;
                }
                MonitorPreviewWidget:hover {
                    border-color: #BDBDBD;
                    background-color: #FAFAFA;
                }
            """)

    def mousePressEvent(self, event):
        """Handle mouse click"""
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class WindowPreviewWidget(QFrame):
    """Widget showing a window thumbnail with selection state"""

    clicked = Signal()

    def __init__(self, window_info: window_manager.WindowInfo, parent=None):
        super().__init__(parent)
        self.window_info = window_info
        self._selected = False

        self.setFixedSize(300, 220)
        self.setCursor(Qt.PointingHandCursor)
        self._update_style()

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 6)
        layout.setSpacing(4)

        # Thumbnail container
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(284, 160)
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border-radius: 6px;
            }
        """)
        layout.addWidget(self.thumbnail_label)

        # Window title (truncated if too long)
        title = self.window_info.title
        if len(title) > 30:
            title = title[:27] + "..."

        self.name_label = QLabel(title)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setFixedHeight(18)
        self.name_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #37352F;")
        self.name_label.setToolTip(self.window_info.title)
        layout.addWidget(self.name_label)

        # Process name and dimensions
        info_text = f"{self.window_info.process_name} - {self.window_info.width}x{self.window_info.height}"
        self.info_label = QLabel(info_text)
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setFixedHeight(16)
        self.info_label.setStyleSheet("font-size: 10px; color: #787774;")
        layout.addWidget(self.info_label)

    def set_thumbnail(self, png_data: bytes):
        """Set the thumbnail image from PNG data"""
        if png_data:
            pixmap = QPixmap()
            pixmap.loadFromData(png_data)
            scaled = pixmap.scaled(
                284, 160,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.thumbnail_label.setPixmap(scaled)
        else:
            self.thumbnail_label.setText("No preview")
            self.thumbnail_label.setStyleSheet("""
                QLabel {
                    background-color: #1a1a1a;
                    border-radius: 6px;
                    color: #666;
                    font-size: 12px;
                }
            """)

    def set_selected(self, selected: bool):
        """Set the selection state"""
        self._selected = selected
        self._update_style()

    def is_selected(self) -> bool:
        """Check if this widget is selected"""
        return self._selected

    def _update_style(self):
        """Update the widget style based on selection state"""
        if self._selected:
            self.setStyleSheet("""
                WindowPreviewWidget {
                    background-color: #E3F2FD;
                    border: 3px solid #2383E2;
                    border-radius: 12px;
                }
            """)
        else:
            self.setStyleSheet("""
                WindowPreviewWidget {
                    background-color: #FFFFFF;
                    border: 2px solid #E0E0E0;
                    border-radius: 12px;
                }
                WindowPreviewWidget:hover {
                    border-color: #BDBDBD;
                    background-color: #FAFAFA;
                }
            """)

    def mousePressEvent(self, event):
        """Handle mouse click"""
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class NewSessionDialog(QDialog):
    """Dialog for creating a new recording session"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Session")
        self.setMinimumWidth(1000)
        self.setMinimumHeight(750)
        self.resize(1100, 800)
        self.setModal(True)

        self._config = get_config_manager()
        self._custom_path: str = ""

        # Monitor capture state
        self._monitors: list[MonitorInfo] = []
        self._monitor_widgets: list[MonitorPreviewWidget] = []
        self._selected_monitor_index: int = 0

        # Window capture state
        self._windows: list[window_manager.WindowInfo] = []
        self._window_widgets: list[WindowPreviewWidget] = []
        self._selected_window_handle: int = None
        self._capture_mode: str = "monitor"  # "monitor" or "window"

        self._setup_ui()
        self._populate_monitors()
        self._populate_windows()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(28, 28, 28, 28)

        # Title
        title = QLabel("Create New Session")
        title.setFont(QFont("Segoe UI", 16, QFont.DemiBold))
        title.setStyleSheet("color: #37352F;")
        layout.addWidget(title)

        layout.addSpacing(4)

        # Session name
        name_label = QLabel("Session Name (optional)")
        name_label.setFont(QFont("Segoe UI", 11))
        name_label.setStyleSheet("color: #787774;")
        layout.addWidget(name_label)

        self.name_edit = QLineEdit()
        default_name = f"Session_{datetime.now().strftime('%Y-%m-%d')}"
        self.name_edit.setPlaceholderText(default_name)
        self.name_edit.setMinimumHeight(40)
        self.name_edit.setStyleSheet("""
            QLineEdit {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                padding: 8px 14px;
                font-size: 14px;
                color: #37352F;
            }
            QLineEdit:focus {
                border-color: #2383E2;
                border-width: 2px;
            }
        """)
        layout.addWidget(self.name_edit)

        layout.addSpacing(16)

        # ===== Screen Recording section (MAIN VISUAL FOCUS) =====
        screen_header_layout = QHBoxLayout()

        screen_label = QLabel("Screen Recording")
        screen_label.setFont(QFont("Segoe UI", 13, QFont.DemiBold))
        screen_label.setStyleSheet("color: #37352F;")
        screen_header_layout.addWidget(screen_label)

        screen_header_layout.addStretch()

        # Refresh button
        self.refresh_btn = QPushButton("Refresh Previews")
        self.refresh_btn.setFixedHeight(28)
        self.refresh_btn.clicked.connect(self._refresh_thumbnails)
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #F7F7F5;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                font-size: 12px;
                color: #37352F;
                padding: 4px 12px;
            }
            QPushButton:hover {
                background-color: #EEEEEE;
                border-color: #BDBDBD;
            }
        """)
        screen_header_layout.addWidget(self.refresh_btn)

        layout.addLayout(screen_header_layout)

        # Screen recording enable checkbox
        self.screen_recording_check = QCheckBox("Enable screen recording")
        self.screen_recording_check.setChecked(self._config.config.screen_recording_enabled)
        self.screen_recording_check.setStyleSheet("""
            QCheckBox {
                font-size: 13px;
                color: #37352F;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        self.screen_recording_check.toggled.connect(self._on_screen_recording_toggled)
        layout.addWidget(self.screen_recording_check)

        # Tab buttons for Monitors / Windows
        tab_layout = QHBoxLayout()
        tab_layout.setSpacing(0)

        self.monitors_tab_btn = QPushButton("Monitors")
        self.monitors_tab_btn.setCheckable(True)
        self.monitors_tab_btn.setChecked(True)
        self.monitors_tab_btn.setFixedHeight(32)
        self.monitors_tab_btn.clicked.connect(lambda: self._switch_capture_tab("monitor"))
        tab_layout.addWidget(self.monitors_tab_btn)

        self.windows_tab_btn = QPushButton("Windows")
        self.windows_tab_btn.setCheckable(True)
        self.windows_tab_btn.setFixedHeight(32)
        self.windows_tab_btn.clicked.connect(lambda: self._switch_capture_tab("window"))
        tab_layout.addWidget(self.windows_tab_btn)

        tab_layout.addStretch()

        # Check if window capture is available
        self._window_capture_available = window_manager.is_window_capture_available()
        if not self._window_capture_available:
            self.windows_tab_btn.setEnabled(False)
            self.windows_tab_btn.setToolTip("Window capture requires pywin32 (Windows) or pyobjc (macOS)")

        self._update_tab_styles()

        layout.addLayout(tab_layout)

        # Monitor selection label
        self.capture_mode_label = QLabel("Select which screen to record:")
        self.capture_mode_label.setStyleSheet("font-size: 12px; color: #787774; margin-top: 4px;")
        layout.addWidget(self.capture_mode_label)

        # Scroll area for monitor previews
        self.monitor_scroll = QScrollArea()
        self.monitor_scroll.setWidgetResizable(True)
        self.monitor_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.monitor_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.monitor_scroll.setFixedHeight(280)
        self.monitor_scroll.setStyleSheet("""
            QScrollArea {
                border: 2px solid #E0E0E0;
                border-radius: 12px;
                background-color: #F7F7F5;
            }
            QScrollArea > QWidget > QWidget {
                background-color: #F7F7F5;
            }
        """)

        # Container widget for monitor previews
        self.monitor_container = QWidget()
        self.monitor_container.setStyleSheet("background-color: #F7F7F5;")
        self.monitor_container.setFixedHeight(260)
        self.monitor_layout = QHBoxLayout(self.monitor_container)
        self.monitor_layout.setContentsMargins(10, 8, 10, 8)
        self.monitor_layout.setSpacing(16)
        self.monitor_layout.addStretch()

        self.monitor_scroll.setWidget(self.monitor_container)
        layout.addWidget(self.monitor_scroll)

        # Scroll area for window previews (initially hidden)
        self.window_scroll = QScrollArea()
        self.window_scroll.setWidgetResizable(True)
        self.window_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.window_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.window_scroll.setFixedHeight(280)
        self.window_scroll.setStyleSheet("""
            QScrollArea {
                border: 2px solid #E0E0E0;
                border-radius: 12px;
                background-color: #F7F7F5;
            }
            QScrollArea > QWidget > QWidget {
                background-color: #F7F7F5;
            }
        """)
        self.window_scroll.setVisible(False)  # Hidden initially

        # Container widget for window previews
        self.window_container = QWidget()
        self.window_container.setStyleSheet("background-color: #F7F7F5;")
        self.window_container.setFixedHeight(260)
        self.window_layout = QHBoxLayout(self.window_container)
        self.window_layout.setContentsMargins(10, 8, 10, 8)
        self.window_layout.setSpacing(16)
        self.window_layout.addStretch()

        self.window_scroll.setWidget(self.window_container)
        layout.addWidget(self.window_scroll)

        # Screen recording availability check
        screen_recorder = get_screen_recorder()
        if not screen_recorder.is_available():
            self.screen_recording_check.setEnabled(False)
            self.screen_recording_check.setChecked(False)
            self.monitor_scroll.setEnabled(False)
            self.refresh_btn.setEnabled(False)
            self.screen_recording_check.setToolTip(
                "Screen recording requires FFmpeg. Please install FFmpeg and restart."
            )
            # Show message in monitor area
            no_ffmpeg_label = QLabel("FFmpeg not available - Screen recording disabled")
            no_ffmpeg_label.setAlignment(Qt.AlignCenter)
            no_ffmpeg_label.setStyleSheet("color: #9E9E9E; font-size: 13px;")
            self.monitor_layout.insertWidget(0, no_ffmpeg_label)

        layout.addSpacing(20)

        # ===== Save location section (below screen recording) =====
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

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setEnabled(False)
        self.browse_btn.setFixedHeight(32)
        self.browse_btn.clicked.connect(self._browse_folder)
        self.browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #F7F7F5;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                font-size: 12px;
                color: #37352F;
                padding: 0 12px;
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

        layout.addSpacing(24)

        # ===== Buttons =====
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
                padding: 10px 20px;
                font-size: 14px;
                color: #37352F;
            }
            QPushButton:hover {
                background-color: #EEEEEE;
            }
        """)
        btn_layout.addWidget(cancel_btn)

        btn_layout.addSpacing(12)

        self.create_btn = QPushButton("Start Recording")
        self.create_btn.setMinimumHeight(40)
        self.create_btn.setMinimumWidth(140)
        self.create_btn.clicked.connect(self.accept)
        self.create_btn.setStyleSheet("""
            QPushButton {
                background-color: #DC2626;
                border: none;
                border-radius: 8px;
                padding: 10px 24px;
                font-size: 14px;
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

    def _on_screen_recording_toggled(self, checked: bool):
        """Handle screen recording checkbox toggle"""
        self.monitor_scroll.setEnabled(checked)
        self.refresh_btn.setEnabled(checked)
        for widget in self._monitor_widgets:
            widget.setEnabled(checked)

    def _on_monitor_clicked(self, widget: MonitorPreviewWidget):
        """Handle monitor preview click"""
        # Deselect all
        for w in self._monitor_widgets:
            w.set_selected(False)

        # Select clicked one
        widget.set_selected(True)
        self._selected_monitor_index = self._monitor_widgets.index(widget)

    def _populate_monitors(self):
        """Populate monitor previews"""
        screen_recorder = get_screen_recorder()
        if not screen_recorder.is_available():
            return

        self._monitors = screen_recorder.get_monitors()
        self._monitor_widgets.clear()

        # Clear existing widgets (except stretch)
        while self.monitor_layout.count() > 1:
            item = self.monitor_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        saved_monitor = self._config.config.screen_recording_monitor

        for i, monitor in enumerate(self._monitors):
            widget = MonitorPreviewWidget(monitor)
            widget.clicked.connect(lambda w=widget: self._on_monitor_clicked(w))

            # Load thumbnail with larger size
            thumbnail = ScreenRecorder.capture_monitor_thumbnail(monitor.index, max_width=320, max_height=180)
            widget.set_thumbnail(thumbnail)

            # Select saved monitor or first one
            if monitor.index == saved_monitor:
                widget.set_selected(True)
                self._selected_monitor_index = i
            elif i == 0 and saved_monitor not in [m.index for m in self._monitors]:
                widget.set_selected(True)
                self._selected_monitor_index = 0

            self._monitor_widgets.append(widget)
            self.monitor_layout.insertWidget(i, widget)

        # Update enabled state based on checkbox
        enabled = self.screen_recording_check.isChecked()
        self.monitor_scroll.setEnabled(enabled)
        for widget in self._monitor_widgets:
            widget.setEnabled(enabled)

    def _refresh_thumbnails(self):
        """Refresh all thumbnails (monitors or windows based on current tab)"""
        if self._capture_mode == "monitor":
            for widget in self._monitor_widgets:
                thumbnail = ScreenRecorder.capture_monitor_thumbnail(widget.monitor.index, max_width=320, max_height=180)
                widget.set_thumbnail(thumbnail)
        else:
            # Refresh windows list completely (new windows may have appeared)
            self._populate_windows()

    def _update_tab_styles(self):
        """Update tab button styles based on selection"""
        active_style = """
            QPushButton {
                background-color: #2383E2;
                border: none;
                border-radius: 6px 6px 0 0;
                color: white;
                font-size: 13px;
                font-weight: bold;
                padding: 6px 20px;
            }
        """
        inactive_style = """
            QPushButton {
                background-color: #F7F7F5;
                border: 1px solid #E0E0E0;
                border-bottom: none;
                border-radius: 6px 6px 0 0;
                color: #37352F;
                font-size: 13px;
                padding: 6px 20px;
            }
            QPushButton:hover {
                background-color: #EEEEEE;
            }
            QPushButton:disabled {
                color: #9E9E9E;
                background-color: #F5F5F5;
            }
        """

        if self._capture_mode == "monitor":
            self.monitors_tab_btn.setStyleSheet(active_style)
            self.monitors_tab_btn.setChecked(True)
            self.windows_tab_btn.setStyleSheet(inactive_style)
            self.windows_tab_btn.setChecked(False)
        else:
            self.monitors_tab_btn.setStyleSheet(inactive_style)
            self.monitors_tab_btn.setChecked(False)
            self.windows_tab_btn.setStyleSheet(active_style)
            self.windows_tab_btn.setChecked(True)

    def _switch_capture_tab(self, mode: str):
        """Switch between monitor and window capture tabs"""
        if mode == self._capture_mode:
            return

        self._capture_mode = mode
        self._update_tab_styles()

        if mode == "monitor":
            self.capture_mode_label.setText("Select which screen to record:")
            self.monitor_scroll.setVisible(True)
            self.window_scroll.setVisible(False)
        else:
            self.capture_mode_label.setText("Select which window to record:")
            self.monitor_scroll.setVisible(False)
            self.window_scroll.setVisible(True)
            # Refresh window list when switching to windows tab
            self._populate_windows()

    def _populate_windows(self):
        """Populate window previews"""
        if not self._window_capture_available:
            return

        self._windows = window_manager.get_windows(exclude_own=True)
        self._window_widgets.clear()

        # Clear existing widgets (except stretch)
        while self.window_layout.count() > 1:
            item = self.window_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._windows:
            # Show "no windows" message
            no_windows_label = QLabel("No windows available to capture")
            no_windows_label.setAlignment(Qt.AlignCenter)
            no_windows_label.setStyleSheet("color: #9E9E9E; font-size: 13px;")
            self.window_layout.insertWidget(0, no_windows_label)
            return

        for i, win_info in enumerate(self._windows):
            widget = WindowPreviewWidget(win_info)
            widget.clicked.connect(lambda w=widget: self._on_window_clicked(w))

            # Load thumbnail
            thumbnail = window_manager.get_window_thumbnail(win_info.handle, max_width=320, max_height=180)
            widget.set_thumbnail(thumbnail)

            # Select first window if none selected
            if self._selected_window_handle is None and i == 0:
                widget.set_selected(True)
                self._selected_window_handle = win_info.handle
            elif win_info.handle == self._selected_window_handle:
                widget.set_selected(True)

            self._window_widgets.append(widget)
            self.window_layout.insertWidget(i, widget)

        # Update enabled state based on checkbox
        enabled = self.screen_recording_check.isChecked()
        self.window_scroll.setEnabled(enabled)
        for widget in self._window_widgets:
            widget.setEnabled(enabled)

    def _on_window_clicked(self, widget: WindowPreviewWidget):
        """Handle window preview click"""
        # Deselect all
        for w in self._window_widgets:
            w.set_selected(False)

        # Select clicked one
        widget.set_selected(True)
        self._selected_window_handle = widget.window_info.handle

    def is_screen_recording_enabled(self) -> bool:
        """Check if screen recording is enabled"""
        return self.screen_recording_check.isChecked()

    def get_capture_mode(self) -> str:
        """Get the capture mode ('monitor' or 'window')"""
        return self._capture_mode

    def get_selected_monitor_index(self) -> int:
        """Get the selected monitor index"""
        if 0 <= self._selected_monitor_index < len(self._monitors):
            return self._monitors[self._selected_monitor_index].index
        return 0  # Default to first monitor

    def get_selected_window_handle(self) -> int:
        """Get the selected window handle, or None if no window selected"""
        return self._selected_window_handle

    def get_selected_window_title(self) -> str:
        """Get the selected window title"""
        if self._selected_window_handle is not None:
            for win in self._windows:
                if win.handle == self._selected_window_handle:
                    return win.title
        return ""
