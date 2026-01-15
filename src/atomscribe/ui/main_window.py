"""Main application window"""

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QSplitter,
    QStatusBar,
    QLabel,
    QPushButton,
    QFrame,
    QMessageBox,
    QDialog,
)
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QFont, QAction
from loguru import logger

from ..signals import get_app_signals
from ..core.config import get_config_manager
from ..core.recording_controller import get_recording_controller
from ..core.doc_generator import get_document_generator, GenerationMode
from .widgets.sidebar import SidebarWidget
from .widgets.realtime_panel import RealtimePanel
from .widgets.preview_panel import PreviewPanel
from .widgets.recording_bar import RecordingBar
from .dialogs import SettingsDialog


class MainWindow(QMainWindow):
    """
    Main application window with Notion-style layout.

    Layout:
    - Header bar with logo and actions
    - Recording bar
    - Left sidebar with file browser
    - Right content area (realtime + preview panels)
    - Status bar
    """

    def __init__(self):
        super().__init__()
        self.signals = get_app_signals()
        self._config = get_config_manager()
        self._recording_controller = get_recording_controller()

        self._is_actually_recording = False  # Track real recording state

        self._setup_window()
        self._setup_ui()
        self._connect_signals()

        # Check for first run after UI is set up
        QTimer.singleShot(100, self._check_first_run)

        # Preload transcription model in background (after UI is ready)
        QTimer.singleShot(500, self._preload_transcription_model)

        # Preload LLM model in background (after transcription model starts loading)
        QTimer.singleShot(1000, self._preload_llm_model)

        # Preload diarization model in background (after LLM model starts loading)
        QTimer.singleShot(1500, self._preload_diarization_model)

    def _setup_window(self):
        """Configure window properties"""
        self.setWindowTitle("AI Lab Scribe")
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)

        # Center on screen
        screen = self.screen().availableGeometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    def _setup_ui(self):
        """Set up the main UI layout"""
        # Central widget
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        # Main layout
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header bar
        header = self._create_header()
        main_layout.addWidget(header)

        # Recording bar - at the top, below header
        self.recording_bar = RecordingBar()
        main_layout.addWidget(self.recording_bar)

        # Content area (horizontal split)
        content_widget = QWidget()
        content_widget.setObjectName("contentArea")
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Sidebar
        self.sidebar = SidebarWidget()
        content_layout.addWidget(self.sidebar)

        # Right content area with padding for card effect
        right_container = QWidget()
        right_container.setObjectName("contentArea")
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(12, 12, 12, 12)  # Padding around cards
        right_layout.setSpacing(12)  # Gap between cards

        # Splitter for realtime/preview panels
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setObjectName("contentSplitter")
        right_splitter.setChildrenCollapsible(False)
        right_splitter.setHandleWidth(12)

        # Realtime transcript panel (card)
        self.realtime_panel = RealtimePanel()
        right_splitter.addWidget(self.realtime_panel)

        # Preview panel (card)
        self.preview_panel = PreviewPanel()
        right_splitter.addWidget(self.preview_panel)

        # Set initial splitter sizes (60% realtime, 40% preview)
        right_splitter.setSizes([540, 360])
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 1)

        right_layout.addWidget(right_splitter)
        content_layout.addWidget(right_container, stretch=1)

        main_layout.addWidget(content_widget, stretch=1)

        # Status bar
        self._setup_status_bar()

    def _create_header(self) -> QWidget:
        """Create the header bar"""
        header = QWidget()
        header.setObjectName("headerBar")
        header.setFixedHeight(56)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(12)

        # Logo icon (placeholder)
        logo_icon = QLabel("🔬")
        logo_icon.setStyleSheet("font-size: 20px;")
        layout.addWidget(logo_icon)

        # Logo / App name container
        title_container = QVBoxLayout()
        title_container.setContentsMargins(0, 0, 0, 0)
        title_container.setSpacing(0)

        logo_label = QLabel("AI Lab Scribe")
        logo_label.setObjectName("logoLabel")
        logo_font = QFont()
        logo_font.setPointSize(14)
        logo_font.setWeight(QFont.Bold)
        logo_label.setFont(logo_font)
        title_container.addWidget(logo_label)

        subtitle_label = QLabel("Intelligent Experiment Recording")
        subtitle_label.setObjectName("logoSubtitle")
        subtitle_font = QFont()
        subtitle_font.setPointSize(9)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setStyleSheet("color: #9B9A97;")
        title_container.addWidget(subtitle_label)

        layout.addLayout(title_container)

        # Spacer
        layout.addStretch()

        # Header buttons with better styling
        settings_btn = QPushButton("⚙ Settings")
        settings_btn.setObjectName("iconButton")
        settings_btn.clicked.connect(self._on_settings_clicked)
        layout.addWidget(settings_btn)

        about_btn = QPushButton("ℹ About")
        about_btn.setObjectName("iconButton")
        about_btn.clicked.connect(self._on_about_clicked)
        layout.addWidget(about_btn)

        return header

    def _setup_status_bar(self):
        """Set up the status bar"""
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)

        # Status message (left)
        self.status_label = QLabel("Ready")
        status_bar.addWidget(self.status_label)

        # Spacer
        status_bar.addWidget(QWidget(), stretch=1)

        # Session info (right)
        self.session_label = QLabel("Session: None")
        status_bar.addPermanentWidget(self.session_label)

        # Separator
        sep = QLabel("•")
        sep.setStyleSheet("color: #D0D0D0; padding: 0 4px;")
        status_bar.addPermanentWidget(sep)

        # Storage info
        self.storage_label = QLabel("")
        status_bar.addPermanentWidget(self.storage_label)
        self._update_storage_info()

    def _update_storage_info(self):
        """Update storage info in status bar"""
        import shutil
        save_dir = self._config.get_default_save_directory()
        if save_dir and save_dir.exists():
            try:
                usage = shutil.disk_usage(save_dir)
                free_gb = usage.free / (1024 ** 3)
                self.storage_label.setText(f"{free_gb:.1f} GB available")
            except:
                self.storage_label.setText("")
        else:
            self.storage_label.setText("No save location set")

    def _connect_signals(self):
        """Connect application signals"""
        signals = self.signals

        # Status messages
        signals.status_message.connect(self._show_status_message)

        # Session updates
        signals.session_opened.connect(self._on_session_opened)
        signals.session_created.connect(self._on_session_created)

        # Button click signals - user wants to start/stop/pause
        signals.record_button_clicked.connect(self._handle_record_button_clicked)
        signals.stop_button_clicked.connect(self._handle_stop_button_clicked)
        signals.pause_button_clicked.connect(self._handle_pause_button_clicked)

        # Recording saved - for reveal in explorer
        signals.recording_saved.connect(self._on_recording_saved)

        # Recording time updates
        signals.recording_time_updated.connect(self._on_recording_time_updated)

        # Document generation
        signals.doc_generation_requested.connect(self._handle_doc_generation_requested)

    def _check_first_run(self):
        """Check if this is the first run and show setup dialog"""
        if self._config.is_first_run():
            logger.info("First run detected, showing setup dialog")
            self._show_first_run_dialog()
        else:
            # Update sidebar with save directory
            save_dir = self._config.get_default_save_directory()
            if save_dir:
                self.sidebar.set_root_path(str(save_dir))

    def _show_first_run_dialog(self):
        """Show the first run setup dialog"""
        from .dialogs.first_run_dialog import FirstRunDialog

        dialog = FirstRunDialog(self)
        result = dialog.exec()

        if result == QDialog.Accepted:
            path = dialog.get_selected_path()
            logger.info(f"First run setup complete, save directory: {path}")
            self._update_storage_info()

            # Update sidebar to show the new directory
            self.sidebar.set_root_path(path)
            self.signals.status_message.emit("Setup complete! Ready to record.", 3000)
        else:
            # User cancelled - show warning
            QMessageBox.warning(
                self,
                "Setup Required",
                "Please select a save directory to continue.\n"
                "You can set this later in Settings.",
            )

    @Slot()
    def _handle_record_button_clicked(self):
        """Handle REC button click - show dialog then start recording"""
        # If already recording, ignore
        if self._is_actually_recording:
            return

        # Check if configured
        if not self._recording_controller.is_configured():
            self._show_first_run_dialog()
            return

        # Show new session dialog FIRST
        from .dialogs.new_session_dialog import NewSessionDialog

        dialog = NewSessionDialog(self)
        result = dialog.exec()

        if result == QDialog.Accepted:
            session_name = dialog.get_session_name()
            save_dir = dialog.get_save_directory()

            # Apply screen recording settings from dialog
            screen_recording_enabled = dialog.is_screen_recording_enabled()
            capture_mode = dialog.get_capture_mode()

            self._recording_controller.set_screen_recording_enabled(screen_recording_enabled)

            if capture_mode == "window":
                # Window capture mode
                window_handle = dialog.get_selected_window_handle()
                window_title = dialog.get_selected_window_title()
                self._recording_controller.set_screen_recording_window(window_handle, window_title)
                logger.info(f"Screen recording: window mode, handle={window_handle}, title={window_title}")
            else:
                # Monitor capture mode
                selected_monitor = dialog.get_selected_monitor_index()
                self._recording_controller.set_screen_recording_monitor(selected_monitor)
                logger.info(f"Screen recording: monitor mode, monitor={selected_monitor}")

            # Also update the recording bar toggle
            self.recording_bar.set_screen_recording_enabled(screen_recording_enabled)

            logger.info(f"Starting recording: {session_name} in {save_dir}")
            logger.info(f"Screen recording: {'enabled' if screen_recording_enabled else 'disabled'}, mode: {capture_mode}")

            # Actually start recording
            success = self._recording_controller.start_recording(
                session_name=session_name,
                save_directory=save_dir,
            )

            if success:
                self._is_actually_recording = True
                # Get actual session info (name may have been auto-generated)
                current_session = self._recording_controller.get_current_session()
                actual_name = current_session.metadata.name if current_session else session_name
                self._current_session_name = actual_name
                self._current_session_dir = current_session.directory if current_session else save_dir
                self.session_label.setText(f"Session: {actual_name}")
                self.setWindowTitle(f"AI Lab Scribe - Recording: {actual_name}")

                # Reset timer
                self.recording_bar.reset_timer()

                # NOW emit recording_started to update UI
                self.signals.recording_started.emit()

                logger.info(f"Recording started successfully: {actual_name}")
            else:
                self.signals.status_message.emit("Failed to start recording", 3000)
        # If cancelled, button already reset in recording_bar

    @Slot()
    def _handle_stop_button_clicked(self):
        """Handle Stop button click"""
        if not self._is_actually_recording:
            return

        logger.info("Stopping recording")
        audio_path = self._recording_controller.stop_recording()

        self._is_actually_recording = False

        # Emit recording_stopped to update UI
        self.signals.recording_stopped.emit()

        if audio_path and audio_path.exists():
            # Emit saved signal with path
            self.signals.recording_saved.emit(str(audio_path))

            self.setWindowTitle("AI Lab Scribe")

            # Refresh sidebar to show new files
            if hasattr(self, '_current_session_dir') and self._current_session_dir:
                self.sidebar.set_root_path(str(self._current_session_dir.parent))
        else:
            logger.warning("Recording stopped but no audio file found")
            self.signals.status_message.emit("Recording stopped (no audio saved)", 3000)

    @Slot()
    def _handle_pause_button_clicked(self):
        """Handle Pause/Resume button click"""
        if not self._is_actually_recording:
            return

        if self.recording_bar._is_paused:
            # Currently paused, resume
            self._recording_controller.resume_recording()
            self.signals.recording_resumed.emit()
        else:
            # Currently recording, pause
            self._recording_controller.pause_recording()
            self.signals.recording_paused.emit()

    @Slot(str)
    def _on_recording_saved(self, audio_path: str):
        """Handle recording saved - show notification with reveal option"""
        from pathlib import Path
        path = Path(audio_path)

        # Create a clearer info message box with light theme
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Recording Saved")
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText("Recording saved successfully!")
        msg_box.setInformativeText(
            f"Location: {path.parent.name}/\n"
            f"File: {path.name}\n\n"
            f"Would you like to open the folder?"
        )
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.Yes)

        # Force light theme for dialog
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #FFFFFF;
            }
            QMessageBox QLabel {
                color: #37352F;
                background-color: transparent;
            }
            QPushButton {
                background-color: #F7F7F5;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 6px 16px;
                min-width: 60px;
                color: #37352F;
            }
            QPushButton:hover {
                background-color: #EEEEEE;
            }
            QPushButton:default {
                background-color: #2383E2;
                border-color: #2383E2;
                color: white;
            }
        """)

        reply = msg_box.exec()

        if reply == QMessageBox.Yes:
            self._reveal_in_explorer(path.parent)

    @Slot(int)
    def _on_recording_time_updated(self, seconds: int):
        """Handle recording time updates"""
        if self._is_actually_recording:
            self._recording_controller.update_duration(seconds)

    def _reveal_in_explorer(self, path):
        """Open the folder in system file explorer"""
        import subprocess
        import sys
        from pathlib import Path

        path = Path(path)
        if not path.exists():
            logger.warning(f"Path does not exist: {path}")
            return

        try:
            if sys.platform == "win32":
                # Windows - use explorer
                subprocess.run(["explorer", str(path)], check=False)
            elif sys.platform == "darwin":
                # macOS - use Finder
                subprocess.run(["open", str(path)], check=False)
            else:
                # Linux - use xdg-open
                subprocess.run(["xdg-open", str(path)], check=False)
            logger.info(f"Opened folder: {path}")
        except Exception as e:
            logger.error(f"Failed to open folder: {e}")
            self.signals.status_message.emit(f"Could not open folder: {e}", 3000)

    def _show_status_message(self, message: str, timeout: int = 3000):
        """Show a message in the status bar"""
        self.status_label.setText(message)
        if timeout > 0:
            QTimer.singleShot(timeout, lambda: self.status_label.setText("Ready"))

    def _on_session_opened(self, session_id: str):
        """Handle session opened"""
        self.session_label.setText(f"Session: {session_id}")
        self.setWindowTitle(f"AI Lab Scribe - {session_id}")

    def _on_session_created(self, session_id: str):
        """Handle new session created"""
        self.session_label.setText(f"Session: {session_id}")
        self.setWindowTitle(f"AI Lab Scribe - {session_id}")

    def _on_settings_clicked(self):
        """Handle settings button click"""
        dialog = SettingsDialog(self)
        dialog.exec()

    def _on_about_clicked(self):
        """Handle about button click"""
        QMessageBox.about(
            self,
            "About AI Lab Scribe",
            "<h3>AI Lab Scribe</h3>"
            "<p>Intelligent Electron Microscope Experiment Recording System</p>"
            "<p>Version 0.1.0</p>"
            "<p>AtomE Corp</p>"
            "<p>2026</p>"
        )

    def _preload_transcription_model(self):
        """Preload the transcription model in background for faster first recording"""
        try:
            self._recording_controller.preload_transcription_model()
        except Exception as e:
            logger.warning(f"Failed to preload transcription model: {e}")
            # Non-fatal - model will load on first recording

    def _preload_llm_model(self):
        """Preload the LLM model in background for transcript correction"""
        try:
            self._recording_controller.preload_llm_model()
        except Exception as e:
            logger.warning(f"Failed to preload LLM model: {e}")
            # Non-fatal - LLM correction will be disabled if model not available

    def _preload_diarization_model(self):
        """Preload the speaker diarization model in background"""
        try:
            self._recording_controller.preload_diarization_model()
        except Exception as e:
            logger.warning(f"Failed to preload diarization model: {e}")
            # Non-fatal - diarization will be disabled if model not available

    @Slot(object)
    def _handle_doc_generation_requested(self, session):
        """Handle document generation request after recording stops."""
        from .dialogs.generation_mode_dialog import GenerationModeDialog
        from .dialogs.generation_progress_dialog import GenerationProgressDialog

        logger.info(f"Document generation requested for session: {session.metadata.name}")

        config = self._config.config

        # Check if we should show the dialog or use saved preference
        if config.doc_generation_show_dialog:
            # Show mode selection dialog
            dialog = GenerationModeDialog(self)
            dialog.mode_selected.connect(
                lambda mode, remember: self._start_document_generation(session, mode)
            )

            result = dialog.exec()
            if result != QDialog.Accepted:
                logger.info("Document generation skipped by user")
                return
        else:
            # Use default mode from config
            mode_str = config.doc_generation_default_mode
            mode = GenerationMode(mode_str) if mode_str in ["training", "experiment_log"] else GenerationMode.TRAINING
            self._start_document_generation(session, mode)

    def _start_document_generation(self, session, mode: GenerationMode):
        """Start document generation with progress dialog."""
        from .dialogs.generation_progress_dialog import GenerationProgressDialog

        logger.info(f"Starting document generation: mode={mode.value}")

        # Get document generator
        doc_gen = get_document_generator()

        # Create and show progress dialog
        progress_dialog = GenerationProgressDialog(self)
        progress_dialog.cancel_requested.connect(doc_gen.cancel)

        # Start generation
        doc_gen.generate_async(session, mode)

        # Show progress dialog (non-blocking)
        progress_dialog.show()

    def closeEvent(self, event):
        """Handle window close"""
        # Check if recording
        if self._is_actually_recording:
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Recording is in progress. Stop recording and exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return

            # Stop recording before closing
            self._recording_controller.stop_recording()

        event.accept()
