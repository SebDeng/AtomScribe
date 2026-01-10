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
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QAction

from ..signals import get_app_signals
from .widgets.sidebar import SidebarWidget
from .widgets.realtime_panel import RealtimePanel
from .widgets.preview_panel import PreviewPanel
from .widgets.recording_bar import RecordingBar


class MainWindow(QMainWindow):
    """
    Main application window with Notion-style layout.

    Layout:
    - Header bar with logo and actions
    - Left sidebar with session list
    - Right content area (realtime + preview panels)
    - Bottom recording control bar
    - Status bar
    """

    def __init__(self):
        super().__init__()
        self.signals = get_app_signals()

        self._setup_window()
        self._setup_ui()
        self._connect_signals()

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
        self.session_label = QLabel("Session: Untitled")
        status_bar.addPermanentWidget(self.session_label)

        # Separator
        sep = QLabel("•")
        sep.setStyleSheet("color: #D0D0D0; padding: 0 4px;")
        status_bar.addPermanentWidget(sep)

        # Storage info
        self.storage_label = QLabel("2.3 GB available")
        status_bar.addPermanentWidget(self.storage_label)

    def _connect_signals(self):
        """Connect application signals"""
        signals = self.signals

        # Status messages
        signals.status_message.connect(self._show_status_message)

        # Session updates
        signals.session_opened.connect(self._on_session_opened)
        signals.session_created.connect(self._on_session_created)

        # Recording state changes
        signals.recording_started.connect(self._on_recording_started)
        signals.recording_stopped.connect(self._on_recording_stopped)

    def _show_status_message(self, message: str, timeout: int = 3000):
        """Show a message in the status bar"""
        self.status_label.setText(message)
        if timeout > 0:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(timeout, lambda: self.status_label.setText("Ready"))

    def _on_session_opened(self, session_id: str):
        """Handle session opened"""
        self.session_label.setText(f"Session: {session_id}")
        self.setWindowTitle(f"AI Lab Scribe - {session_id}")

    def _on_session_created(self, session_id: str):
        """Handle new session created"""
        self.session_label.setText(f"Session: {session_id}")
        self.setWindowTitle(f"AI Lab Scribe - {session_id}")

    def _on_recording_started(self):
        """Handle recording started"""
        self.setWindowTitle("AI Lab Scribe - Recording...")

    def _on_recording_stopped(self):
        """Handle recording stopped"""
        session = self.session_label.text().replace("Session: ", "")
        if session and session != "Untitled":
            self.setWindowTitle(f"AI Lab Scribe - {session}")
        else:
            self.setWindowTitle("AI Lab Scribe")

    def _on_settings_clicked(self):
        """Handle settings button click"""
        self.signals.status_message.emit("Settings dialog (coming soon)", 2000)

    def _on_about_clicked(self):
        """Handle about button click"""
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(
            self,
            "About AI Lab Scribe",
            "<h3>AI Lab Scribe</h3>"
            "<p>Intelligent Electron Microscope Experiment Recording System</p>"
            "<p>Version 0.1.0</p>"
            "<p>AtomSTEM / Yale University</p>"
            "<p>2026</p>"
        )

    def closeEvent(self, event):
        """Handle window close"""
        # Could add confirmation dialog if recording
        if self.recording_bar._is_recording:
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Recording is in progress. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return

        event.accept()
