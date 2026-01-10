"""File preview panel with tabs for different content types"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QTextBrowser,
    QLineEdit,
    QPushButton,
    QFrame,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from ...signals import get_app_signals


class PreviewPanel(QWidget):
    """
    Panel for previewing transcript files and session data.
    Contains tabs for Transcript, Summary, and Events views.
    Card-style design with header.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = get_app_signals()
        self.setObjectName("previewPanel")

        self._current_file = None

        self._setup_ui()
        self._connect_signals()
        self._load_demo_content()

    def _setup_ui(self):
        """Set up the panel UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header (styled as card header)
        header_widget = QWidget()
        header_widget.setObjectName("previewPanelHeader")
        header_widget.setFixedHeight(52)

        header = QHBoxLayout(header_widget)
        header.setContentsMargins(20, 0, 20, 0)
        header.setSpacing(12)

        # Icon
        icon_label = QLabel("📄")
        icon_label.setStyleSheet("font-size: 16px;")
        header.addWidget(icon_label)

        # Title
        title = QLabel("File Preview")
        title.setObjectName("panelTitle")
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setWeight(QFont.DemiBold)
        title.setFont(title_font)
        title.setStyleSheet("padding: 0; color: #37352F;")
        header.addWidget(title)

        header.addStretch()

        # Search in preview
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("🔍 Search...")
        self.search_box.setMaximumWidth(180)
        self.search_box.setObjectName("searchBox")
        header.addWidget(self.search_box)

        layout.addWidget(header_widget)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setObjectName("previewTabs")
        self.tabs.setDocumentMode(True)

        # Transcript tab
        self.transcript_view = QTextBrowser()
        self.transcript_view.setOpenExternalLinks(False)
        self.tabs.addTab(self.transcript_view, "Transcript")

        # Summary tab
        self.summary_view = QTextBrowser()
        self.tabs.addTab(self.summary_view, "Summary")

        # Events tab
        self.events_view = QTextBrowser()
        self.tabs.addTab(self.events_view, "Events")

        layout.addWidget(self.tabs)

    def _connect_signals(self):
        """Connect to app signals"""
        self.signals.file_selected.connect(self.load_file)
        self.signals.session_opened.connect(self._on_session_opened)

    def _load_demo_content(self):
        """Load demo content for UI preview"""
        # Demo transcript content
        transcript_html = """
        <style>
            body { font-family: 'Segoe UI', sans-serif; line-height: 1.6; color: #37352F; }
            .timestamp { color: #9B9A97; font-size: 12px; font-family: 'Consolas', monospace; }
            .speaker { font-weight: 600; color: #1976D2; }
            .entry {
                margin-bottom: 16px;
                padding: 14px 16px;
                background: #F8F9FA;
                border-radius: 10px;
                border: 1px solid #E8E8E8;
            }
        </style>
        <div class="entry">
            <span class="timestamp">14:30:00</span><br>
            <span class="speaker">Speaker A:</span><br>
            We're starting the STEM session now. First, let's check the beam alignment and adjust the stigmator.
        </div>
        <div class="entry">
            <span class="timestamp">14:30:45</span><br>
            <span class="speaker">Speaker B:</span><br>
            The Y direction needs a bit more adjustment. Let me fine-tune it here.
        </div>
        <div class="entry">
            <span class="timestamp">14:31:20</span><br>
            <span class="speaker">Speaker A:</span><br>
            Looking at this EELS spectrum, there appears to be an artifact around 284 eV.
            This might be carbon contamination from the sample prep.
        </div>
        <div class="entry">
            <span class="timestamp">14:32:05</span><br>
            <span class="speaker">Speaker B:</span><br>
            Agreed. Let's increase the exposure time and see if we can get a cleaner signal.
        </div>
        """
        self.transcript_view.setHtml(transcript_html)

        # Demo summary content
        summary_html = """
        <style>
            body { font-family: 'Segoe UI', sans-serif; line-height: 1.6; color: #37352F; }
            h3 { color: #37352F; border-bottom: 2px solid #E8E8E8; padding-bottom: 8px; margin-top: 16px; }
            ul { padding-left: 20px; }
            li { margin-bottom: 10px; }
            .highlight { background: #E3F2FD; padding: 3px 8px; border-radius: 4px; font-weight: 500; }
            .stat-box {
                display: inline-block;
                background: #F8F9FA;
                border: 1px solid #E8E8E8;
                padding: 8px 14px;
                border-radius: 8px;
                margin-right: 10px;
                margin-bottom: 8px;
            }
            .stat-label { color: #787774; font-size: 12px; }
            .stat-value { font-weight: 600; color: #37352F; font-size: 14px; }
        </style>
        <h3>Session Summary</h3>
        <div>
            <span class="stat-box">
                <span class="stat-label">Duration</span><br>
                <span class="stat-value">00:15:32</span>
            </span>
            <span class="stat-box">
                <span class="stat-label">Speakers</span><br>
                <span class="stat-value">2 participants</span>
            </span>
            <span class="stat-box">
                <span class="stat-label">Events</span><br>
                <span class="stat-value">12 recorded</span>
            </span>
        </div>

        <h3>Key Activities</h3>
        <ul>
            <li>Beam alignment and <span class="highlight">stigmator adjustment</span></li>
            <li>EELS spectrum analysis</li>
            <li>Carbon contamination identified at <span class="highlight">284 eV</span></li>
            <li>Exposure time optimization</li>
        </ul>

        <h3>Parameters Changed</h3>
        <ul>
            <li><strong>Stigmator Y:</strong> -2.3 → +1.1</li>
            <li><strong>Exposure:</strong> 0.5s → 1.0s</li>
        </ul>
        """
        self.summary_view.setHtml(summary_html)

        # Demo events content
        events_html = """
        <style>
            body { font-family: 'Segoe UI', sans-serif; line-height: 1.6; color: #37352F; }
            .event {
                padding: 12px 16px;
                margin-bottom: 10px;
                border-left: 4px solid #2383E2;
                background: #F8F9FA;
                border-radius: 0 8px 8px 0;
            }
            .event-time {
                color: #787774;
                font-size: 12px;
                font-family: 'Consolas', monospace;
                background: #E8E8E8;
                padding: 2px 6px;
                border-radius: 4px;
            }
            .event-type {
                font-weight: 600;
                color: #1976D2;
                margin-left: 8px;
            }
            .event-detail { color: #37352F; margin-top: 4px; }
        </style>
        <div class="event">
            <span class="event-time">14:30:15</span>
            <span class="event-type">Menu Open</span><br>
            <span class="event-detail">TIA → Imaging → Stigmator</span>
        </div>
        <div class="event">
            <span class="event-time">14:30:18</span>
            <span class="event-type">Parameter Change</span><br>
            <span class="event-detail">Stigmator Y: -2.3 → +1.1</span>
        </div>
        <div class="event">
            <span class="event-time">14:31:02</span>
            <span class="event-type">Window Switch</span><br>
            <span class="event-detail">TIA → Velox</span>
        </div>
        <div class="event">
            <span class="event-time">14:31:45</span>
            <span class="event-type">ROI Selection</span><br>
            <span class="event-detail">EELS spectrum region selected</span>
        </div>
        <div class="event">
            <span class="event-time">14:32:30</span>
            <span class="event-type">Parameter Change</span><br>
            <span class="event-detail">Exposure: 0.5s → 1.0s</span>
        </div>
        """
        self.events_view.setHtml(events_html)

    def load_file(self, file_path: str):
        """Load a file for preview"""
        self._current_file = file_path
        # In actual implementation, would load and parse the file
        self.signals.status_message.emit(f"Loaded: {file_path}", 3000)

    def _on_session_opened(self, session_id: str):
        """Handle session opened event"""
        # In actual implementation, would load session data
        self.signals.status_message.emit(f"Session opened: {session_id}", 3000)

    def clear_preview(self):
        """Clear all preview content"""
        self.transcript_view.clear()
        self.summary_view.clear()
        self.events_view.clear()
        self._current_file = None
