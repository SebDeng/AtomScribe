"""Real-time transcript display panel with message bubbles"""

from datetime import datetime
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QScrollArea,
    QPushButton,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from ...signals import get_app_signals
from ...styles.colors import NotionColors


class MessageBubble(QFrame):
    """Single transcript message bubble"""

    def __init__(self, speaker: str, text: str, timestamp: datetime, parent=None):
        super().__init__(parent)
        self.setObjectName("messageBubble")
        self.setProperty("speaker", speaker)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        # Header with speaker and time
        header = QHBoxLayout()
        header.setSpacing(12)

        speaker_label = QLabel(f"Speaker {speaker}")
        speaker_label.setObjectName("speakerLabel")
        speaker_font = QFont()
        speaker_font.setWeight(QFont.Bold)
        speaker_label.setFont(speaker_font)
        header.addWidget(speaker_label)

        time_label = QLabel(timestamp.strftime("%H:%M:%S"))
        time_label.setObjectName("timestampLabel")
        header.addWidget(time_label)

        header.addStretch()
        layout.addLayout(header)

        # Message text
        self.text_label = QLabel(text)
        self.text_label.setObjectName("transcriptText")
        self.text_label.setWordWrap(True)
        text_font = QFont()
        text_font.setPointSize(10)
        self.text_label.setFont(text_font)
        layout.addWidget(self.text_label)

    def update_text(self, text: str):
        """Update the message text (for streaming updates)"""
        self.text_label.setText(text)


class RealtimePanel(QWidget):
    """
    Panel displaying real-time transcript with message bubbles.
    Supports streaming updates and auto-scroll.
    Card-style design with header.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = get_app_signals()
        self.setObjectName("realtimePanel")

        self._messages: dict[str, MessageBubble] = {}
        self._auto_scroll = True

        self._setup_ui()
        self._connect_signals()
        self._add_demo_messages()

    def _setup_ui(self):
        """Set up the panel UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header (styled as card header)
        header_widget = QWidget()
        header_widget.setObjectName("realtimePanelHeader")
        header_widget.setFixedHeight(52)

        header = QHBoxLayout(header_widget)
        header.setContentsMargins(20, 0, 20, 0)
        header.setSpacing(12)

        # Icon
        icon_label = QLabel("💬")
        icon_label.setStyleSheet("font-size: 16px;")
        header.addWidget(icon_label)

        # Title
        title = QLabel("Realtime Transcript")
        title.setObjectName("panelTitle")
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setWeight(QFont.DemiBold)
        title.setFont(title_font)
        title.setStyleSheet("padding: 0; color: #37352F;")
        header.addWidget(title)

        header.addStretch()

        # Auto-scroll toggle
        self.auto_scroll_btn = QPushButton("↓ Auto-scroll")
        self.auto_scroll_btn.setObjectName("iconButton")
        self.auto_scroll_btn.setCheckable(True)
        self.auto_scroll_btn.setChecked(True)
        self.auto_scroll_btn.clicked.connect(self._toggle_auto_scroll)
        header.addWidget(self.auto_scroll_btn)

        layout.addWidget(header_widget)

        # Scroll area for messages
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("transcriptScroll")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QFrame.NoFrame)

        # Container for messages
        self.messages_container = QWidget()
        self.messages_container.setStyleSheet("background-color: #FFFFFF;")
        self.messages_layout = QVBoxLayout(self.messages_container)
        self.messages_layout.setContentsMargins(0, 12, 0, 20)
        self.messages_layout.setSpacing(8)
        self.messages_layout.addStretch()

        self.scroll_area.setWidget(self.messages_container)
        layout.addWidget(self.scroll_area)

    def _connect_signals(self):
        """Connect to app signals"""
        self.signals.transcript_received.connect(self._on_transcript_received)
        self.signals.transcript_updated.connect(self._on_transcript_updated)
        self.signals.recording_started.connect(self._on_recording_started)
        self.signals.recording_stopped.connect(self._on_recording_stopped)

    def _add_demo_messages(self):
        """Add demo messages for UI preview"""
        demo_messages = [
            ("A", "We're now going to adjust the stigmator. Let's see how the beam looks after this...", datetime(2026, 1, 10, 14, 30, 12)),
            ("B", "I'll adjust the Y direction a bit more. The astigmatism should be better now.", datetime(2026, 1, 10, 14, 30, 45)),
            ("A", "Looking at this EELS spectrum, there seems to be an artifact around 532 eV. Could be from oxygen contamination.", datetime(2026, 1, 10, 14, 31, 20)),
            ("B", "Let me check the vacuum level. It was at 2.3e-7 this morning.", datetime(2026, 1, 10, 14, 31, 55)),
        ]

        for speaker, text, timestamp in demo_messages:
            self.add_message(f"demo_{speaker}_{timestamp.timestamp()}", speaker, text, timestamp)

    def add_message(self, msg_id: str, speaker: str, text: str, timestamp: datetime = None):
        """Add a new message bubble"""
        if timestamp is None:
            timestamp = datetime.now()

        bubble = MessageBubble(speaker, text, timestamp)
        self._messages[msg_id] = bubble

        # Insert before the stretch
        count = self.messages_layout.count()
        self.messages_layout.insertWidget(count - 1, bubble)

        # Auto-scroll to bottom
        if self._auto_scroll:
            QTimer.singleShot(50, self._scroll_to_bottom)

    def update_message(self, msg_id: str, text: str):
        """Update an existing message (for streaming)"""
        if msg_id in self._messages:
            self._messages[msg_id].update_text(text)
            if self._auto_scroll:
                QTimer.singleShot(50, self._scroll_to_bottom)

    def clear_messages(self):
        """Clear all messages"""
        for bubble in self._messages.values():
            bubble.deleteLater()
        self._messages.clear()

    def _toggle_auto_scroll(self, checked: bool):
        """Toggle auto-scroll behavior"""
        self._auto_scroll = checked
        if checked:
            self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        """Scroll to the bottom of the message list"""
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_transcript_received(self, data: dict):
        """Handle new transcript from signal"""
        msg_id = data.get("id", str(datetime.now().timestamp()))
        speaker = data.get("speaker", "A")
        text = data.get("text", "")
        timestamp = data.get("timestamp")

        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        self.add_message(msg_id, speaker, text, timestamp)

    def _on_transcript_updated(self, msg_id: str, text: str):
        """Handle transcript update from signal"""
        self.update_message(msg_id, text)

    def _on_recording_started(self):
        """Handle recording start"""
        # Could add a visual indicator here
        pass

    def _on_recording_stopped(self):
        """Handle recording stop"""
        pass
