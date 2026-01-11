"""Real-time transcript display panel with message bubbles"""

import threading
import numpy as np
from datetime import datetime
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QScrollArea,
    QPushButton,
)
from PySide6.QtCore import Qt, QTimer, Slot, Signal
from PySide6.QtGui import QFont
from loguru import logger

from ...signals import get_app_signals
from ...styles.colors import NotionColors
from ...core.transcriber import TranscriptSegment, Word

# Audio playback imports
try:
    import sounddevice as sd
    import wave
    AUDIO_PLAYBACK_AVAILABLE = True
except ImportError:
    AUDIO_PLAYBACK_AVAILABLE = False
    logger.warning("sounddevice not available, audio playback disabled")


def split_segment_by_speakers(segment: TranscriptSegment, speaker_segments: list) -> list:
    """
    Split a transcript segment into conversation parts based on speaker time ranges.

    Args:
        segment: TranscriptSegment with words containing timing info
        speaker_segments: List of SpeakerSegment objects with start_time, end_time, speaker_label

    Returns:
        List of dicts: [{"speaker": str, "text": str, "start": float, "end": float}, ...]
    """
    if not segment.words or not speaker_segments:
        # Fallback: return the whole text with primary speaker
        primary = speaker_segments[0].speaker_label if speaker_segments else "Unknown"
        return [{"speaker": primary, "text": segment.text, "start": segment.start, "end": segment.end}]

    # Create conversation parts by assigning each word to a speaker segment
    parts = []
    current_part = None

    for word in segment.words:
        word_mid = (word.start + word.end) / 2  # Use word midpoint for assignment

        # Find which speaker segment this word belongs to
        assigned_speaker = None
        for spk_seg in speaker_segments:
            if spk_seg.start_time <= word_mid <= spk_seg.end_time:
                assigned_speaker = spk_seg.speaker_label
                break

        # Fallback: assign to nearest speaker segment
        if assigned_speaker is None:
            min_dist = float('inf')
            for spk_seg in speaker_segments:
                dist = min(abs(word_mid - spk_seg.start_time), abs(word_mid - spk_seg.end_time))
                if dist < min_dist:
                    min_dist = dist
                    assigned_speaker = spk_seg.speaker_label

        if assigned_speaker is None:
            assigned_speaker = speaker_segments[0].speaker_label

        # Add word to current part or start new part
        if current_part is None or current_part["speaker"] != assigned_speaker:
            # Start new part
            if current_part is not None:
                parts.append(current_part)
            current_part = {
                "speaker": assigned_speaker,
                "text": word.word,
                "start": word.start,
                "end": word.end,
            }
        else:
            # Continue current part - use smart spacing
            prev_text = current_part["text"]
            word_text = word.word

            # Check if we need space between words
            if prev_text and word_text:
                last_char = prev_text[-1]
                first_char = word_text[0]

                # Check for CJK characters
                def is_cjk(c):
                    return 0x4E00 <= ord(c) <= 0x9FFF or 0x3000 <= ord(c) <= 0x303F

                # Don't add space between CJK, or before punctuation
                if first_char in "',.!?;:，。！？；：、":
                    current_part["text"] += word_text
                elif is_cjk(last_char) and is_cjk(first_char):
                    current_part["text"] += word_text
                elif last_char in "，。！？；：、（）【】":
                    current_part["text"] += word_text
                else:
                    current_part["text"] += " " + word_text
            else:
                current_part["text"] += word_text
            current_part["end"] = word.end

    # Add final part
    if current_part is not None:
        parts.append(current_part)

    return parts


class TranscriptBubble(QFrame):
    """Single transcript segment bubble with streaming and correction support"""

    # Signal emitted when play button is clicked (start_time, end_time)
    play_requested = Signal(float, float)

    # Speaker color mapping
    SPEAKER_COLORS = {
        "Speaker A": "#2383E2",  # Blue
        "Speaker B": "#0F7B6C",  # Green/Teal
        "Speaker C": "#D9730D",  # Orange
        "Speaker D": "#9B51E0",  # Purple
        "Speaker E": "#E03E3E",  # Red
        "Speaker F": "#0E7C86",  # Cyan
        "Speaker G": "#AD5700",  # Brown
        "Speaker H": "#5E5E5E",  # Gray
    }

    def __init__(self, text: str, start_time: float, end_time: float = None, is_partial: bool = False, speaker: str = None, segment: TranscriptSegment = None, parent=None):
        super().__init__(parent)
        self.setObjectName("transcriptBubble")
        self._start_time = start_time
        self._end_time = end_time
        self._is_partial = is_partial
        self._is_corrected = False
        self._original_text = text
        self._corrected_text = ""
        self._showing_original = False  # Toggle state
        self._is_playing = False
        self._speaker = speaker
        self._has_multiple_speakers = False
        self._speaker_sequence = ""
        self._segment = segment  # Store full segment for word-level access

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)

        # Header row with timestamp and status
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        # Play button
        self.play_button = QPushButton("▶")
        self.play_button.setObjectName("playButton")
        self.play_button.setCursor(Qt.PointingHandCursor)
        self.play_button.setFixedSize(24, 24)
        self.play_button.setStyleSheet("""
            QPushButton#playButton {
                color: #6B7280;
                font-size: 10px;
                background: transparent;
                border: 1px solid #E5E7EB;
                border-radius: 12px;
                padding: 0px;
            }
            QPushButton#playButton:hover {
                background-color: #F3F4F6;
                border-color: #D1D5DB;
                color: #374151;
            }
            QPushButton#playButton:pressed {
                background-color: #E5E7EB;
            }
        """)
        self.play_button.setToolTip("Play this segment")
        self.play_button.clicked.connect(self._on_play_clicked)
        header_layout.addWidget(self.play_button)

        # Speaker label (if speaker is assigned)
        self.speaker_label = QLabel("")
        self.speaker_label.setObjectName("speakerLabel")
        if speaker:
            self._update_speaker_label(speaker)
        else:
            self.speaker_label.setVisible(False)
        header_layout.addWidget(self.speaker_label)

        # Timestamp
        self.time_label = QLabel(self._format_timestamp())
        self.time_label.setObjectName("timestampLabel")
        self.time_label.setStyleSheet("color: #9B9A97; font-size: 11px;")
        header_layout.addWidget(self.time_label)

        header_layout.addStretch()

        # Correction indicator (clickable, hidden by default)
        self.correction_indicator = QPushButton("✓ AI")
        self.correction_indicator.setObjectName("correctionIndicator")
        self.correction_indicator.setCursor(Qt.PointingHandCursor)
        self.correction_indicator.setStyleSheet("""
            QPushButton#correctionIndicator {
                color: #10B981;
                font-size: 10px;
                font-weight: bold;
                background: transparent;
                border: none;
                padding: 2px 6px;
                border-radius: 3px;
            }
            QPushButton#correctionIndicator:hover {
                background-color: #D1FAE5;
            }
        """)
        self.correction_indicator.setToolTip("Click to show original text")
        self.correction_indicator.setVisible(False)
        self.correction_indicator.clicked.connect(self._toggle_original)
        header_layout.addWidget(self.correction_indicator)

        layout.addLayout(header_layout)

        # Transcript text
        self.text_label = QLabel(text)
        self.text_label.setObjectName("transcriptText")
        self.text_label.setWordWrap(True)
        self.text_label.setStyleSheet("color: #37352F; font-size: 13px; line-height: 1.5;")
        layout.addWidget(self.text_label)

        # Apply styling based on partial state
        self._update_style()

    def _format_timestamp(self) -> str:
        """Format the timestamp display"""
        timestamp_str = self._format_time(self._start_time)
        if self._end_time and self._end_time != self._start_time:
            timestamp_str += f" - {self._format_time(self._end_time)}"
        if self._is_partial:
            timestamp_str += " ..."
        return timestamp_str

    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _update_style(self):
        """Update bubble style based on state (partial, corrected)"""
        if self._is_partial:
            # Streaming/partial - blue border
            self.setStyleSheet("""
                QFrame#transcriptBubble {
                    background-color: #FEFEFE;
                    border: 1px solid #2383E2;
                    border-radius: 8px;
                    margin: 4px 16px;
                }
            """)
        elif self._is_corrected:
            # LLM corrected - green tint
            self.setStyleSheet("""
                QFrame#transcriptBubble {
                    background-color: #F0FDF4;
                    border: 1px solid #86EFAC;
                    border-radius: 8px;
                    margin: 4px 16px;
                }
            """)
        else:
            # Complete segment - default gray
            self.setStyleSheet("""
                QFrame#transcriptBubble {
                    background-color: #F7F7F5;
                    border: 1px solid #E8E8E6;
                    border-radius: 8px;
                    margin: 4px 16px;
                }
            """)

    def update_segment(self, text: str, end_time: float = None, is_partial: bool = False):
        """Update the segment text and state (for streaming)"""
        self.text_label.setText(text)
        if end_time is not None:
            self._end_time = end_time
        self._is_partial = is_partial
        self.time_label.setText(self._format_timestamp())
        self._update_style()

    def update_text(self, text: str):
        """Update the transcript text (legacy method)"""
        self.text_label.setText(text)

    def apply_correction(self, corrected_text: str):
        """Apply LLM correction to this bubble"""
        self._original_text = self.text_label.text()
        self._corrected_text = corrected_text
        self.text_label.setText(corrected_text)
        self._is_corrected = True
        self._showing_original = False
        self.correction_indicator.setVisible(True)
        self.correction_indicator.setText("✓ AI")
        self.correction_indicator.setToolTip("Click to show original text")
        self._update_style()

    def _toggle_original(self):
        """Toggle between showing original and corrected text"""
        if not self._is_corrected:
            return

        self._showing_original = not self._showing_original

        if self._showing_original:
            # Show original text
            self.text_label.setText(self._original_text)
            self.correction_indicator.setText("Original")
            self.correction_indicator.setToolTip("Click to show AI-corrected text")
            self.correction_indicator.setStyleSheet("""
                QPushButton#correctionIndicator {
                    color: #9B9A97;
                    font-size: 10px;
                    font-weight: bold;
                    background: transparent;
                    border: none;
                    padding: 2px 6px;
                    border-radius: 3px;
                }
                QPushButton#correctionIndicator:hover {
                    background-color: #E8E8E6;
                }
            """)
            # Update bubble style to show it's displaying original
            self.setStyleSheet("""
                QFrame#transcriptBubble {
                    background-color: #F7F7F5;
                    border: 1px solid #E8E8E6;
                    border-radius: 8px;
                    margin: 4px 16px;
                }
            """)
        else:
            # Show corrected text
            self.text_label.setText(self._corrected_text)
            self.correction_indicator.setText("✓ AI")
            self.correction_indicator.setToolTip("Click to show original text")
            self.correction_indicator.setStyleSheet("""
                QPushButton#correctionIndicator {
                    color: #10B981;
                    font-size: 10px;
                    font-weight: bold;
                    background: transparent;
                    border: none;
                    padding: 2px 6px;
                    border-radius: 3px;
                }
                QPushButton#correctionIndicator:hover {
                    background-color: #D1FAE5;
                }
            """)
            # Restore corrected style
            self._update_style()

    def is_corrected(self) -> bool:
        """Check if this bubble has been corrected"""
        return self._is_corrected

    def get_original_text(self) -> str:
        """Get the original (uncorrected) text"""
        return self._original_text

    def get_corrected_text(self) -> str:
        """Get the corrected text"""
        return self._corrected_text

    def _on_play_clicked(self):
        """Handle play button click"""
        if self._is_playing:
            # Stop playback
            self.stop_playback()
        else:
            # Request playback
            end_time = self._end_time if self._end_time else self._start_time + 10
            self.play_requested.emit(self._start_time, end_time)

    def set_playing(self, playing: bool):
        """Update the play button state"""
        self._is_playing = playing
        if playing:
            self.play_button.setText("■")
            self.play_button.setToolTip("Stop playback")
            self.play_button.setStyleSheet("""
                QPushButton#playButton {
                    color: #DC2626;
                    font-size: 10px;
                    background: #FEE2E2;
                    border: 1px solid #FECACA;
                    border-radius: 12px;
                    padding: 0px;
                }
                QPushButton#playButton:hover {
                    background-color: #FECACA;
                    border-color: #F87171;
                }
            """)
        else:
            self.play_button.setText("▶")
            self.play_button.setToolTip("Play this segment")
            self.play_button.setStyleSheet("""
                QPushButton#playButton {
                    color: #6B7280;
                    font-size: 10px;
                    background: transparent;
                    border: 1px solid #E5E7EB;
                    border-radius: 12px;
                    padding: 0px;
                }
                QPushButton#playButton:hover {
                    background-color: #F3F4F6;
                    border-color: #D1D5DB;
                    color: #374151;
                }
            """)

    def stop_playback(self):
        """Stop any current playback"""
        if AUDIO_PLAYBACK_AVAILABLE:
            sd.stop()
        self.set_playing(False)

    def get_time_range(self) -> tuple:
        """Get the start and end time of this segment"""
        return (self._start_time, self._end_time)

    def _update_speaker_label(self, speaker: str, has_multiple: bool = False, sequence: str = ""):
        """Update the speaker label with color"""
        color = self.SPEAKER_COLORS.get(speaker, "#787774")

        if has_multiple and sequence:
            # Show speaker sequence for multi-speaker segments
            display_text = sequence
            # Use primary speaker's color
            self.speaker_label.setStyleSheet(f"""
                QLabel#speakerLabel {{
                    color: {color};
                    font-size: 11px;
                    font-weight: bold;
                    padding: 2px 6px;
                    background-color: #FEF3C7;
                    border: 1px solid #F59E0B;
                    border-radius: 3px;
                }}
            """)
        else:
            display_text = speaker
            self.speaker_label.setStyleSheet(f"""
                QLabel#speakerLabel {{
                    color: {color};
                    font-size: 11px;
                    font-weight: bold;
                    padding: 2px 6px;
                    background-color: {color}20;
                    border-radius: 3px;
                }}
            """)

        self.speaker_label.setText(display_text)
        self.speaker_label.setVisible(True)

    def set_speaker(self, speaker: str):
        """Set or update the speaker for this bubble (single speaker)"""
        self._speaker = speaker
        self._has_multiple_speakers = False
        self._speaker_sequence = ""
        self._update_speaker_label(speaker)

    def set_speakers(self, primary_speaker: str, has_multiple: bool, sequence: str):
        """Set or update speakers for this bubble (supports multi-speaker)"""
        self._speaker = primary_speaker
        self._has_multiple_speakers = has_multiple
        self._speaker_sequence = sequence
        self._update_speaker_label(primary_speaker, has_multiple, sequence)

    def get_speaker(self) -> str:
        """Get the primary speaker"""
        return self._speaker

    def has_multiple_speakers(self) -> bool:
        """Check if this bubble has multiple speakers"""
        return getattr(self, '_has_multiple_speakers', False)

    def set_segment(self, segment: TranscriptSegment):
        """Store the full segment for word-level access"""
        self._segment = segment

    def get_segment(self) -> TranscriptSegment:
        """Get the stored segment"""
        return self._segment


class ConversationSubBubble(QFrame):
    """A single speaker's portion in a conversation-style display"""

    # Signal emitted when play button is clicked (start_time, end_time)
    play_requested = Signal(float, float)

    # Speaker color mapping (same as TranscriptBubble)
    SPEAKER_COLORS = {
        "Speaker A": "#2383E2",  # Blue
        "Speaker B": "#0F7B6C",  # Green/Teal
        "Speaker C": "#D9730D",  # Orange
        "Speaker D": "#9B51E0",  # Purple
        "Speaker E": "#E03E3E",  # Red
        "Speaker F": "#0E7C86",  # Cyan
        "Speaker G": "#AD5700",  # Brown
        "Speaker H": "#5E5E5E",  # Gray
    }

    def __init__(self, speaker: str, text: str, start_time: float, end_time: float, is_left: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("conversationSubBubble")
        self._speaker = speaker
        self._start_time = start_time
        self._end_time = end_time
        self._is_playing = False

        color = self.SPEAKER_COLORS.get(speaker, "#787774")

        # Main horizontal layout for alignment
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 4, 8, 4)
        main_layout.setSpacing(0)

        # Create the bubble container
        bubble_container = QFrame()
        bubble_container.setObjectName("subBubbleContainer")
        bubble_layout = QVBoxLayout(bubble_container)
        bubble_layout.setContentsMargins(12, 8, 12, 8)
        bubble_layout.setSpacing(4)

        # Header with speaker and play button
        header_layout = QHBoxLayout()
        header_layout.setSpacing(6)

        # Play button
        self.play_button = QPushButton("▶")
        self.play_button.setObjectName("subPlayButton")
        self.play_button.setCursor(Qt.PointingHandCursor)
        self.play_button.setFixedSize(20, 20)
        self.play_button.setStyleSheet(f"""
            QPushButton#subPlayButton {{
                color: {color};
                font-size: 9px;
                background: transparent;
                border: 1px solid {color}40;
                border-radius: 10px;
                padding: 0px;
            }}
            QPushButton#subPlayButton:hover {{
                background-color: {color}20;
            }}
        """)
        self.play_button.setToolTip("Play this portion")
        self.play_button.clicked.connect(self._on_play_clicked)
        header_layout.addWidget(self.play_button)

        # Speaker label
        speaker_label = QLabel(speaker)
        speaker_label.setStyleSheet(f"""
            color: {color};
            font-size: 10px;
            font-weight: bold;
        """)
        header_layout.addWidget(speaker_label)

        header_layout.addStretch()

        bubble_layout.addLayout(header_layout)

        # Text content
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        text_label.setStyleSheet("color: #37352F; font-size: 13px;")
        bubble_layout.addWidget(text_label)

        # Style the bubble with speaker color
        bg_color = f"{color}10"  # Very light tint
        border_color = f"{color}40"  # Lighter border
        bubble_container.setStyleSheet(f"""
            QFrame#subBubbleContainer {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 12px;
            }}
        """)

        # Set max width to 85% for conversation style
        bubble_container.setMaximumWidth(600)

        # Add to main layout with alignment
        if is_left:
            main_layout.addWidget(bubble_container)
            main_layout.addStretch()
        else:
            main_layout.addStretch()
            main_layout.addWidget(bubble_container)

    def _on_play_clicked(self):
        """Handle play button click"""
        if self._is_playing:
            self.stop_playback()
        else:
            self.play_requested.emit(self._start_time, self._end_time)

    def set_playing(self, playing: bool):
        """Update the play button state"""
        self._is_playing = playing
        color = self.SPEAKER_COLORS.get(self._speaker, "#787774")
        if playing:
            self.play_button.setText("■")
            self.play_button.setStyleSheet(f"""
                QPushButton#subPlayButton {{
                    color: #DC2626;
                    font-size: 9px;
                    background: #FEE2E2;
                    border: 1px solid #FECACA;
                    border-radius: 10px;
                    padding: 0px;
                }}
            """)
        else:
            self.play_button.setText("▶")
            self.play_button.setStyleSheet(f"""
                QPushButton#subPlayButton {{
                    color: {color};
                    font-size: 9px;
                    background: transparent;
                    border: 1px solid {color}40;
                    border-radius: 10px;
                    padding: 0px;
                }}
                QPushButton#subPlayButton:hover {{
                    background-color: {color}20;
                }}
            """)

    def stop_playback(self):
        """Stop any current playback"""
        if AUDIO_PLAYBACK_AVAILABLE:
            sd.stop()
        self.set_playing(False)

    def get_time_range(self) -> tuple:
        """Get the start and end time of this sub-bubble"""
        return (self._start_time, self._end_time)


class ConversationContainer(QFrame):
    """Container for multi-speaker conversation-style display"""

    # Signal emitted when play button is clicked (start_time, end_time)
    play_requested = Signal(float, float)

    def __init__(self, segment_id: int, start_time: float, end_time: float, parent=None):
        super().__init__(parent)
        self.setObjectName("conversationContainer")
        self._segment_id = segment_id
        self._start_time = start_time
        self._end_time = end_time
        self._sub_bubbles: list[ConversationSubBubble] = []

        self.setStyleSheet("""
            QFrame#conversationContainer {
                background-color: #FFFFFF;
                border: 1px solid #E8E8E6;
                border-radius: 8px;
                margin: 4px 16px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Header with segment timestamp
        header_layout = QHBoxLayout()
        self.time_label = QLabel(self._format_timestamp())
        self.time_label.setStyleSheet("color: #9B9A97; font-size: 10px;")
        header_layout.addWidget(self.time_label)
        header_layout.addStretch()

        # Conversation indicator
        conv_label = QLabel("💬 Conversation")
        conv_label.setStyleSheet("color: #F59E0B; font-size: 10px; font-weight: bold;")
        header_layout.addWidget(conv_label)

        layout.addLayout(header_layout)

        # Container for sub-bubbles
        self.bubbles_container = QWidget()
        self.bubbles_layout = QVBoxLayout(self.bubbles_container)
        self.bubbles_layout.setContentsMargins(0, 4, 0, 0)
        self.bubbles_layout.setSpacing(4)
        layout.addWidget(self.bubbles_container)

    def _format_timestamp(self) -> str:
        """Format the timestamp display"""
        return f"{self._format_time(self._start_time)} - {self._format_time(self._end_time)}"

    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def add_speaker_part(self, speaker: str, text: str, start_time: float, end_time: float):
        """Add a speaker's portion to the conversation"""
        # Alternate left/right based on speaker
        speaker_idx = hash(speaker) % 2  # 0 or 1
        is_left = (speaker_idx == 0)

        sub_bubble = ConversationSubBubble(speaker, text, start_time, end_time, is_left)
        sub_bubble.play_requested.connect(self.play_requested.emit)
        self._sub_bubbles.append(sub_bubble)
        self.bubbles_layout.addWidget(sub_bubble)

    def get_sub_bubbles(self) -> list:
        """Get all sub-bubbles"""
        return self._sub_bubbles

    def get_time_range(self) -> tuple:
        """Get the start and end time of this container"""
        return (self._start_time, self._end_time)


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

        self._bubbles: dict[int, TranscriptBubble] = {}  # segment_id -> bubble
        self._segments: dict[int, TranscriptSegment] = {}  # segment_id -> segment (for word-level access)
        self._conversation_containers: dict[int, ConversationContainer] = {}  # segment_id -> container
        self._auto_scroll = True
        self._is_recording = False
        self._audio_file_path: Path = None
        self._currently_playing_bubble = None  # Can be TranscriptBubble or ConversationSubBubble
        self._playback_thread: threading.Thread = None

        self._setup_ui()
        self._connect_signals()

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

        # Status indicator
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #9B9A97; font-size: 11px;")
        header.addWidget(self.status_label)

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

        # Show placeholder when empty
        self._show_placeholder()

    def _connect_signals(self):
        """Connect to app signals"""
        self.signals.transcript_segment.connect(self._on_transcript_segment)
        self.signals.transcript_segment_updated.connect(self._on_transcript_segment_updated)
        self.signals.transcript_segment_corrected.connect(self._on_transcript_segment_corrected)
        self.signals.recording_started.connect(self._on_recording_started)
        self.signals.recording_stopped.connect(self._on_recording_stopped)
        self.signals.transcription_model_loaded.connect(self._on_model_loaded)
        self.signals.recording_saved.connect(self._on_recording_saved)
        self.signals.speaker_identified.connect(self._on_speaker_identified)

    def _show_placeholder(self):
        """Show placeholder text when no transcript"""
        if hasattr(self, '_placeholder') and self._placeholder:
            return

        self._placeholder = QLabel("Transcript will appear here when recording starts...")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet("color: #9B9A97; font-size: 13px; padding: 40px;")

        # Insert before stretch
        count = self.messages_layout.count()
        self.messages_layout.insertWidget(count - 1, self._placeholder)

    def _hide_placeholder(self):
        """Hide the placeholder"""
        if hasattr(self, '_placeholder') and self._placeholder:
            self._placeholder.deleteLater()
            self._placeholder = None

    @Slot(object)
    def _on_transcript_segment(self, segment):
        """Handle new transcript segment from transcriber"""
        self._hide_placeholder()

        segment_id = segment.id
        text = segment.text
        start_time = segment.start
        end_time = segment.end
        is_partial = getattr(segment, 'is_partial', False)

        # Store the full segment for word-level access
        self._segments[segment_id] = segment

        # Create new bubble with streaming support, including segment reference
        bubble = TranscriptBubble(text, start_time, end_time, is_partial=is_partial, segment=segment)
        self._bubbles[segment_id] = bubble

        # Connect play signal
        bubble.play_requested.connect(self._play_audio_segment)

        # Insert before the stretch
        count = self.messages_layout.count()
        self.messages_layout.insertWidget(count - 1, bubble)

        # Update status
        self.status_label.setText(f"{len(self._bubbles)} segments")

        # Auto-scroll to bottom
        if self._auto_scroll:
            QTimer.singleShot(10, self._scroll_to_bottom)

    @Slot(object)
    def _on_transcript_segment_updated(self, segment):
        """Handle update to existing transcript segment (streaming mode)"""
        segment_id = segment.id

        # Update stored segment
        self._segments[segment_id] = segment

        if segment_id in self._bubbles:
            # Update existing bubble
            bubble = self._bubbles[segment_id]
            is_partial = getattr(segment, 'is_partial', False)
            bubble.update_segment(
                text=segment.text,
                end_time=segment.end,
                is_partial=is_partial
            )
            # Also update the segment reference
            bubble.set_segment(segment)

            # Auto-scroll to bottom during streaming
            if self._auto_scroll:
                QTimer.singleShot(10, self._scroll_to_bottom)

    @Slot(object)
    def _on_transcript_segment_corrected(self, correction_result):
        """Handle LLM correction for a transcript segment"""
        segment_id = correction_result.segment_id

        # Handle merge with previous segment
        if correction_result.merge_with_previous and correction_result.merged_text:
            self._merge_with_previous_bubble(segment_id, correction_result)
            return

        if segment_id in self._bubbles:
            bubble = self._bubbles[segment_id]

            # Only apply if actually corrected
            if correction_result.is_corrected:
                bubble.apply_correction(correction_result.corrected_text)

                # Auto-scroll to show the correction
                if self._auto_scroll:
                    QTimer.singleShot(10, self._scroll_to_bottom)

    def _merge_with_previous_bubble(self, segment_id: int, correction_result):
        """Merge current segment's bubble with the previous one"""
        if segment_id not in self._bubbles:
            return

        current_bubble = self._bubbles[segment_id]

        # Find the previous bubble (segment_id - 1)
        prev_id = segment_id - 1
        if prev_id not in self._bubbles:
            # No previous bubble to merge with, just apply correction
            if correction_result.is_corrected:
                current_bubble.apply_correction(correction_result.corrected_text)
            return

        prev_bubble = self._bubbles[prev_id]

        # Update previous bubble with merged text
        prev_bubble.apply_correction(correction_result.merged_text)

        # Update previous bubble's end time
        current_start, current_end = current_bubble.get_time_range()
        prev_start, prev_end = prev_bubble.get_time_range()
        prev_bubble._end_time = current_end
        prev_bubble.time_label.setText(prev_bubble._format_timestamp())

        # Remove current bubble
        self.messages_layout.removeWidget(current_bubble)
        current_bubble.deleteLater()
        del self._bubbles[segment_id]

        # Update status
        self.status_label.setText(f"{len(self._bubbles)} segments")

        logger.info(f"Merged segment {segment_id} into {prev_id}")

        # Auto-scroll
        if self._auto_scroll:
            QTimer.singleShot(10, self._scroll_to_bottom)

    def clear_transcript(self):
        """Clear all transcript bubbles and conversation containers"""
        for bubble in self._bubbles.values():
            bubble.deleteLater()
        self._bubbles.clear()

        for container in self._conversation_containers.values():
            container.deleteLater()
        self._conversation_containers.clear()

        self._segments.clear()
        self.status_label.setText("")
        self._show_placeholder()

    def _toggle_auto_scroll(self, checked: bool):
        """Toggle auto-scroll behavior"""
        self._auto_scroll = checked
        if checked:
            self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        """Scroll to the bottom of the message list"""
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @Slot()
    def _on_recording_started(self):
        """Handle recording start"""
        self._is_recording = True
        self.clear_transcript()
        self.status_label.setText("Recording...")

    @Slot()
    def _on_recording_stopped(self):
        """Handle recording stop"""
        self._is_recording = False
        if self._bubbles:
            self.status_label.setText(f"{len(self._bubbles)} segments")
        else:
            self.status_label.setText("")

    @Slot()
    def _on_model_loaded(self):
        """Handle transcription model loaded"""
        # Could show a ready indicator
        pass

    @Slot(str)
    def _on_recording_saved(self, audio_path: str):
        """Handle recording saved - store audio path for playback"""
        self._audio_file_path = Path(audio_path)
        logger.debug(f"Audio file path set: {self._audio_file_path}")

    @Slot(object)
    def _on_speaker_identified(self, result):
        """Handle speaker identification result (supports multi-speaker segments)"""
        segment_id = result.segment_id

        if result.has_multiple_speakers and segment_id in self._bubbles:
            # Multi-speaker: replace bubble with conversation container
            self._create_conversation_container(segment_id, result)
        elif segment_id in self._bubbles:
            # Single speaker: just update the bubble label
            bubble = self._bubbles[segment_id]
            bubble.set_speakers(
                primary_speaker=result.primary_speaker,
                has_multiple=False,
                sequence=""
            )
            logger.debug(f"Set speaker for segment {segment_id}: {result.primary_speaker}")

    def _create_conversation_container(self, segment_id: int, result):
        """Replace a bubble with a conversation container for multi-speaker segments"""
        if segment_id not in self._bubbles:
            return

        bubble = self._bubbles[segment_id]
        segment = self._segments.get(segment_id)

        if segment is None:
            segment = bubble.get_segment()

        if segment is None:
            # Fallback: just update the label
            logger.warning(f"No segment data for {segment_id}, falling back to label")
            bubble.set_speakers(
                primary_speaker=result.primary_speaker,
                has_multiple=result.has_multiple_speakers,
                sequence=result.get_speaker_sequence()
            )
            return

        # Split the segment text by speaker time ranges
        parts = split_segment_by_speakers(segment, result.speakers)

        if len(parts) <= 1:
            # No actual split happened, just update the label
            bubble.set_speakers(
                primary_speaker=result.primary_speaker,
                has_multiple=result.has_multiple_speakers,
                sequence=result.get_speaker_sequence()
            )
            return

        # Create conversation container
        container = ConversationContainer(segment_id, segment.start, segment.end)
        container.play_requested.connect(self._play_audio_segment)

        # Add speaker parts
        for part in parts:
            container.add_speaker_part(
                speaker=part["speaker"],
                text=part["text"],
                start_time=part["start"],
                end_time=part["end"]
            )

        # Find the bubble's position in the layout
        bubble_index = self.messages_layout.indexOf(bubble)

        # Remove the old bubble and insert the container
        self.messages_layout.removeWidget(bubble)
        bubble.deleteLater()
        del self._bubbles[segment_id]

        # Insert container at the same position
        self.messages_layout.insertWidget(bubble_index, container)
        self._conversation_containers[segment_id] = container

        logger.info(f"Created conversation container for segment {segment_id}: {result.get_speaker_sequence()} ({len(parts)} parts)")

        # Auto-scroll to show the new container
        if self._auto_scroll:
            QTimer.singleShot(10, self._scroll_to_bottom)

    def set_audio_file(self, path: Path):
        """Set the audio file path for playback"""
        self._audio_file_path = path

    @Slot(float, float)
    def _play_audio_segment(self, start_time: float, end_time: float):
        """Play a segment of the audio file"""
        if not AUDIO_PLAYBACK_AVAILABLE:
            logger.warning("Audio playback not available")
            self.signals.status_message.emit("Audio playback not available", 2000)
            return

        if not self._audio_file_path or not self._audio_file_path.exists():
            logger.warning(f"Audio file not found: {self._audio_file_path}")
            self.signals.status_message.emit("Audio file not found", 2000)
            return

        # Stop any current playback
        self._stop_current_playback()

        # Find the bubble that requested playback
        requesting_bubble = None
        for bubble in self._bubbles.values():
            bubble_start, bubble_end = bubble.get_time_range()
            if abs(bubble_start - start_time) < 0.1:
                requesting_bubble = bubble
                break

        if requesting_bubble:
            requesting_bubble.set_playing(True)
            self._currently_playing_bubble = requesting_bubble

        # Start playback in a separate thread
        self._playback_thread = threading.Thread(
            target=self._playback_worker,
            args=(start_time, end_time),
            daemon=True
        )
        self._playback_thread.start()

    def _playback_worker(self, start_time: float, end_time: float):
        """Worker thread for audio playback"""
        try:
            # Read the WAV file
            with wave.open(str(self._audio_file_path), 'rb') as wf:
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()

                # Calculate frame positions
                start_frame = int(start_time * sample_rate)
                end_frame = int(end_time * sample_rate)

                # Seek to start position
                wf.setpos(start_frame)

                # Read the segment
                n_frames = end_frame - start_frame
                audio_data = wf.readframes(n_frames)

                # Convert to numpy array
                if sample_width == 2:
                    dtype = np.int16
                elif sample_width == 4:
                    dtype = np.int32
                else:
                    dtype = np.uint8

                audio_array = np.frombuffer(audio_data, dtype=dtype)

                # Reshape for stereo
                if n_channels > 1:
                    audio_array = audio_array.reshape(-1, n_channels)

                # Play the audio (blocking)
                sd.play(audio_array, sample_rate)
                sd.wait()

        except Exception as e:
            logger.error(f"Error playing audio segment: {e}")

        finally:
            # Reset the play button state (must use QTimer for thread safety)
            QTimer.singleShot(0, self._on_playback_finished)

    def _on_playback_finished(self):
        """Called when playback finishes"""
        if self._currently_playing_bubble:
            self._currently_playing_bubble.set_playing(False)
            self._currently_playing_bubble = None

    def _stop_current_playback(self):
        """Stop any current audio playback"""
        if AUDIO_PLAYBACK_AVAILABLE:
            sd.stop()
        if self._currently_playing_bubble:
            self._currently_playing_bubble.set_playing(False)
            self._currently_playing_bubble = None
