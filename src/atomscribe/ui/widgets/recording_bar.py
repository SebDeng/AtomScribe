"""Bottom recording control bar"""

from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QSpacerItem,
    QSizePolicy,
)
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QFont
from loguru import logger

from ...signals import get_app_signals
from ...core.audio_recorder import get_audio_recorder, AudioDevice
from .waveform import WaveformWidget


class RecordingBar(QWidget):
    """
    Recording control bar at the top of the window.
    Contains record/pause/stop buttons, waveform, timer, and audio device selector.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = get_app_signals()
        self.setObjectName("recordingBar")
        self.setFixedHeight(56)

        self._is_recording = False
        self._is_paused = False
        self._elapsed_seconds = 0
        self._audio_devices: list[AudioDevice] = []

        self._setup_ui()
        self._connect_signals()
        self._setup_timer()

    def _setup_ui(self):
        """Set up the recording bar UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(10)

        # Record button - compact size with explicit bounds
        self.record_btn = QPushButton("REC")
        self.record_btn.setObjectName("recordButton")
        self.record_btn.setCheckable(True)
        self.record_btn.setToolTip("Start Recording")
        self.record_btn.clicked.connect(self._on_record_clicked)
        self.record_btn.setFixedSize(52, 28)
        self.record_btn.setStyleSheet("""
            QPushButton#recordButton {
                background-color: #FEE2E2;
                border: 1px solid #F87171;
                border-radius: 4px;
                color: #DC2626;
                font-size: 11px;
                font-weight: bold;
                margin: 0px;
                padding: 0px;
            }
            QPushButton#recordButton:hover {
                background-color: #FECACA;
                border-color: #EF4444;
            }
            QPushButton#recordButton:checked {
                background-color: #DC2626;
                border-color: #B91C1C;
                color: white;
            }
        """)
        layout.addWidget(self.record_btn)

        # Pause button - use text
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setObjectName("pauseButton")
        self.pause_btn.setEnabled(False)
        self.pause_btn.setToolTip("Pause Recording")
        self.pause_btn.clicked.connect(self._on_pause_clicked)
        self.pause_btn.setFixedHeight(32)
        self.pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #FFF8E1;
                border: 1px solid #FFB300;
                border-radius: 6px;
                color: #FF8F00;
                font-size: 12px;
                font-weight: 500;
                padding: 0 12px;
            }
            QPushButton:hover {
                background-color: #FFECB3;
            }
            QPushButton:disabled {
                background-color: #F5F5F5;
                border-color: #E0E0E0;
                color: #BDBDBD;
            }
        """)
        layout.addWidget(self.pause_btn)

        # Stop button - use text
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip("Stop Recording")
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.stop_btn.setFixedHeight(32)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #F5F5F5;
                border: 1px solid #BDBDBD;
                border-radius: 6px;
                color: #616161;
                font-size: 12px;
                font-weight: 500;
                padding: 0 12px;
            }
            QPushButton:hover {
                background-color: #EEEEEE;
                border-color: #9E9E9E;
            }
            QPushButton:disabled {
                background-color: #F5F5F5;
                border-color: #E0E0E0;
                color: #BDBDBD;
            }
        """)
        layout.addWidget(self.stop_btn)

        # Separator
        layout.addSpacing(12)

        # Waveform visualizer
        self.waveform = WaveformWidget()
        self.waveform.setMinimumWidth(200)
        self.waveform.setFixedHeight(32)
        layout.addWidget(self.waveform, stretch=1)

        # Separator
        layout.addSpacing(12)

        # Timer display
        self.timer_label = QLabel("00:00:00")
        self.timer_label.setObjectName("timerLabel")
        timer_font = QFont("Consolas", 13, QFont.DemiBold)
        self.timer_label.setFont(timer_font)
        self.timer_label.setMinimumWidth(80)
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("""
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
            border-radius: 6px;
            padding: 4px 10px;
            color: #37352F;
        """)
        layout.addWidget(self.timer_label)

        # Separator
        layout.addSpacing(12)

        # Microphone label
        mic_label = QLabel("Mic:")
        mic_label.setStyleSheet("color: #787774; font-size: 12px;")
        layout.addWidget(mic_label)

        # Audio device selector
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(180)
        self.device_combo.setFixedHeight(32)
        self.device_combo.setToolTip("Select Audio Input Device")
        self.device_combo.setStyleSheet("""
            QComboBox {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 4px 10px;
                color: #37352F;
                font-size: 12px;
            }
            QComboBox:hover {
                border-color: #BDBDBD;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
        """)
        self._populate_devices()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        layout.addWidget(self.device_combo)

    def _connect_signals(self):
        """Connect to app signals"""
        self.signals.recording_started.connect(self._on_recording_started)
        self.signals.recording_stopped.connect(self._on_recording_stopped)
        self.signals.recording_paused.connect(self._on_recording_paused)
        self.signals.recording_resumed.connect(self._on_recording_resumed)
        self.signals.audio_level_updated.connect(self.waveform.set_level)

    def _setup_timer(self):
        """Set up the recording timer"""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_timer)

    def _populate_devices(self):
        """Populate audio device dropdown using sounddevice"""
        self.device_combo.clear()
        self._audio_devices = []

        try:
            # Use the audio recorder's device detection
            devices = get_audio_recorder().get_audio_devices()

            if devices:
                for device in devices:
                    self._audio_devices.append(device)
                    # Mark default device with asterisk
                    if device.is_default:
                        self.device_combo.addItem(f"* {device.name}")
                        # Select default device
                        self.device_combo.setCurrentIndex(self.device_combo.count() - 1)
                    else:
                        self.device_combo.addItem(device.name)
            else:
                self.device_combo.addItem("No microphones found")
                logger.warning("No audio input devices found")

        except Exception as e:
            logger.error(f"Error populating audio devices: {e}")
            self.device_combo.addItem("Error loading devices")

    def _on_device_changed(self, index: int):
        """Handle audio device selection change"""
        if 0 <= index < len(self._audio_devices):
            device = self._audio_devices[index]
            # Set the device on the audio recorder
            get_audio_recorder().set_device(device.index)
            logger.info(f"Selected audio device: {device.name} (index {device.index})")
            self.signals.audio_device_changed.emit(device.name)

    @Slot()
    def _on_record_clicked(self):
        """Handle record button click"""
        if not self._is_recording:
            # Emit button click signal - main window will show dialog
            # Don't update UI yet - wait for recording_started signal
            self.record_btn.setChecked(False)  # Reset button until confirmed
            self.signals.record_button_clicked.emit()
        else:
            # Clicking REC while recording = stop
            self.signals.stop_button_clicked.emit()

    @Slot()
    def _on_pause_clicked(self):
        """Handle pause button click"""
        # Emit button click - actual state change happens via signals
        self.signals.pause_button_clicked.emit()

    @Slot()
    def _on_stop_clicked(self):
        """Handle stop button click"""
        self.signals.stop_button_clicked.emit()

    @Slot()
    def _on_recording_started(self):
        """Update UI when recording starts"""
        self._is_recording = True
        self._is_paused = False
        self._elapsed_seconds = 0

        self.record_btn.setChecked(True)
        self.record_btn.setToolTip("Recording...")
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("Pause")
        self.stop_btn.setEnabled(True)

        # Disable device selection during recording
        self.device_combo.setEnabled(False)

        self.waveform.set_active(True)
        self._timer.start(1000)

        self.signals.status_message.emit("Recording started", 2000)

    @Slot()
    def _on_recording_stopped(self):
        """Update UI when recording stops"""
        self._is_recording = False
        self._is_paused = False

        self.record_btn.setChecked(False)
        self.record_btn.setToolTip("Start Recording")
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("Pause")
        self.stop_btn.setEnabled(False)

        # Re-enable device selection
        self.device_combo.setEnabled(True)

        self.waveform.set_active(False)
        self._timer.stop()

        # Don't reset timer - keep showing final duration

    @Slot()
    def _on_recording_paused(self):
        """Update UI when recording is paused"""
        self._is_paused = True
        self.pause_btn.setText("Resume")
        self.pause_btn.setToolTip("Resume Recording")
        self.waveform.set_active(False)
        self._timer.stop()

        self.signals.status_message.emit("Recording paused", 2000)

    @Slot()
    def _on_recording_resumed(self):
        """Update UI when recording is resumed"""
        self._is_paused = False
        self.pause_btn.setText("Pause")
        self.pause_btn.setToolTip("Pause Recording")
        self.waveform.set_active(True)
        self._timer.start(1000)

        self.signals.status_message.emit("Recording resumed", 2000)

    def _update_timer(self):
        """Update the timer display"""
        self._elapsed_seconds += 1
        hours = self._elapsed_seconds // 3600
        minutes = (self._elapsed_seconds % 3600) // 60
        seconds = self._elapsed_seconds % 60
        self.timer_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        self.signals.recording_time_updated.emit(self._elapsed_seconds)

    def get_elapsed_time(self) -> int:
        """Get elapsed recording time in seconds"""
        return self._elapsed_seconds

    def reset_timer(self):
        """Reset the timer to zero"""
        self._elapsed_seconds = 0
        self.timer_label.setText("00:00:00")

    def get_selected_device(self) -> AudioDevice | None:
        """Get the currently selected audio device"""
        index = self.device_combo.currentIndex()
        if 0 <= index < len(self._audio_devices):
            return self._audio_devices[index]
        return None

    def get_selected_device_index(self) -> int:
        """Get the index of the currently selected device"""
        index = self.device_combo.currentIndex()
        if 0 <= index < len(self._audio_devices):
            return self._audio_devices[index].index
        return -1

    def refresh_devices(self):
        """Refresh the list of audio devices"""
        self._populate_devices()
