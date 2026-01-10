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

from ...signals import get_app_signals
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
        self._audio_devices = []

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
        """Populate audio device dropdown with actual system microphones"""
        self.device_combo.clear()
        self._audio_devices = []

        try:
            # Try to use Qt Multimedia to get audio devices
            from PySide6.QtMultimedia import QMediaDevices, QAudioDevice

            audio_inputs = QMediaDevices.audioInputs()

            if audio_inputs:
                for device in audio_inputs:
                    device_name = device.description()
                    self._audio_devices.append(device)
                    # Mark default device
                    if device == QMediaDevices.defaultAudioInput():
                        self.device_combo.addItem(f"* {device_name}")
                    else:
                        self.device_combo.addItem(device_name)
            else:
                self.device_combo.addItem("No microphones found")

        except ImportError:
            # Fallback if QtMultimedia is not available
            self._populate_devices_fallback()

    def _populate_devices_fallback(self):
        """Fallback method to detect audio devices using pyaudio or system APIs"""
        try:
            import pyaudio
            p = pyaudio.PyAudio()

            default_device = p.get_default_input_device_info()
            default_index = default_device.get('index', -1)

            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                # Only show input devices
                if device_info.get('maxInputChannels', 0) > 0:
                    name = device_info.get('name', f'Device {i}')
                    if i == default_index:
                        self.device_combo.addItem(f"* {name}")
                    else:
                        self.device_combo.addItem(name)
                    self._audio_devices.append(device_info)

            p.terminate()

            if self.device_combo.count() == 0:
                self.device_combo.addItem("No microphones found")

        except Exception as e:
            # Ultimate fallback
            self.device_combo.addItem("System Default Microphone")
            self._audio_devices.append(None)

    def _on_device_changed(self, index: int):
        """Handle audio device selection change"""
        if 0 <= index < len(self._audio_devices):
            device = self._audio_devices[index]
            if device:
                try:
                    device_name = device.description() if hasattr(device, 'description') else str(device.get('name', 'Unknown'))
                    self.signals.audio_device_changed.emit(device_name)
                except:
                    pass

    @Slot()
    def _on_record_clicked(self):
        """Handle record button click"""
        if not self._is_recording:
            self.signals.recording_started.emit()
        else:
            self.signals.recording_stopped.emit()

    @Slot()
    def _on_pause_clicked(self):
        """Handle pause button click"""
        if self._is_paused:
            self.signals.recording_resumed.emit()
        else:
            self.signals.recording_paused.emit()

    @Slot()
    def _on_stop_clicked(self):
        """Handle stop button click"""
        self.signals.recording_stopped.emit()

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

        self.waveform.set_active(False)
        self._timer.stop()

        self.signals.status_message.emit("Recording stopped", 2000)

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

    def get_selected_device(self):
        """Get the currently selected audio device"""
        index = self.device_combo.currentIndex()
        if 0 <= index < len(self._audio_devices):
            return self._audio_devices[index]
        return None

    def refresh_devices(self):
        """Refresh the list of audio devices"""
        self._populate_devices()
