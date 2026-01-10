"""Recording controller - coordinates UI, audio recorder, and session management"""

from pathlib import Path
from typing import Optional
from loguru import logger
from PySide6.QtCore import QObject, Slot

from .session import Session, get_session_manager
from .audio_recorder import get_audio_recorder, RecordingState
from .config import get_config_manager
from ..signals import get_app_signals


class RecordingController(QObject):
    """
    Controller that manages the recording workflow.
    Coordinates between UI signals, audio recorder, and session management.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._signals = get_app_signals()
        self._session_manager = get_session_manager()
        self._audio_recorder = get_audio_recorder()
        self._config = get_config_manager()

        self._current_session: Optional[Session] = None

        # Connect audio level callback
        self._audio_recorder.set_level_callback(self._on_audio_level)

        # Connect signals
        self._connect_signals()

    def _connect_signals(self):
        """Connect to UI signals"""
        # These will be emitted by the recording bar when user clicks buttons
        # We handle them here to do the actual recording
        pass  # Signals are connected in main_window

    def _on_audio_level(self, level: float):
        """Handle audio level updates from recorder"""
        # Emit signal for waveform visualization
        self._signals.audio_level_updated.emit(level)

    def is_configured(self) -> bool:
        """Check if the app is configured (has a save directory)"""
        return self._config.get_default_save_directory() is not None

    def is_first_run(self) -> bool:
        """Check if this is the first run"""
        return self._config.is_first_run()

    def start_recording(self, session_name: Optional[str] = None, save_directory: Optional[Path] = None) -> bool:
        """
        Start a new recording session.

        Args:
            session_name: Name for the session (auto-generated if None)
            save_directory: Directory to save to (uses default if None)

        Returns:
            True if recording started successfully
        """
        try:
            # Create new session
            self._current_session = self._session_manager.create_session(
                name=session_name,
                directory=save_directory,
            )

            # Get audio output path
            audio_path = self._current_session.get_audio_path()

            # Start recording
            self._audio_recorder.start_recording(audio_path, convert_to_mp3=True)

            # Update session status
            self._current_session.set_status("recording")

            logger.info(f"Recording started: {self._current_session.metadata.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self._signals.status_message.emit(f"Failed to start: {str(e)}", 5000)
            return False

    def pause_recording(self):
        """Pause the current recording"""
        if self._audio_recorder.is_recording:
            self._audio_recorder.pause_recording()
            if self._current_session:
                self._current_session.set_status("paused")
            logger.info("Recording paused")

    def resume_recording(self):
        """Resume the current recording"""
        if self._audio_recorder.is_paused:
            self._audio_recorder.resume_recording()
            if self._current_session:
                self._current_session.set_status("recording")
            logger.info("Recording resumed")

    def stop_recording(self) -> Optional[Path]:
        """
        Stop the current recording and finalize.

        Returns:
            Path to the saved audio file, or None if failed
        """
        if self._audio_recorder.state == RecordingState.IDLE:
            logger.warning("No recording in progress")
            return None

        try:
            # Stop recording
            audio_path = self._audio_recorder.stop_recording()

            if self._current_session:
                # Update session metadata
                self._current_session.metadata.audio_file = str(audio_path) if audio_path else None
                self._current_session.set_status("completed")

                session_dir = self._current_session.directory
                logger.info(f"Recording saved to: {session_dir}")

                # Emit signal with session path
                self._signals.status_message.emit(
                    f"Recording saved to {session_dir.name}", 3000
                )

            self._current_session = None
            return audio_path

        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            self._signals.status_message.emit(f"Error stopping: {str(e)}", 5000)
            return None

    def update_duration(self, seconds: int):
        """Update the current session duration"""
        if self._current_session:
            self._current_session.update_duration(seconds)

    def get_current_session(self) -> Optional[Session]:
        """Get the current session"""
        return self._current_session

    def set_audio_device(self, device_index: int):
        """Set the audio input device"""
        self._audio_recorder.set_device(device_index)

    def get_recording_state(self) -> RecordingState:
        """Get the current recording state"""
        return self._audio_recorder.state


# Singleton
_controller_instance: Optional[RecordingController] = None


def get_recording_controller() -> RecordingController:
    """Get the singleton recording controller"""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = RecordingController()
    return _controller_instance
