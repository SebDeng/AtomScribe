"""Recording controller - coordinates UI, audio recorder, and session management"""

from pathlib import Path
from typing import Optional
from loguru import logger
from PySide6.QtCore import QObject, Slot

from .session import Session, get_session_manager
from .audio_recorder import get_audio_recorder, RecordingState
from .config import get_config_manager
from .transcriber import get_transcriber, TranscriptSegment
from .llm_processor import get_llm_processor, CorrectionResult
from ..signals import get_app_signals


class RecordingController(QObject):
    """
    Controller that manages the recording workflow.
    Coordinates between UI signals, audio recorder, session management, and transcription.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._signals = get_app_signals()
        self._session_manager = get_session_manager()
        self._audio_recorder = get_audio_recorder()
        self._config = get_config_manager()

        # Get transcriber with config settings
        config = self._config.config

        # Parse hotwords from comma-separated string
        hotwords = None
        if config.transcription_hotwords:
            hotwords = [w.strip() for w in config.transcription_hotwords.split(",")]

        self._transcriber = get_transcriber(
            language=config.transcription_language,
            initial_prompt=config.transcription_initial_prompt,
            hotwords=hotwords,
            replacements=config.transcription_replacements,
        )

        # Get LLM processor for post-processing
        self._llm_processor = get_llm_processor()
        self._llm_enabled = True  # Can be toggled

        self._current_session: Optional[Session] = None
        self._transcription_enabled = True

        # Connect audio level callback
        self._audio_recorder.set_level_callback(self._on_audio_level)

        # Connect audio data callback for transcription
        self._audio_recorder.set_audio_data_callback(self._on_audio_data)

        # Connect transcriber callbacks
        self._transcriber.set_on_segment_callback(self._on_transcript_segment)
        self._transcriber.set_on_segment_update_callback(self._on_transcript_segment_updated)
        self._transcriber.set_on_model_loaded_callback(self._on_model_loaded)

        # Connect LLM processor callbacks
        self._llm_processor.set_on_correction_callback(self._on_transcript_corrected)
        self._llm_processor.set_on_model_loaded_callback(self._on_llm_model_loaded)

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

    def _on_audio_data(self, audio_data):
        """Handle raw audio data - send to transcriber"""
        if self._transcription_enabled and self._transcriber.is_running():
            # Send audio to transcriber (it expects 16kHz, recorder uses 44.1kHz)
            self._transcriber.feed_audio(audio_data, source_sample_rate=44100)

    def _on_transcript_segment(self, segment: TranscriptSegment):
        """Handle new transcript segment from transcriber"""
        # Emit signal for UI update
        self._signals.transcript_segment.emit(segment)
        logger.debug(f"Transcript: [{segment.start:.1f}s] {segment.text}")

        # Queue for LLM correction if enabled and not partial
        is_partial = getattr(segment, 'is_partial', False)
        logger.debug(f"LLM check: enabled={self._llm_enabled}, running={self._llm_processor.is_running()}, model_loaded={self._llm_processor.is_model_loaded()}, is_partial={is_partial}")
        if self._llm_enabled and self._llm_processor.is_running() and not is_partial:
            logger.info(f"Queueing segment {segment.id} for LLM correction")
            self._llm_processor.queue_segment(segment.id, segment.text)
        elif not is_partial and self._llm_enabled:
            if not self._llm_processor.is_model_loaded():
                logger.warning("LLM model not loaded, skipping correction")
            elif not self._llm_processor.is_running():
                logger.warning("LLM processor not running, skipping correction")

    def _on_transcript_segment_updated(self, segment: TranscriptSegment):
        """Handle transcript segment update (streaming mode)"""
        # Emit signal for UI update
        self._signals.transcript_segment_updated.emit(segment)

        # Queue for LLM correction when segment is finalized (is_partial=False)
        is_partial = getattr(segment, 'is_partial', False)
        if not is_partial and self._llm_enabled and self._llm_processor.is_running():
            logger.info(f"Segment {segment.id} finalized, queueing for LLM correction")
            self._llm_processor.queue_segment(segment.id, segment.text)

    def _on_model_loaded(self):
        """Handle transcription model loaded"""
        self._signals.transcription_model_loaded.emit()
        self._signals.status_message.emit("Transcription model loaded", 2000)
        logger.info("Transcription model loaded and ready")

    def _on_transcript_corrected(self, result: CorrectionResult):
        """Handle LLM correction result"""
        # Emit signal for UI update
        self._signals.transcript_segment_corrected.emit(result)
        if result.is_corrected:
            logger.debug(f"Segment {result.segment_id} corrected by LLM")

    def _on_llm_model_loaded(self):
        """Handle LLM model loaded"""
        self._signals.llm_model_loaded.emit()
        self._signals.status_message.emit("LLM model loaded", 2000)
        logger.info("LLM model loaded and ready")

    def is_configured(self) -> bool:
        """Check if the app is configured (has a save directory)"""
        return self._config.get_default_save_directory() is not None

    def is_first_run(self) -> bool:
        """Check if this is the first run"""
        return self._config.is_first_run()

    def preload_transcription_model(self):
        """Preload the transcription model in background"""
        if not self._transcriber.is_model_loaded():
            logger.info("Preloading transcription model...")
            self._signals.status_message.emit("Loading transcription model...", 0)
            self._transcriber.load_model(blocking=False)

    def preload_llm_model(self):
        """Preload the LLM model in background"""
        if not self._llm_processor.is_model_loaded():
            logger.info("Preloading LLM model...")
            self._signals.status_message.emit("Loading LLM model...", 0)
            self._llm_processor.load_model(blocking=False)

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

            # Start transcription
            if self._transcription_enabled:
                transcript_path = self._current_session.get_transcript_path()
                self._transcriber.start(output_path=transcript_path)
                self._signals.transcription_started.emit()

            # Start LLM post-processor if enabled and model is loaded
            logger.info(f"LLM status: enabled={self._llm_enabled}, model_loaded={self._llm_processor.is_model_loaded()}")
            if self._llm_enabled and self._llm_processor.is_model_loaded():
                self._llm_processor.start()
                self._signals.llm_processing_started.emit()
                logger.info("LLM post-processor started")
            else:
                logger.warning("LLM post-processor NOT started (model not loaded or disabled)")

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
            # Stop transcription first
            if self._transcriber.is_running():
                segments = self._transcriber.stop()
                self._signals.transcription_stopped.emit()
                logger.info(f"Transcription stopped with {len(segments)} segments")

            # Stop LLM post-processor
            if self._llm_processor.is_running():
                self._llm_processor.stop()
                self._signals.llm_processing_stopped.emit()
                logger.info("LLM post-processor stopped")

            # Stop recording
            audio_path = self._audio_recorder.stop_recording()

            if self._current_session:
                # Update session metadata
                self._current_session.metadata.audio_file = str(audio_path) if audio_path else None
                self._current_session.metadata.transcript_file = str(self._current_session.get_transcript_path())
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

    def set_transcription_enabled(self, enabled: bool):
        """Enable or disable transcription"""
        self._transcription_enabled = enabled
        logger.info(f"Transcription {'enabled' if enabled else 'disabled'}")

    def is_transcription_enabled(self) -> bool:
        """Check if transcription is enabled"""
        return self._transcription_enabled

    def is_transcription_model_loaded(self) -> bool:
        """Check if transcription model is loaded"""
        return self._transcriber.is_model_loaded()

    def set_llm_enabled(self, enabled: bool):
        """Enable or disable LLM post-processing"""
        self._llm_enabled = enabled
        logger.info(f"LLM post-processing {'enabled' if enabled else 'disabled'}")

    def is_llm_enabled(self) -> bool:
        """Check if LLM post-processing is enabled"""
        return self._llm_enabled

    def is_llm_model_loaded(self) -> bool:
        """Check if LLM model is loaded"""
        return self._llm_processor.is_model_loaded()


# Singleton
_controller_instance: Optional[RecordingController] = None


def get_recording_controller() -> RecordingController:
    """Get the singleton recording controller"""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = RecordingController()
    return _controller_instance
