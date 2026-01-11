"""Recording controller - coordinates UI, audio recorder, and session management"""

from pathlib import Path
from typing import Optional
from loguru import logger
from PySide6.QtCore import QObject, Slot

from .session import Session, get_session_manager
from .audio_recorder import get_audio_recorder, RecordingState
from .screen_recorder import get_screen_recorder, ScreenRecordingState
from .config import get_config_manager
from .transcriber import get_transcriber, TranscriptSegment
from .llm_processor import get_llm_processor, CorrectionResult
from .speaker_diarizer import get_speaker_diarizer, SpeakerResult, SPEECHBRAIN_AVAILABLE
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

        # Get config settings
        config = self._config.config

        # Get screen recorder with config settings
        self._screen_recorder = get_screen_recorder(
            fps=config.screen_recording_fps,
            monitor_index=config.screen_recording_monitor,
            quality=config.screen_recording_quality,
            codec=config.screen_recording_codec,
        )
        self._screen_recording_enabled = config.screen_recording_enabled

        # Get transcriber with config settings

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

        # Get speaker diarizer for speaker identification
        self._diarizer = get_speaker_diarizer(
            num_speakers=config.diarization_num_speakers,
            min_speakers=config.diarization_min_speakers,
            max_speakers=config.diarization_max_speakers,
        ) if SPEECHBRAIN_AVAILABLE else None
        self._diarization_enabled = config.diarization_enabled

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

        # Connect speaker diarizer callbacks
        if self._diarizer:
            self._diarizer.set_on_speaker_callback(self._on_speaker_identified)
            self._diarizer.set_on_model_loaded_callback(self._on_diarization_model_loaded)

        # Connect screen recorder callbacks
        self._screen_recorder.set_on_error_callback(self._on_screen_recording_error)

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

        # Queue for speaker diarization when segment is finalized
        if not is_partial and self._diarization_enabled and self._diarizer and self._diarizer.is_running():
            # Get audio segment from recorder buffer
            audio_segment = self._audio_recorder.get_audio_segment(segment.start, segment.end)
            if audio_segment is not None and len(audio_segment) > 0:
                logger.info(f"Segment {segment.id} finalized, queueing for diarization")
                self._diarizer.queue_segment(segment.id, audio_segment, segment.start, segment.end)
            else:
                logger.warning(f"Could not get audio for segment {segment.id} diarization")

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

    def _on_speaker_identified(self, result: SpeakerResult):
        """Handle speaker identification result"""
        # Emit signal for UI update (now sends full SpeakerResult for multi-speaker support)
        self._signals.speaker_identified.emit(result)
        if result.has_multiple_speakers:
            logger.info(f"Segment {result.segment_id}: {result.get_speaker_sequence()}")
        else:
            logger.debug(f"Segment {result.segment_id} assigned to {result.primary_speaker}")

    def _on_diarization_model_loaded(self):
        """Handle diarization model loaded"""
        self._signals.diarization_model_loaded.emit()
        self._signals.status_message.emit("Speaker diarization model loaded", 2000)
        logger.info("Speaker diarization model loaded and ready")

    def _on_screen_recording_error(self, error: str):
        """Handle screen recording error"""
        self._signals.screen_recording_error.emit(error)
        self._signals.status_message.emit(f"Screen recording error: {error}", 5000)
        logger.error(f"Screen recording error: {error}")

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

    def preload_diarization_model(self):
        """Preload the speaker diarization model in background"""
        if self._diarizer and not self._diarizer.is_model_loaded():
            logger.info("Preloading speaker diarization model...")
            self._signals.status_message.emit("Loading speaker diarization model...", 0)
            self._diarizer.load_model(blocking=False)

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

            # Start speaker diarizer if enabled and model is loaded
            logger.info(f"Diarization status: enabled={self._diarization_enabled}, diarizer={self._diarizer is not None}, model_loaded={self._diarizer.is_model_loaded() if self._diarizer else False}")
            if self._diarization_enabled and self._diarizer and self._diarizer.is_model_loaded():
                self._diarizer.start()
                self._signals.diarization_started.emit()
                logger.info("Speaker diarizer started")
            else:
                logger.warning("Speaker diarizer NOT started (model not loaded or disabled)")

            # Start screen recording if enabled and available
            logger.info(f"Screen recording status: enabled={self._screen_recording_enabled}, available={self._screen_recorder.is_available()}")
            if self._screen_recording_enabled and self._screen_recorder.is_available():
                try:
                    video_path = self._current_session.get_video_path()
                    self._screen_recorder.start_recording(video_path)
                    self._signals.screen_recording_started.emit()
                    logger.info(f"Screen recording started: {video_path}")
                except Exception as e:
                    logger.error(f"Failed to start screen recording: {e}")
                    self._signals.status_message.emit(f"Screen recording failed: {str(e)}", 5000)
            else:
                if not self._screen_recording_enabled:
                    logger.info("Screen recording disabled in config")
                else:
                    logger.warning("Screen recording NOT started (dependencies not available)")

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

            # Pause screen recording
            if self._screen_recorder.is_recording:
                self._screen_recorder.pause_recording()
                self._signals.screen_recording_paused.emit()

            if self._current_session:
                self._current_session.set_status("paused")
            logger.info("Recording paused")

    def resume_recording(self):
        """Resume the current recording"""
        if self._audio_recorder.is_paused:
            self._audio_recorder.resume_recording()

            # Resume screen recording
            if self._screen_recorder.is_paused:
                self._screen_recorder.resume_recording()
                self._signals.screen_recording_resumed.emit()

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

            # Stop speaker diarizer
            if self._diarizer and self._diarizer.is_running():
                self._diarizer.stop()
                self._signals.diarization_stopped.emit()
                logger.info("Speaker diarizer stopped")

            # Stop screen recording
            video_path = None
            if self._screen_recorder.state != ScreenRecordingState.IDLE:
                video_path = self._screen_recorder.stop_recording()
                self._signals.screen_recording_stopped.emit()
                logger.info(f"Screen recording stopped: {video_path}")

            # Stop recording
            audio_path = self._audio_recorder.stop_recording()

            # Merge audio into video if both exist
            if video_path and audio_path:
                self._signals.status_message.emit("Merging audio into video...", 0)
                logger.info("Merging audio track into screen recording...")
                merged_path = self._screen_recorder.merge_audio_video(video_path, audio_path)
                if merged_path:
                    video_path = merged_path
                    logger.info(f"Audio/video merge complete: {merged_path}")
                    self._signals.status_message.emit("Recording saved with audio", 3000)
                else:
                    logger.warning("Failed to merge audio into video, keeping separate files")
                    self._signals.status_message.emit("Warning: Audio not merged into video", 5000)

            if self._current_session:
                # Update session metadata
                self._current_session.metadata.audio_file = str(audio_path) if audio_path else None
                self._current_session.metadata.video_file = str(video_path) if video_path else None
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

    def set_diarization_enabled(self, enabled: bool):
        """Enable or disable speaker diarization"""
        self._diarization_enabled = enabled
        logger.info(f"Speaker diarization {'enabled' if enabled else 'disabled'}")

    def is_diarization_enabled(self) -> bool:
        """Check if speaker diarization is enabled"""
        return self._diarization_enabled

    def is_diarization_model_loaded(self) -> bool:
        """Check if diarization model is loaded"""
        return self._diarizer.is_model_loaded() if self._diarizer else False

    def set_screen_recording_enabled(self, enabled: bool):
        """Enable or disable screen recording"""
        self._screen_recording_enabled = enabled
        logger.info(f"Screen recording {'enabled' if enabled else 'disabled'}")

    def is_screen_recording_enabled(self) -> bool:
        """Check if screen recording is enabled"""
        return self._screen_recording_enabled

    def is_screen_recording_available(self) -> bool:
        """Check if screen recording is available (dependencies installed)"""
        return self._screen_recorder.is_available()

    def get_available_monitors(self):
        """Get list of available monitors for screen recording"""
        return self._screen_recorder.get_monitors()

    def set_screen_recording_monitor(self, monitor_index: int):
        """Set the monitor to record (clears any window selection)"""
        self._screen_recorder.clear_window()
        self._screen_recorder.set_monitor(monitor_index)

    def set_screen_recording_window(self, window_handle: int, window_title: str = ""):
        """Set a specific window to record instead of a monitor"""
        self._screen_recorder.set_window(window_handle, window_title)


# Singleton
_controller_instance: Optional[RecordingController] = None


def get_recording_controller() -> RecordingController:
    """Get the singleton recording controller"""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = RecordingController()
    return _controller_instance
