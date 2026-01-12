"""Application-wide signals for cross-component communication"""

from PySide6.QtCore import QObject, Signal


class AppSignals(QObject):
    """
    Singleton class containing all application-wide signals.
    Use get_app_signals() to access the instance.
    """

    _instance = None

    # ===== Recording signals =====
    # Request signals (from UI buttons)
    record_button_clicked = Signal()  # REC button clicked - triggers dialog
    stop_button_clicked = Signal()    # Stop button clicked
    pause_button_clicked = Signal()   # Pause/Resume button clicked

    # State signals (actual recording state changes)
    recording_started = Signal()      # Recording actually started
    recording_paused = Signal()
    recording_resumed = Signal()
    recording_stopped = Signal()
    recording_saved = Signal(str)     # Path to saved recording
    recording_time_updated = Signal(int)  # elapsed seconds

    # ===== Audio signals =====
    audio_level_updated = Signal(float)  # 0.0 - 1.0
    audio_device_changed = Signal(str)

    # ===== Transcript signals =====
    transcript_segment = Signal(object)  # TranscriptSegment object (new segment)
    transcript_segment_updated = Signal(object)  # TranscriptSegment object (update to existing)
    transcript_segment_corrected = Signal(object)  # CorrectionResult object (LLM corrected)
    transcript_received = Signal(dict)  # {"timestamp", "speaker", "text", "is_final"}
    transcript_updated = Signal(str, str)  # (id, new_text)
    transcription_model_loaded = Signal()  # Model finished loading
    transcription_started = Signal()
    transcription_stopped = Signal()

    # ===== LLM signals =====
    llm_model_loaded = Signal()  # LLM model finished loading
    llm_processing_started = Signal()
    llm_processing_stopped = Signal()

    # ===== Speaker diarization signals =====
    speaker_identified = Signal(object)  # SpeakerResult object (supports multi-speaker segments)
    speaker_updated = Signal(object)  # Re-assignment after re-clustering
    diarization_model_loaded = Signal()
    diarization_started = Signal()
    diarization_stopped = Signal()

    # ===== Screen recording signals =====
    screen_recording_started = Signal()
    screen_recording_paused = Signal()
    screen_recording_resumed = Signal()
    screen_recording_stopped = Signal()
    screen_recording_error = Signal(str)  # Error message

    # ===== Input recording signals (keyboard/mouse) =====
    input_recording_started = Signal()
    input_recording_paused = Signal()
    input_recording_resumed = Signal()
    input_recording_stopped = Signal()
    input_event_recorded = Signal(object)  # InputEvent object

    # ===== Session signals =====
    session_created = Signal(str)  # session_id
    session_opened = Signal(str)
    session_closed = Signal(str)
    session_deleted = Signal(str)

    # ===== File signals =====
    file_selected = Signal(str)  # file_path
    file_preview_requested = Signal(str)

    # ===== UI signals =====
    status_message = Signal(str, int)  # (message, timeout_ms)
    busy_state_changed = Signal(bool)

    def __init__(self):
        # Only initialize once
        if not hasattr(self, '_initialized'):
            super().__init__()
            self._initialized = True


# Module-level singleton instance
_app_signals_instance: AppSignals | None = None


def get_app_signals() -> AppSignals:
    """Get the singleton AppSignals instance"""
    global _app_signals_instance
    if _app_signals_instance is None:
        _app_signals_instance = AppSignals()
    return _app_signals_instance
