"""Core business logic"""

from .config import AppConfig, ConfigManager, get_config_manager
from .session import Session, SessionMetadata, SessionManager, get_session_manager
from .audio_recorder import AudioRecorder, AudioDevice, RecordingState, get_audio_recorder
from .recording_controller import RecordingController, get_recording_controller
from .transcriber import RealtimeTranscriber, TranscriptSegment, get_transcriber

# Document generation
from .doc_generator import DocumentGenerator, GenerationMode, get_document_generator
from .transcript_analyzer import TranscriptAnalyzer, KeyPoint, KeyPointType, create_transcript_analyzer
from .frame_extractor import FrameExtractor, ExtractedFrame, create_frame_extractor
from .vlm_processor import VLMProcessor, CropRegion, get_vlm_processor
from .markdown_writer import MarkdownWriter, ImageReference, create_markdown_writer

__all__ = [
    "AppConfig",
    "ConfigManager",
    "get_config_manager",
    "Session",
    "SessionMetadata",
    "SessionManager",
    "get_session_manager",
    "AudioRecorder",
    "AudioDevice",
    "RecordingState",
    "get_audio_recorder",
    "RecordingController",
    "get_recording_controller",
    "RealtimeTranscriber",
    "TranscriptSegment",
    "get_transcriber",
    # Document generation
    "DocumentGenerator",
    "GenerationMode",
    "get_document_generator",
    "TranscriptAnalyzer",
    "KeyPoint",
    "KeyPointType",
    "create_transcript_analyzer",
    "FrameExtractor",
    "ExtractedFrame",
    "create_frame_extractor",
    "VLMProcessor",
    "CropRegion",
    "get_vlm_processor",
    "MarkdownWriter",
    "ImageReference",
    "create_markdown_writer",
]
