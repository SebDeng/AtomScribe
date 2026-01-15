"""Application configuration management"""

import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from loguru import logger


class AppConfig(BaseModel):
    """Application configuration"""

    # Default directory for saving sessions
    default_save_directory: Optional[str] = None

    # Audio settings
    audio_format: str = "mp3"  # mp3 or wav
    audio_sample_rate: int = 44100
    audio_channels: int = 1  # mono for speech
    audio_bitrate: int = 128  # kbps for mp3

    # Transcription settings
    transcription_language: Optional[str] = None  # None = auto-detect, "zh", "en", etc.
    transcription_initial_prompt: Optional[str] = None  # Domain-specific vocabulary hint
    transcription_model: str = "large-v3"  # Whisper model size
    transcription_hotwords: Optional[str] = None  # Comma-separated hotwords to boost

    # Post-processing replacements for common transcription errors
    # Format: {"wrong": "correct", ...}
    transcription_replacements: Optional[dict] = None

    # Speaker diarization settings
    diarization_enabled: bool = False  # Enable speaker diarization (disabled by default, code preserved)
    diarization_num_speakers: Optional[int] = None  # None = auto-detect (2-4)
    diarization_min_speakers: int = 2  # Minimum speakers for auto-detect
    diarization_max_speakers: int = 4  # Maximum speakers for auto-detect

    # Screen recording settings
    screen_recording_enabled: bool = True  # Enable screen recording by default
    screen_recording_fps: int = 5  # Frames per second (5 for better sync, 10 may drop frames)
    screen_recording_monitor: int = 0  # Monitor index (0 = primary, -1 = all monitors)
    screen_recording_quality: int = 23  # FFmpeg CRF value (0-51, lower = better quality, 23 = default)
    screen_recording_codec: str = "libx264"  # Video codec (libx264 for compatibility, libx265 for better compression)

    # Input recording settings (keyboard/mouse)
    input_recording_enabled: bool = True  # Enable input recording by default
    input_recording_keyboard: bool = True  # Record keyboard events
    input_recording_mouse_clicks: bool = True  # Record mouse clicks
    input_recording_mouse_scroll: bool = False  # Record mouse scroll (usually not needed)
    input_recording_mouse_moves: bool = False  # Record mouse movement (generates lots of data)

    # Click screenshot settings (capture screen on mouse click)
    click_screenshot_enabled: bool = True  # Enable click screenshots by default
    click_screenshot_quality: int = 85  # JPEG quality (1-100)

    # Document generation settings
    doc_generation_enabled: bool = True  # Enable document generation
    doc_generation_show_dialog: bool = True  # Show mode selection dialog after recording
    doc_generation_default_mode: str = "training"  # "training" | "experiment_log"

    # VLM server settings (llama.cpp server with Qwen3-VL-8B)
    vlm_server_url: str = "http://localhost:8080"  # llama-server address
    vlm_server_port: int = 8080  # Port for auto-started llama-server
    vlm_model_path: Optional[str] = None  # Path to Qwen3-VL-8B GGUF (auto-detected if None)
    vlm_mmproj_path: Optional[str] = None  # Path to vision encoder GGUF (auto-detected if None)
    vlm_auto_start_server: bool = True  # Auto-start llama-server when needed
    vlm_gpu_layers: int = 99  # Number of layers to offload to GPU (-1 = all)

    # UI settings
    last_window_geometry: Optional[str] = None
    last_selected_device: Optional[str] = None

    # First run flag
    is_first_run: bool = True


def get_config_dir() -> Path:
    """Get the application config directory"""
    import sys

    if sys.platform == "win32":
        config_dir = Path.home() / "AppData" / "Local" / "AtomScribe"
    elif sys.platform == "darwin":
        config_dir = Path.home() / "Library" / "Application Support" / "AtomScribe"
    else:
        config_dir = Path.home() / ".config" / "AtomScribe"

    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_path() -> Path:
    """Get the config file path"""
    return get_config_dir() / "config.json"


class ConfigManager:
    """Singleton config manager"""

    _instance: Optional["ConfigManager"] = None
    _config: Optional[AppConfig] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._config = self._load_config()

    def _load_config(self) -> AppConfig:
        """Load config from file or create default"""
        config_path = get_config_path()

        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Loaded config from {config_path}")
                return AppConfig(**data)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
                return AppConfig()
        else:
            logger.info("No config file found, using defaults")
            return AppConfig()

    def save(self):
        """Save config to file"""
        config_path = get_config_path()

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self._config.model_dump(), f, indent=2)
            logger.info(f"Saved config to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    @property
    def config(self) -> AppConfig:
        """Get the current config"""
        return self._config

    def set_default_save_directory(self, path: str):
        """Set the default save directory"""
        self._config.default_save_directory = path
        self._config.is_first_run = False
        self.save()

    def get_default_save_directory(self) -> Optional[Path]:
        """Get the default save directory as Path"""
        if self._config.default_save_directory:
            return Path(self._config.default_save_directory)
        return None

    def is_first_run(self) -> bool:
        """Check if this is the first run"""
        return self._config.is_first_run


def get_config_manager() -> ConfigManager:
    """Get the singleton config manager"""
    return ConfigManager()
