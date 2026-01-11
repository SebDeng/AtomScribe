"""Session management for AtomScribe"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
from loguru import logger

from .config import get_config_manager


class SessionMetadata(BaseModel):
    """Metadata for a recording session"""

    # Unique session ID (timestamp-based)
    session_id: str

    # Display name (can be renamed by user)
    name: str

    # Creation timestamp
    created_at: str

    # Recording duration in seconds
    duration_seconds: int = 0

    # Recording status
    status: str = "created"  # created, recording, paused, completed

    # Audio file info
    audio_file: Optional[str] = None
    audio_format: str = "wav"

    # Video file info (screen recording)
    video_file: Optional[str] = None
    video_format: str = "mp4"

    # Transcript file
    transcript_file: Optional[str] = None

    # Summary file
    summary_file: Optional[str] = None

    # Events file
    events_file: Optional[str] = None

    # Notes
    notes: str = ""


class Session:
    """Represents a recording session"""

    def __init__(self, directory: Path, metadata: SessionMetadata):
        self.directory = directory
        self.metadata = metadata

    @classmethod
    def create_new(cls, base_directory: Path, name: Optional[str] = None) -> "Session":
        """Create a new session.

        Directory structure: base_directory/YYYY-MM-DD/session_name/
        If no name is provided, uses timestamp (HHMMSS) as folder name.
        """
        # Generate session ID from timestamp
        now = datetime.now()
        session_id = now.strftime("%Y%m%d_%H%M%S")
        date_folder = now.strftime("%Y-%m-%d")

        # Generate default name if not provided (use time as folder name)
        if name is None or name.strip() == "":
            name = now.strftime("%H%M%S")  # e.g., "180530"

        # Create session directory: base/YYYY-MM-DD/name/
        session_dir = base_directory / date_folder / name
        session_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata
        metadata = SessionMetadata(
            session_id=session_id,
            name=name,
            created_at=now.isoformat(),
        )

        session = cls(session_dir, metadata)
        session.save_metadata()

        logger.info(f"Created new session: {name} at {session_dir}")
        return session

    @classmethod
    def load_from_directory(cls, directory: Path) -> Optional["Session"]:
        """Load a session from an existing directory"""
        metadata_path = directory / "session.json"

        if not metadata_path.exists():
            logger.warning(f"No session.json found in {directory}")
            return None

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            metadata = SessionMetadata(**data)
            logger.info(f"Loaded session: {metadata.name}")
            return cls(directory, metadata)
        except Exception as e:
            logger.error(f"Failed to load session from {directory}: {e}")
            return None

    def save_metadata(self):
        """Save session metadata to file"""
        metadata_path = self.directory / "session.json"

        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata.model_dump(), f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved session metadata to {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save session metadata: {e}")

    def get_audio_path(self) -> Path:
        """Get the path for the audio file.

        Each session has its own folder, so simple filename is sufficient.
        """
        # If we already have an audio file path, return it
        if self.metadata.audio_file:
            return Path(self.metadata.audio_file)

        # Simple filename - folder provides uniqueness
        ext = self.metadata.audio_format
        audio_path = self.directory / f"recording.{ext}"

        # Store it in metadata
        self.metadata.audio_file = str(audio_path)
        self.save_metadata()

        return audio_path

    def get_video_path(self) -> Path:
        """Get the path for the video file (screen recording).

        Each session has its own folder, so simple filename is sufficient.
        """
        # If we already have a video file path, return it
        if self.metadata.video_file:
            return Path(self.metadata.video_file)

        # Simple filename - folder provides uniqueness
        ext = self.metadata.video_format
        video_path = self.directory / f"screen.{ext}"

        # Store it in metadata
        self.metadata.video_file = str(video_path)
        self.save_metadata()

        return video_path

    def get_transcript_path(self) -> Path:
        """Get the path for the transcript file."""
        return self.directory / "transcript.json"

    def get_summary_path(self) -> Path:
        """Get the path for the summary file."""
        return self.directory / "summary.md"

    def get_events_path(self) -> Path:
        """Get the path for the events file."""
        return self.directory / "events.json"

    def update_duration(self, seconds: int):
        """Update the recording duration (saves to disk every 10 seconds)"""
        self.metadata.duration_seconds = seconds
        # Only save metadata every 10 seconds to reduce disk I/O
        if seconds % 10 == 0:
            self.save_metadata()

    def set_status(self, status: str):
        """Update the session status"""
        self.metadata.status = status
        self.save_metadata()

    def rename(self, new_name: str):
        """Rename the session (and optionally the directory)"""
        old_name = self.metadata.name
        self.metadata.name = new_name

        # Optionally rename directory
        new_dir = self.directory.parent / new_name
        if not new_dir.exists() and self.directory.name == old_name:
            try:
                self.directory.rename(new_dir)
                self.directory = new_dir
                logger.info(f"Renamed session directory: {old_name} -> {new_name}")
            except Exception as e:
                logger.warning(f"Could not rename directory: {e}")

        self.save_metadata()


class SessionManager:
    """Manages all sessions"""

    _instance: Optional["SessionManager"] = None
    _current_session: Optional[Session] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._config = get_config_manager()

    @property
    def current_session(self) -> Optional[Session]:
        """Get the current active session"""
        return self._current_session

    def create_session(self, name: Optional[str] = None, directory: Optional[Path] = None) -> Session:
        """Create a new session"""
        # Use provided directory or default
        if directory is None:
            directory = self._config.get_default_save_directory()
            if directory is None:
                raise ValueError("No save directory configured. Please set a default directory.")

        session = Session.create_new(directory, name)
        self._current_session = session
        return session

    def load_session(self, directory: Path) -> Optional[Session]:
        """Load an existing session"""
        session = Session.load_from_directory(directory)
        if session:
            self._current_session = session
        return session

    def list_sessions(self, directory: Optional[Path] = None) -> List[Session]:
        """List all sessions in a directory"""
        if directory is None:
            directory = self._config.get_default_save_directory()
            if directory is None:
                return []

        sessions = []
        if directory.exists():
            for item in directory.iterdir():
                if item.is_dir():
                    session = Session.load_from_directory(item)
                    if session:
                        sessions.append(session)

        # Sort by creation time, newest first
        sessions.sort(key=lambda s: s.metadata.created_at, reverse=True)
        return sessions

    def close_current_session(self):
        """Close the current session"""
        if self._current_session:
            self._current_session.set_status("completed")
            self._current_session = None


def get_session_manager() -> SessionManager:
    """Get the singleton session manager"""
    return SessionManager()
