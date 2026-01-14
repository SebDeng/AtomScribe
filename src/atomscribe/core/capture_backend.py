"""Abstract capture backend interface for screen recording.

This module defines the abstract interface for capture backends,
allowing both monitor capture (mss) and window capture (platform-specific)
to be used interchangeably by the screen recorder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from PIL import Image as PILImage

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False
    mss = None


class CaptureType(Enum):
    """Type of capture target."""
    MONITOR = "monitor"
    WINDOW = "window"


@dataclass
class CaptureTarget:
    """Target for capture - either a monitor or a window."""
    capture_type: CaptureType
    # For monitors: index (0, 1, 2... or -1 for all)
    # For windows: window handle (hwnd)
    target_id: int
    # Human-readable name for logging
    name: str


class CaptureBackend(ABC):
    """Abstract base class for capture backends."""

    @abstractmethod
    def get_region(self) -> Optional[Tuple[int, int, int, int]]:
        """Get the current capture region.

        Returns:
            Tuple of (x, y, width, height), or None if not available
        """
        pass

    @abstractmethod
    def capture_frame(self) -> Optional[Image.Image]:
        """Capture a single frame.

        Returns:
            PIL Image, or None if capture failed
        """
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        """Check if the capture target is still valid.

        Returns:
            True if the target exists and can be captured
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name/description of the capture target."""
        pass


class MonitorCaptureBackend(CaptureBackend):
    """Capture backend for monitors using mss.

    Note: mss is not thread-safe. We create a new mss context for each capture
    to ensure it works correctly when called from different threads.
    """

    def __init__(self, monitor_index: int):
        """Initialize monitor capture.

        Args:
            monitor_index: Monitor index (0, 1, 2... or -1 for all monitors)
        """
        self._monitor_index = monitor_index
        self._mss_index: int = 0
        self._monitor_info: Optional[dict] = None
        self._initialized = False

        if HAS_MSS:
            self._init_monitor_info()

    def _init_monitor_info(self):
        """Initialize monitor info (but not the mss context - that's per-capture)."""
        try:
            # Create temporary mss context just to get monitor info
            with mss.mss() as sct:
                # Get monitor info
                if self._monitor_index == -1:
                    # All monitors combined
                    self._mss_index = 0
                else:
                    # Specific monitor (mss uses 1-based indexing)
                    self._mss_index = self._monitor_index + 1

                # Validate index
                if self._mss_index >= len(sct.monitors):
                    logger.warning(f"Monitor index {self._monitor_index} out of range, using 0")
                    self._mss_index = 1
                    self._monitor_index = 0

                self._monitor_info = dict(sct.monitors[self._mss_index])
                self._initialized = True
                logger.debug(f"Monitor capture initialized: {self._monitor_info}")

        except Exception as e:
            logger.error(f"Failed to initialize monitor capture: {e}")
            self._monitor_info = None
            self._initialized = False

    def get_region(self) -> Optional[Tuple[int, int, int, int]]:
        """Get the monitor region."""
        if not self._monitor_info:
            return None

        return (
            self._monitor_info["left"],
            self._monitor_info["top"],
            self._monitor_info["width"],
            self._monitor_info["height"],
        )

    def capture_frame(self) -> Optional[Image.Image]:
        """Capture a frame from the monitor.

        Creates a new mss context for each capture to ensure thread safety.
        """
        if not HAS_MSS or not HAS_PIL or not self._monitor_info:
            return None

        try:
            # Create new mss context for this capture (thread-safe)
            with mss.mss() as sct:
                # Capture the monitor
                screenshot = sct.grab(self._monitor_info)

                # Convert to PIL Image (RGB)
                return Image.frombytes(
                    "RGB",
                    (screenshot.width, screenshot.height),
                    screenshot.rgb,
                )

        except Exception as e:
            logger.error(f"Failed to capture monitor frame: {e}")
            return None

    def is_valid(self) -> bool:
        """Monitors are always valid (unless initialization failed)."""
        return self._initialized and self._monitor_info is not None

    def get_name(self) -> str:
        """Get monitor name."""
        if self._monitor_index == -1:
            return "All Monitors"
        return f"Monitor {self._monitor_index + 1}"

    def close(self):
        """Clean up (no-op since we use context managers now)."""
        pass


class WindowCaptureBackend(CaptureBackend):
    """Capture backend for specific windows."""

    def __init__(self, window_handle: int, window_title: str = ""):
        """Initialize window capture.

        Args:
            window_handle: Window handle (hwnd on Windows, window id on macOS)
            window_title: Window title for display purposes
        """
        self._handle = window_handle
        self._title = window_title
        self._last_valid_frame: Optional[Image.Image] = None

    def get_region(self) -> Optional[Tuple[int, int, int, int]]:
        """Get the window region."""
        from . import window_manager

        rect = window_manager.get_window_rect(self._handle)
        return rect

    def capture_frame(self) -> Optional[Image.Image]:
        """Capture a frame from the window."""
        from . import window_manager

        # Check if window is still valid
        if not window_manager.is_window_valid(self._handle):
            logger.warning(f"Window {self._handle} is no longer valid")
            return self._last_valid_frame  # Return last good frame

        # Capture the window
        frame = window_manager.capture_window_frame(self._handle)

        if frame is not None:
            self._last_valid_frame = frame
            return frame
        elif self._last_valid_frame is not None:
            # Return last valid frame if capture failed
            logger.debug("Using last valid frame due to capture failure")
            return self._last_valid_frame

        return None

    def is_valid(self) -> bool:
        """Check if the window is still valid."""
        from . import window_manager
        return window_manager.is_window_valid(self._handle)

    def get_name(self) -> str:
        """Get window name."""
        return self._title or f"Window {self._handle}"

    @property
    def handle(self) -> int:
        """Get the window handle."""
        return self._handle


def create_capture_backend(target: CaptureTarget) -> Optional[CaptureBackend]:
    """Factory function to create appropriate capture backend.

    Args:
        target: The capture target specification

    Returns:
        CaptureBackend instance, or None if creation failed
    """
    if target.capture_type == CaptureType.MONITOR:
        return MonitorCaptureBackend(target.target_id)
    elif target.capture_type == CaptureType.WINDOW:
        return WindowCaptureBackend(target.target_id, target.name)
    else:
        logger.error(f"Unknown capture type: {target.capture_type}")
        return None
