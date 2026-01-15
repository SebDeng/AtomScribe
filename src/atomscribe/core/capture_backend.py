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

# Platform-specific imports for cursor capture
import sys
if sys.platform == "win32":
    try:
        import win32gui
        import win32ui
        import ctypes
        HAS_WIN32_CURSOR = True
    except ImportError:
        HAS_WIN32_CURSOR = False
else:
    HAS_WIN32_CURSOR = False


def _draw_cursor_on_image_win32(
    img: "Image.Image",
    region_x: int,
    region_y: int
) -> "Image.Image":
    """Draw the mouse cursor on the captured image (Windows only).

    Args:
        img: The captured image
        region_x: Left coordinate of the captured region
        region_y: Top coordinate of the captured region

    Returns:
        Image with cursor drawn on it
    """
    if not HAS_WIN32_CURSOR or not HAS_PIL:
        return img

    try:
        # Get cursor position
        cursor_x, cursor_y = win32gui.GetCursorPos()

        # Calculate cursor position relative to captured region
        rel_x = cursor_x - region_x
        rel_y = cursor_y - region_y

        # Check if cursor is within the image bounds
        if rel_x < 0 or rel_y < 0 or rel_x >= img.width or rel_y >= img.height:
            return img  # Cursor not in captured region

        # Get cursor info
        cursor_info = win32gui.GetCursorInfo()
        # cursor_info: (flags, hCursor, (x, y))
        if cursor_info[0] == 0:  # Cursor is hidden
            return img

        hcursor = cursor_info[1]

        # Get cursor icon info
        icon_info = win32gui.GetIconInfo(hcursor)
        # icon_info: (fIcon, xHotspot, yHotspot, hbmMask, hbmColor)
        hotspot_x = icon_info[1]
        hotspot_y = icon_info[2]
        hbm_mask = icon_info[3]
        hbm_color = icon_info[4]

        # Get cursor size (typically 32x32)
        cursor_size = 32

        # Create a DC for cursor capture
        screen_dc = win32gui.GetDC(0)
        mem_dc = win32ui.CreateDCFromHandle(screen_dc)
        save_dc = mem_dc.CreateCompatibleDC()

        # Create bitmap for cursor
        cursor_bitmap = win32ui.CreateBitmap()
        cursor_bitmap.CreateCompatibleBitmap(mem_dc, cursor_size, cursor_size)
        save_dc.SelectObject(cursor_bitmap)

        # Fill with transparent background (magenta for transparency)
        save_dc.FillSolidRect((0, 0, cursor_size, cursor_size), 0xFF00FF)

        # Draw the cursor
        win32gui.DrawIconEx(
            save_dc.GetSafeHdc(),
            0, 0,
            hcursor,
            cursor_size, cursor_size,
            0, None,
            3  # DI_NORMAL
        )

        # Convert cursor bitmap to PIL
        bmpinfo = cursor_bitmap.GetInfo()
        bmpstr = cursor_bitmap.GetBitmapBits(True)

        cursor_img = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1
        )

        # Cleanup cursor DC
        win32gui.DeleteObject(cursor_bitmap.GetHandle())
        save_dc.DeleteDC()
        mem_dc.DeleteDC()
        win32gui.ReleaseDC(0, screen_dc)

        # Cleanup icon bitmaps
        if hbm_mask:
            win32gui.DeleteObject(hbm_mask)
        if hbm_color:
            win32gui.DeleteObject(hbm_color)

        # Create mask from magenta background
        cursor_rgba = cursor_img.convert('RGBA')
        pixels = cursor_rgba.load()
        for y in range(cursor_rgba.height):
            for x in range(cursor_rgba.width):
                r, g, b, a = pixels[x, y]
                if r > 240 and g < 20 and b > 240:  # Magenta
                    pixels[x, y] = (0, 0, 0, 0)  # Transparent
                else:
                    pixels[x, y] = (r, g, b, 255)  # Opaque

        # Paste cursor onto image at correct position (accounting for hotspot)
        paste_x = rel_x - hotspot_x
        paste_y = rel_y - hotspot_y

        # Convert main image to RGBA if needed
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # Paste with alpha compositing
        img.paste(cursor_rgba, (paste_x, paste_y), cursor_rgba)

        # Convert back to RGB
        return img.convert('RGB')

    except Exception as e:
        logger.debug(f"Failed to draw cursor: {e}")
        return img


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
        Includes cursor overlay on Windows.
        """
        if not HAS_MSS or not HAS_PIL or not self._monitor_info:
            return None

        try:
            # Create new mss context for this capture (thread-safe)
            with mss.mss() as sct:
                # Capture the monitor
                screenshot = sct.grab(self._monitor_info)

                # Convert to PIL Image (RGB)
                img = Image.frombytes(
                    "RGB",
                    (screenshot.width, screenshot.height),
                    screenshot.rgb,
                )

                # Draw cursor on the image (Windows only)
                if sys.platform == "win32":
                    img = _draw_cursor_on_image_win32(
                        img,
                        self._monitor_info["left"],
                        self._monitor_info["top"]
                    )

                return img

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
