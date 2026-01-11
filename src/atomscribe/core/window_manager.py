"""Window manager for enumerating and capturing application windows.

This module provides platform-specific window enumeration and thumbnail capture
for the window-specific screen recording feature.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from PIL import Image as PILImage

# Platform-specific imports
if sys.platform == "win32":
    try:
        import win32gui
        import win32ui
        import win32con
        import win32process
        import win32api
        import ctypes
        from ctypes import wintypes
        HAS_WIN32 = True
    except ImportError:
        HAS_WIN32 = False
        logger.warning("pywin32 not installed. Window capture will not work on Windows.")
elif sys.platform == "darwin":
    try:
        import subprocess
        HAS_MACOS_CAPTURE = True
    except Exception:
        HAS_MACOS_CAPTURE = False
else:
    HAS_WIN32 = False
    HAS_MACOS_CAPTURE = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("PIL not installed. Window thumbnails will not work.")


@dataclass
class WindowInfo:
    """Information about a window."""
    handle: int  # Window handle (hwnd on Windows, window id on macOS)
    title: str
    process_name: str
    process_id: int
    x: int
    y: int
    width: int
    height: int
    is_visible: bool
    is_minimized: bool


def get_platform() -> str:
    """Get the current platform."""
    return sys.platform


def is_window_capture_available() -> bool:
    """Check if window capture is available on this platform."""
    if sys.platform == "win32":
        return HAS_WIN32 and HAS_PIL
    elif sys.platform == "darwin":
        return HAS_MACOS_CAPTURE
    return False


def get_windows(exclude_own: bool = True) -> List[WindowInfo]:
    """Get list of all visible windows.

    Args:
        exclude_own: If True, exclude the current application's windows

    Returns:
        List of WindowInfo objects
    """
    if sys.platform == "win32":
        return _get_windows_win32(exclude_own)
    elif sys.platform == "darwin":
        return _get_windows_macos(exclude_own)
    else:
        logger.warning(f"Window enumeration not supported on {sys.platform}")
        return []


def get_window_thumbnail(
    handle: int,
    max_width: int = 200,
    max_height: int = 120
) -> Optional[bytes]:
    """Capture a thumbnail of the specified window.

    Args:
        handle: Window handle
        max_width: Maximum thumbnail width
        max_height: Maximum thumbnail height

    Returns:
        PNG image bytes, or None if failed
    """
    if sys.platform == "win32":
        return _get_window_thumbnail_win32(handle, max_width, max_height)
    elif sys.platform == "darwin":
        return _get_window_thumbnail_macos(handle, max_width, max_height)
    else:
        return None


def is_window_valid(handle: int) -> bool:
    """Check if a window handle is still valid.

    Args:
        handle: Window handle

    Returns:
        True if window exists and is valid
    """
    if sys.platform == "win32" and HAS_WIN32:
        return win32gui.IsWindow(handle)
    elif sys.platform == "darwin":
        # macOS: we'd need to re-enumerate windows to check
        # For simplicity, assume valid unless we get an error
        return True
    return False


def get_window_rect(handle: int) -> Optional[Tuple[int, int, int, int]]:
    """Get the current window rectangle.

    Args:
        handle: Window handle

    Returns:
        Tuple of (x, y, width, height), or None if failed
    """
    if sys.platform == "win32" and HAS_WIN32:
        try:
            rect = win32gui.GetWindowRect(handle)
            x, y, right, bottom = rect
            return (x, y, right - x, bottom - y)
        except Exception as e:
            logger.error(f"Failed to get window rect: {e}")
            return None
    return None


# =============================================================================
# Windows-specific implementation
# =============================================================================

def _get_windows_win32(exclude_own: bool = True) -> List[WindowInfo]:
    """Get windows using Win32 API."""
    if not HAS_WIN32:
        return []

    windows = []
    current_pid = win32api.GetCurrentProcessId() if exclude_own else -1

    def enum_callback(hwnd, _):
        # Skip invisible windows
        if not win32gui.IsWindowVisible(hwnd):
            return True

        # Get window title
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return True

        # Get window rect
        try:
            rect = win32gui.GetWindowRect(hwnd)
            x, y, right, bottom = rect
            width = right - x
            height = bottom - y
        except Exception:
            return True

        # Skip windows with no size
        if width <= 0 or height <= 0:
            return True

        # Skip windows that are too small (likely system windows)
        if width < 100 or height < 50:
            return True

        # Get process info
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
        except Exception:
            pid = 0

        # Skip own process if requested
        if exclude_own and pid == current_pid:
            return True

        # Get process name
        process_name = _get_process_name_win32(pid)

        # Skip certain system windows
        if process_name.lower() in ["explorer.exe", "shellexperiencehost.exe",
                                     "searchhost.exe", "startmenuexperiencehost.exe",
                                     "textinputhost.exe", "applicationframehost.exe"]:
            # Allow explorer.exe windows that have a title (like File Explorer)
            if process_name.lower() == "explorer.exe" and title:
                pass  # Keep it
            else:
                return True

        # Check if minimized
        is_minimized = win32gui.IsIconic(hwnd)

        windows.append(WindowInfo(
            handle=hwnd,
            title=title,
            process_name=process_name,
            process_id=pid,
            x=x,
            y=y,
            width=width,
            height=height,
            is_visible=True,
            is_minimized=is_minimized,
        ))

        return True

    try:
        win32gui.EnumWindows(enum_callback, None)
    except Exception as e:
        logger.error(f"Failed to enumerate windows: {e}")

    # Sort by title for consistent ordering
    windows.sort(key=lambda w: w.title.lower())

    return windows


def _get_process_name_win32(pid: int) -> str:
    """Get process name from PID on Windows."""
    if not HAS_WIN32 or pid == 0:
        return "Unknown"

    try:
        # Try to open the process and get its name
        import win32process
        import win32con

        handle = win32api.OpenProcess(
            win32con.PROCESS_QUERY_LIMITED_INFORMATION,
            False,
            pid
        )
        try:
            exe_path = win32process.GetModuleFileNameEx(handle, 0)
            return exe_path.split("\\")[-1]
        finally:
            win32api.CloseHandle(handle)
    except Exception:
        return "Unknown"


def _get_window_thumbnail_win32(
    hwnd: int,
    max_width: int,
    max_height: int
) -> Optional[bytes]:
    """Capture window thumbnail using Win32 API."""
    if not HAS_WIN32 or not HAS_PIL:
        return None

    try:
        # Get window dimensions
        rect = win32gui.GetWindowRect(hwnd)
        x, y, right, bottom = rect
        width = right - x
        height = bottom - y

        if width <= 0 or height <= 0:
            return None

        # Get the window device context
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        # Create a bitmap to hold the captured image
        save_bitmap = win32ui.CreateBitmap()
        save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(save_bitmap)

        # Use PrintWindow for better capture (works with layered windows)
        # PW_RENDERFULLCONTENT = 2 (captures even when window is obscured)
        try:
            result = ctypes.windll.user32.PrintWindow(
                hwnd,
                save_dc.GetSafeHdc(),
                2  # PW_RENDERFULLCONTENT
            )
        except Exception:
            # Fallback to BitBlt
            save_dc.BitBlt(
                (0, 0), (width, height),
                mfc_dc, (0, 0),
                win32con.SRCCOPY
            )

        # Convert bitmap to bytes
        bmpinfo = save_bitmap.GetInfo()
        bmpstr = save_bitmap.GetBitmapBits(True)

        # Create PIL Image
        img = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1
        )

        # Cleanup
        win32gui.DeleteObject(save_bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)

        # Resize to thumbnail
        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

        # Convert to PNG bytes
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()

    except Exception as e:
        logger.error(f"Failed to capture window thumbnail: {e}")
        return None


# =============================================================================
# macOS-specific implementation
# =============================================================================

def _get_windows_macos(exclude_own: bool = True) -> List[WindowInfo]:
    """Get windows using macOS APIs."""
    if not HAS_MACOS_CAPTURE:
        return []

    windows = []

    try:
        # Use Quartz to enumerate windows
        import Quartz

        # Get window list
        window_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID
        )

        current_pid = os.getpid() if exclude_own else -1

        for window in window_list:
            # Get window info
            window_id = window.get(Quartz.kCGWindowNumber, 0)
            title = window.get(Quartz.kCGWindowName, "")
            owner_name = window.get(Quartz.kCGWindowOwnerName, "Unknown")
            owner_pid = window.get(Quartz.kCGWindowOwnerPID, 0)

            # Skip own windows
            if exclude_own and owner_pid == current_pid:
                continue

            # Get bounds
            bounds = window.get(Quartz.kCGWindowBounds, {})
            x = int(bounds.get('X', 0))
            y = int(bounds.get('Y', 0))
            width = int(bounds.get('Width', 0))
            height = int(bounds.get('Height', 0))

            # Skip small/invalid windows
            if width < 100 or height < 50:
                continue

            # Skip windows without title (usually system windows)
            if not title:
                continue

            windows.append(WindowInfo(
                handle=window_id,
                title=title,
                process_name=owner_name,
                process_id=owner_pid,
                x=x,
                y=y,
                width=width,
                height=height,
                is_visible=True,
                is_minimized=False,  # Can't easily detect on macOS
            ))

    except ImportError:
        logger.warning("Quartz not available. Using screencapture CLI fallback.")
        # Fallback: can't enumerate windows without Quartz
        return []
    except Exception as e:
        logger.error(f"Failed to enumerate windows on macOS: {e}")

    return windows


def _get_window_thumbnail_macos(
    window_id: int,
    max_width: int,
    max_height: int
) -> Optional[bytes]:
    """Capture window thumbnail using macOS screencapture."""
    if not HAS_PIL:
        return None

    try:
        import tempfile
        import subprocess
        import os

        # Create temp file for screenshot
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name

        try:
            # Use screencapture to capture specific window
            # -l <windowid> captures the specified window
            # -o excludes window shadow
            subprocess.run(
                ['screencapture', '-l', str(window_id), '-o', '-x', temp_path],
                check=True,
                capture_output=True
            )

            # Load and resize
            img = Image.open(temp_path)
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

            # Convert to PNG bytes
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            return buffer.getvalue()

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        logger.error(f"Failed to capture window thumbnail on macOS: {e}")
        return None


# =============================================================================
# Capture a window frame for recording
# =============================================================================

def capture_window_frame(handle: int) -> Optional[Image.Image]:
    """Capture a full-resolution frame of a window for recording.

    Args:
        handle: Window handle

    Returns:
        PIL Image, or None if failed
    """
    if sys.platform == "win32":
        return _capture_window_frame_win32(handle)
    elif sys.platform == "darwin":
        return _capture_window_frame_macos(handle)
    return None


def _draw_cursor_on_image(img: Image.Image, window_x: int, window_y: int) -> Image.Image:
    """Draw the mouse cursor on the captured image.

    Args:
        img: The captured image
        window_x: Window left coordinate
        window_y: Window top coordinate

    Returns:
        Image with cursor drawn on it
    """
    if not HAS_WIN32 or not HAS_PIL:
        return img

    try:
        # Get cursor position
        cursor_x, cursor_y = win32gui.GetCursorPos()

        # Calculate cursor position relative to window
        rel_x = cursor_x - window_x
        rel_y = cursor_y - window_y

        # Check if cursor is within the image bounds
        if rel_x < 0 or rel_y < 0 or rel_x >= img.width or rel_y >= img.height:
            return img  # Cursor not in window

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


def _capture_window_frame_win32(hwnd: int) -> Optional[Image.Image]:
    """Capture full window frame on Windows."""
    if not HAS_WIN32 or not HAS_PIL:
        return None

    try:
        # Get window dimensions
        rect = win32gui.GetWindowRect(hwnd)
        x, y, right, bottom = rect
        width = right - x
        height = bottom - y

        if width <= 0 or height <= 0:
            return None

        # Get the window device context
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        # Create a bitmap to hold the captured image
        save_bitmap = win32ui.CreateBitmap()
        save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(save_bitmap)

        # Use PrintWindow for better capture
        try:
            ctypes.windll.user32.PrintWindow(
                hwnd,
                save_dc.GetSafeHdc(),
                2  # PW_RENDERFULLCONTENT
            )
        except Exception:
            # Fallback to BitBlt
            save_dc.BitBlt(
                (0, 0), (width, height),
                mfc_dc, (0, 0),
                win32con.SRCCOPY
            )

        # Convert bitmap to PIL Image
        bmpinfo = save_bitmap.GetInfo()
        bmpstr = save_bitmap.GetBitmapBits(True)

        img = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1
        )

        # Cleanup
        win32gui.DeleteObject(save_bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)

        # Draw cursor on the captured image
        img = _draw_cursor_on_image(img, x, y)

        return img

    except Exception as e:
        logger.error(f"Failed to capture window frame: {e}")
        return None


def _capture_window_frame_macos(window_id: int) -> Optional[Image.Image]:
    """Capture full window frame on macOS."""
    if not HAS_PIL:
        return None

    try:
        import tempfile
        import subprocess
        import os

        # Create temp file for screenshot
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name

        try:
            # Use screencapture to capture specific window
            subprocess.run(
                ['screencapture', '-l', str(window_id), '-o', '-x', temp_path],
                check=True,
                capture_output=True
            )

            # Load and return
            return Image.open(temp_path).copy()

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        logger.error(f"Failed to capture window frame on macOS: {e}")
        return None
