"""Input recorder - captures keyboard and mouse events.

This module uses pynput to record keyboard and mouse events
with timestamps and active window context.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any
from loguru import logger

# Check for pynput availability
try:
    from pynput import keyboard, mouse
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False
    keyboard = None
    mouse = None

# Platform-specific window title retrieval
if sys.platform == "win32":
    try:
        import win32gui
        HAS_WIN32 = True
    except ImportError:
        HAS_WIN32 = False
        win32gui = None
elif sys.platform == "darwin":
    HAS_WIN32 = False
    win32gui = None
    # macOS: use AppKit if available
    try:
        from AppKit import NSWorkspace
        HAS_APPKIT = True
    except ImportError:
        HAS_APPKIT = False
        NSWorkspace = None
else:
    HAS_WIN32 = False
    HAS_APPKIT = False
    win32gui = None
    NSWorkspace = None


class InputEventType(Enum):
    """Type of input event."""
    MOUSE_MOVE = "mouse_move"
    MOUSE_CLICK = "mouse_click"
    MOUSE_SCROLL = "mouse_scroll"
    KEY_PRESS = "key_press"
    KEY_RELEASE = "key_release"


@dataclass
class InputEvent:
    """Represents a single input event."""
    timestamp: float  # Time since recording start (seconds)
    event_type: InputEventType

    # Mouse event data
    x: Optional[int] = None
    y: Optional[int] = None
    button: Optional[str] = None  # left, right, middle
    pressed: Optional[bool] = None  # True for press, False for release
    scroll_dx: Optional[int] = None
    scroll_dy: Optional[int] = None

    # Keyboard event data
    key: Optional[str] = None  # Key name or character
    key_code: Optional[int] = None  # Virtual key code

    # Context
    active_window: Optional[str] = None  # Active window title

    # Screenshot (for mouse clicks with click_screenshot enabled)
    screenshot_path: Optional[str] = None  # Relative path to click screenshot

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "timestamp": round(self.timestamp, 3),
            "event_type": self.event_type.value,
        }

        # Only include non-None values
        if self.x is not None:
            result["x"] = self.x
        if self.y is not None:
            result["y"] = self.y
        if self.button is not None:
            result["button"] = self.button
        if self.pressed is not None:
            result["pressed"] = self.pressed
        if self.scroll_dx is not None:
            result["scroll_dx"] = self.scroll_dx
        if self.scroll_dy is not None:
            result["scroll_dy"] = self.scroll_dy
        if self.key is not None:
            result["key"] = self.key
        if self.key_code is not None:
            result["key_code"] = self.key_code
        if self.active_window is not None:
            result["active_window"] = self.active_window
        if self.screenshot_path is not None:
            result["screenshot_path"] = self.screenshot_path

        return result


class InputRecorderState(Enum):
    """Recording state."""
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"


class InputRecorder:
    """Records keyboard and mouse events with timestamps."""

    def __init__(
        self,
        record_mouse_moves: bool = False,  # Mouse moves generate lots of data
        record_mouse_clicks: bool = True,
        record_mouse_scroll: bool = True,
        record_keyboard: bool = True,
        move_throttle_ms: int = 100,  # Throttle mouse move events
    ):
        """Initialize the input recorder.

        Args:
            record_mouse_moves: Whether to record mouse movement events
            record_mouse_clicks: Whether to record mouse click events
            record_mouse_scroll: Whether to record mouse scroll events
            record_keyboard: Whether to record keyboard events
            move_throttle_ms: Minimum ms between mouse move events
        """
        self._record_mouse_moves = record_mouse_moves
        self._record_mouse_clicks = record_mouse_clicks
        self._record_mouse_scroll = record_mouse_scroll
        self._record_keyboard = record_keyboard
        self._move_throttle_ms = move_throttle_ms

        self._state = InputRecorderState.IDLE
        self._events: List[InputEvent] = []
        self._start_time: Optional[float] = None
        self._pause_time: Optional[float] = None
        self._total_paused: float = 0.0

        self._keyboard_listener: Optional[keyboard.Listener] = None
        self._mouse_listener: Optional[mouse.Listener] = None

        self._output_path: Optional[Path] = None
        self._lock = threading.Lock()

        # Throttle tracking for mouse moves
        self._last_move_time: float = 0.0

        # Callback for events (optional)
        self._on_event_callback: Optional[Callable[[InputEvent], None]] = None

        # Callback for click screenshots (called on mouse press)
        self._on_click_screenshot: Optional[Callable[[float, int, int], Optional[str]]] = None

        # Write buffer
        self._write_buffer: List[InputEvent] = []
        self._write_interval: float = 5.0  # Write to disk every 5 seconds
        self._last_write_time: float = 0.0
        self._write_thread: Optional[threading.Thread] = None
        self._stop_write_thread: bool = False

    @property
    def state(self) -> InputRecorderState:
        """Get current recording state."""
        return self._state

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._state == InputRecorderState.RECORDING

    @property
    def is_paused(self) -> bool:
        """Check if recording is paused."""
        return self._state == InputRecorderState.PAUSED

    def set_on_event_callback(self, callback: Optional[Callable[[InputEvent], None]]):
        """Set callback for input events."""
        self._on_event_callback = callback

    def set_click_screenshot_callback(
        self,
        callback: Optional[Callable[[float, int, int], Optional[str]]]
    ):
        """Set callback for capturing screenshots on mouse click.

        The callback receives (timestamp, x, y) and should return the
        relative path to the saved screenshot, or None if capture failed.
        """
        self._on_click_screenshot = callback

    def _get_active_window_title(self) -> Optional[str]:
        """Get the title of the currently active window."""
        try:
            if sys.platform == "win32" and HAS_WIN32:
                hwnd = win32gui.GetForegroundWindow()
                if hwnd:
                    return win32gui.GetWindowText(hwnd)
            elif sys.platform == "darwin" and HAS_APPKIT:
                active_app = NSWorkspace.sharedWorkspace().activeApplication()
                if active_app:
                    return active_app.get("NSApplicationName", "")
        except Exception as e:
            logger.debug(f"Failed to get active window: {e}")
        return None

    def _get_timestamp(self) -> float:
        """Get current timestamp relative to recording start."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time - self._total_paused

    def _add_event(self, event: InputEvent):
        """Add an event to the buffer."""
        if self._state != InputRecorderState.RECORDING:
            return

        with self._lock:
            self._events.append(event)
            self._write_buffer.append(event)

        # Call callback if set
        if self._on_event_callback:
            try:
                self._on_event_callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    def _on_mouse_move(self, x: int, y: int):
        """Handle mouse move event."""
        if not self._record_mouse_moves or self._state != InputRecorderState.RECORDING:
            return

        # Throttle move events
        current_time = time.time() * 1000  # Convert to ms
        if current_time - self._last_move_time < self._move_throttle_ms:
            return
        self._last_move_time = current_time

        event = InputEvent(
            timestamp=self._get_timestamp(),
            event_type=InputEventType.MOUSE_MOVE,
            x=x,
            y=y,
            active_window=self._get_active_window_title(),
        )
        self._add_event(event)

    def _on_mouse_click(self, x: int, y: int, button: mouse.Button, pressed: bool):
        """Handle mouse click event."""
        if not self._record_mouse_clicks or self._state != InputRecorderState.RECORDING:
            return

        # Convert button to string
        button_name = button.name if hasattr(button, 'name') else str(button)

        # Get timestamp first
        timestamp = self._get_timestamp()

        # Capture screenshot on mouse press (not release)
        screenshot_path = None
        if pressed and self._on_click_screenshot:
            try:
                screenshot_path = self._on_click_screenshot(timestamp, x, y)
            except Exception as e:
                logger.error(f"Click screenshot callback failed: {e}")

        event = InputEvent(
            timestamp=timestamp,
            event_type=InputEventType.MOUSE_CLICK,
            x=x,
            y=y,
            button=button_name,
            pressed=pressed,
            active_window=self._get_active_window_title(),
            screenshot_path=screenshot_path,
        )
        self._add_event(event)
        logger.debug(f"Mouse {'click' if pressed else 'release'}: ({x}, {y}) {button_name}")

    def _on_mouse_scroll(self, x: int, y: int, dx: int, dy: int):
        """Handle mouse scroll event."""
        if not self._record_mouse_scroll or self._state != InputRecorderState.RECORDING:
            return

        event = InputEvent(
            timestamp=self._get_timestamp(),
            event_type=InputEventType.MOUSE_SCROLL,
            x=x,
            y=y,
            scroll_dx=dx,
            scroll_dy=dy,
            active_window=self._get_active_window_title(),
        )
        self._add_event(event)

    def _on_key_press(self, key):
        """Handle key press event."""
        if not self._record_keyboard or self._state != InputRecorderState.RECORDING:
            return

        key_name, key_code = self._parse_key(key)

        event = InputEvent(
            timestamp=self._get_timestamp(),
            event_type=InputEventType.KEY_PRESS,
            key=key_name,
            key_code=key_code,
            active_window=self._get_active_window_title(),
        )
        self._add_event(event)
        logger.debug(f"Key press: {key_name}")

    def _on_key_release(self, key):
        """Handle key release event."""
        if not self._record_keyboard or self._state != InputRecorderState.RECORDING:
            return

        key_name, key_code = self._parse_key(key)

        event = InputEvent(
            timestamp=self._get_timestamp(),
            event_type=InputEventType.KEY_RELEASE,
            key=key_name,
            key_code=key_code,
            active_window=self._get_active_window_title(),
        )
        self._add_event(event)

    def _parse_key(self, key) -> tuple[str, Optional[int]]:
        """Parse pynput key to name and code."""
        key_code = None

        if hasattr(key, 'char') and key.char:
            # Regular character key
            key_name = key.char
        elif hasattr(key, 'name'):
            # Special key (shift, ctrl, etc.)
            key_name = key.name
        else:
            key_name = str(key)

        # Try to get virtual key code
        if hasattr(key, 'vk'):
            key_code = key.vk
        elif hasattr(key, 'value') and hasattr(key.value, 'vk'):
            key_code = key.value.vk

        return key_name, key_code

    def _write_buffer_thread(self):
        """Background thread to periodically write events to disk."""
        while not self._stop_write_thread:
            time.sleep(1.0)  # Check every second

            current_time = time.time()
            if current_time - self._last_write_time >= self._write_interval:
                self._flush_buffer()

    def _flush_buffer(self):
        """Write buffered events to disk."""
        if not self._output_path:
            return

        with self._lock:
            if not self._write_buffer:
                return
            events_to_write = self._write_buffer.copy()
            self._write_buffer.clear()

        self._last_write_time = time.time()

        try:
            # Read existing file if it exists
            existing_events = []
            if self._output_path.exists():
                try:
                    with open(self._output_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        existing_events = data.get('events', [])
                except (json.JSONDecodeError, KeyError):
                    pass

            # Append new events
            all_events = existing_events + [e.to_dict() for e in events_to_write]

            # Write back
            output_data = {
                "version": "1.0",
                "total_events": len(all_events),
                "events": all_events,
            }

            with open(self._output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Flushed {len(events_to_write)} input events to disk")

        except Exception as e:
            logger.error(f"Failed to write input events: {e}")
            # Put events back in buffer
            with self._lock:
                self._write_buffer = events_to_write + self._write_buffer

    def start(self, output_path: Optional[Path] = None):
        """Start recording input events.

        Args:
            output_path: Path to save events JSON file
        """
        if not HAS_PYNPUT:
            logger.error("pynput not available, cannot record input")
            return False

        if self._state != InputRecorderState.IDLE:
            logger.warning("Input recorder already running")
            return False

        self._output_path = output_path
        self._events.clear()
        self._write_buffer.clear()
        self._start_time = time.time()
        self._total_paused = 0.0
        self._last_write_time = time.time()

        # Initialize output file
        if self._output_path:
            try:
                output_data = {
                    "version": "1.0",
                    "total_events": 0,
                    "events": [],
                }
                with open(self._output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Failed to initialize events file: {e}")

        # Start keyboard listener
        if self._record_keyboard:
            self._keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release,
            )
            self._keyboard_listener.start()

        # Start mouse listener
        if self._record_mouse_moves or self._record_mouse_clicks or self._record_mouse_scroll:
            self._mouse_listener = mouse.Listener(
                on_move=self._on_mouse_move if self._record_mouse_moves else None,
                on_click=self._on_mouse_click if self._record_mouse_clicks else None,
                on_scroll=self._on_mouse_scroll if self._record_mouse_scroll else None,
            )
            self._mouse_listener.start()

        # Start write thread
        self._stop_write_thread = False
        self._write_thread = threading.Thread(target=self._write_buffer_thread, daemon=True)
        self._write_thread.start()

        self._state = InputRecorderState.RECORDING
        logger.info("Input recording started")
        return True

    def pause(self):
        """Pause recording."""
        if self._state != InputRecorderState.RECORDING:
            return

        self._pause_time = time.time()
        self._state = InputRecorderState.PAUSED
        logger.info("Input recording paused")

    def resume(self):
        """Resume recording."""
        if self._state != InputRecorderState.PAUSED:
            return

        if self._pause_time:
            self._total_paused += time.time() - self._pause_time
            self._pause_time = None

        self._state = InputRecorderState.RECORDING
        logger.info("Input recording resumed")

    def stop(self) -> List[InputEvent]:
        """Stop recording and return all events.

        Returns:
            List of all recorded events
        """
        if self._state == InputRecorderState.IDLE:
            return []

        self._state = InputRecorderState.IDLE

        # Stop listeners
        if self._keyboard_listener:
            self._keyboard_listener.stop()
            self._keyboard_listener = None

        if self._mouse_listener:
            self._mouse_listener.stop()
            self._mouse_listener = None

        # Stop write thread
        self._stop_write_thread = True
        if self._write_thread and self._write_thread.is_alive():
            self._write_thread.join(timeout=2.0)
        self._write_thread = None

        # Final flush
        self._flush_buffer()

        events = self._events.copy()
        logger.info(f"Input recording stopped with {len(events)} events")

        return events

    def get_events(self) -> List[InputEvent]:
        """Get all recorded events so far."""
        with self._lock:
            return self._events.copy()


# Singleton
_input_recorder_instance: Optional[InputRecorder] = None


def get_input_recorder(
    record_mouse_moves: bool = False,
    record_mouse_clicks: bool = True,
    record_mouse_scroll: bool = True,
    record_keyboard: bool = True,
) -> InputRecorder:
    """Get the singleton input recorder.

    Args:
        record_mouse_moves: Record mouse movement (generates lots of data)
        record_mouse_clicks: Record mouse clicks
        record_mouse_scroll: Record mouse scroll
        record_keyboard: Record keyboard events
    """
    global _input_recorder_instance
    if _input_recorder_instance is None:
        _input_recorder_instance = InputRecorder(
            record_mouse_moves=record_mouse_moves,
            record_mouse_clicks=record_mouse_clicks,
            record_mouse_scroll=record_mouse_scroll,
            record_keyboard=record_keyboard,
        )
    return _input_recorder_instance


def is_input_recording_available() -> bool:
    """Check if input recording is available (pynput installed)."""
    return HAS_PYNPUT
