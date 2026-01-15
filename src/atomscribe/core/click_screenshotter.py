"""Click screenshotter - captures screenshots when mouse clicks occur.

This module captures screenshots immediately on mouse press to ensure
transient UI elements (dropdowns, menus, tooltips) are captured before
they disappear.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Callable, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from .capture_backend import CaptureBackend

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None


class ClickScreenshotter:
    """Captures screenshots on mouse click events.

    Uses ThreadPoolExecutor to capture and save screenshots asynchronously,
    ensuring the mouse event handler doesn't block.
    """

    def __init__(
        self,
        output_dir: Path,
        capture_backend: "CaptureBackend",
        quality: int = 85,
        max_workers: int = 2,
    ):
        """Initialize the click screenshotter.

        Args:
            output_dir: Directory to save click screenshots
            capture_backend: Capture backend from screen recorder
            quality: JPEG quality (1-100, default 85)
            max_workers: Max concurrent screenshot tasks
        """
        self._output_dir = Path(output_dir)
        self._capture_backend = capture_backend
        self._quality = quality

        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._click_count = 0
        self._pending_futures = []

        # Ensure output directory exists
        self._output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ClickScreenshotter initialized, saving to: {self._output_dir}")

    def capture(self, timestamp: float, x: int, y: int) -> Optional[str]:
        """Capture a screenshot for a click event.

        This method returns immediately with the expected filename.
        The actual capture happens asynchronously.

        Args:
            timestamp: Click timestamp (seconds since recording start)
            x: Click X position
            y: Click Y position

        Returns:
            Relative path to screenshot file (e.g., "clicks/click_001_26.500s.jpg"),
            or None if capture backend not available
        """
        if not self._capture_backend or not self._capture_backend.is_valid():
            logger.warning("Capture backend not available for click screenshot")
            return None

        with self._lock:
            self._click_count += 1
            seq = self._click_count

        # Generate filename
        filename = f"click_{seq:03d}_{timestamp:.3f}s.jpg"
        filepath = self._output_dir / filename
        relative_path = f"clicks/{filename}"

        # Submit async capture task
        future = self._executor.submit(
            self._capture_and_save,
            filepath,
            timestamp,
            x,
            y,
            seq
        )

        with self._lock:
            self._pending_futures.append(future)

        logger.debug(f"Scheduled click screenshot: {filename} at ({x}, {y})")
        return relative_path

    def _capture_and_save(
        self,
        filepath: Path,
        timestamp: float,
        x: int,
        y: int,
        seq: int
    ):
        """Capture and save screenshot (runs in thread pool).

        Args:
            filepath: Full path to save the screenshot
            timestamp: Click timestamp
            x: Click X position
            y: Click Y position
            seq: Sequence number
        """
        try:
            # Capture frame
            frame = self._capture_backend.capture_frame()

            if frame is None:
                logger.error(f"Failed to capture frame for click {seq}")
                return

            # Save as JPEG
            frame.save(filepath, "JPEG", quality=self._quality)
            logger.debug(f"Saved click screenshot: {filepath.name}")

        except Exception as e:
            logger.error(f"Failed to save click screenshot {seq}: {e}")

    def wait_for_pending(self, timeout: float = 10.0):
        """Wait for all pending screenshot captures to complete.

        Args:
            timeout: Maximum seconds to wait
        """
        with self._lock:
            futures = self._pending_futures.copy()

        if not futures:
            return

        logger.info(f"Waiting for {len(futures)} pending click screenshots...")

        for future in futures:
            try:
                future.result(timeout=timeout)
            except Exception as e:
                logger.error(f"Error waiting for screenshot: {e}")

        with self._lock:
            self._pending_futures.clear()

        logger.info("All click screenshots completed")

    def shutdown(self):
        """Shutdown the executor and wait for pending tasks."""
        self.wait_for_pending()
        self._executor.shutdown(wait=True)
        logger.info(f"ClickScreenshotter shutdown, captured {self._click_count} screenshots")

    @property
    def click_count(self) -> int:
        """Get the number of screenshots captured."""
        with self._lock:
            return self._click_count
