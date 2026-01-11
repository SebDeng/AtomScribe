"""Screen recording service for AtomScribe"""

import subprocess
import threading
import time
import shutil
from pathlib import Path
from typing import Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
from loguru import logger

try:
    import mss
    import mss.tools
except ImportError:
    mss = None
    logger.warning("mss not installed. Screen recording will not work.")

try:
    import numpy as np
except ImportError:
    np = None


class ScreenRecordingState(Enum):
    """Screen recording state enumeration"""
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPING = "stopping"


@dataclass
class MonitorInfo:
    """Monitor information"""
    index: int
    name: str
    width: int
    height: int
    left: int
    top: int
    is_primary: bool = False


class ScreenRecorder:
    """
    Screen recorder that captures screen and saves to video file.
    Uses mss for capture and FFmpeg for encoding.
    """

    def __init__(
        self,
        fps: int = 10,
        monitor_index: int = 0,
        quality: int = 23,
        codec: str = "libx264",
    ):
        """
        Initialize the screen recorder.

        Args:
            fps: Frames per second (10 recommended for UI recording)
            monitor_index: Monitor to record (0 = primary, -1 = all monitors)
            quality: FFmpeg CRF value (0-51, lower = better, 23 = default)
            codec: Video codec (libx264 or libx265)
        """
        self.fps = fps
        self.monitor_index = monitor_index
        self.quality = quality
        self.codec = codec

        self._state = ScreenRecordingState.IDLE
        self._output_path: Optional[Path] = None
        self._ffmpeg_process: Optional[subprocess.Popen] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()  # For clean thread shutdown

        # Callbacks
        self._on_error_callback: Optional[Callable[[str], None]] = None

        # Check FFmpeg availability
        self._ffmpeg_path = self._find_ffmpeg()

    def _find_ffmpeg(self) -> Optional[str]:
        """Find FFmpeg executable"""
        import sys
        import os

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            logger.debug(f"Found FFmpeg at: {ffmpeg}")
            return ffmpeg

        # Build list of common locations
        common_paths = [
            "C:/ffmpeg/bin/ffmpeg.exe",
            "C:/Program Files/ffmpeg/bin/ffmpeg.exe",
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
        ]

        # Add conda/python environment paths
        # This handles cases where PATH isn't set properly in GUI mode
        python_dir = Path(sys.executable).parent
        conda_paths = [
            python_dir / "ffmpeg.exe",  # Same dir as python
            python_dir / "Library" / "bin" / "ffmpeg.exe",  # Conda on Windows
            python_dir.parent / "Library" / "bin" / "ffmpeg.exe",  # Conda env root
        ]

        # For conda envs, also check base anaconda installation
        # Path: envs/EnvName/python.exe -> go up to anaconda3/Library/bin
        if "envs" in str(python_dir):
            # Go from envs/EnvName to anaconda3
            base_anaconda = python_dir.parent.parent  # anaconda3 or miniconda3
            conda_paths.append(base_anaconda / "Library" / "bin" / "ffmpeg.exe")

        common_paths.extend([str(p) for p in conda_paths])

        # Also check CONDA_PREFIX if set
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            common_paths.append(f"{conda_prefix}/Library/bin/ffmpeg.exe")
            common_paths.append(f"{conda_prefix}/bin/ffmpeg")

        # Check user's home directory for common install locations
        home = Path.home()
        user_paths = [
            home / "anaconda3" / "Library" / "bin" / "ffmpeg.exe",
            home / "miniconda3" / "Library" / "bin" / "ffmpeg.exe",
            home / "AppData" / "Local" / "Programs" / "ffmpeg" / "bin" / "ffmpeg.exe",
        ]
        common_paths.extend([str(p) for p in user_paths])

        for path in common_paths:
            if Path(path).exists():
                logger.debug(f"Found FFmpeg at: {path}")
                return path

        logger.warning("FFmpeg not found. Screen recording will not work.")
        return None

    @property
    def state(self) -> ScreenRecordingState:
        """Get current recording state"""
        return self._state

    @property
    def is_recording(self) -> bool:
        """Check if currently recording"""
        return self._state == ScreenRecordingState.RECORDING

    @property
    def is_paused(self) -> bool:
        """Check if recording is paused"""
        return self._state == ScreenRecordingState.PAUSED

    def set_on_error_callback(self, callback: Callable[[str], None]):
        """Set callback for error notifications"""
        self._on_error_callback = callback

    def set_monitor(self, monitor_index: int):
        """Set the monitor to record"""
        self.monitor_index = monitor_index
        logger.info(f"Set screen recording monitor to index {monitor_index}")

    def set_fps(self, fps: int):
        """Set the frame rate"""
        self.fps = max(1, min(60, fps))
        logger.info(f"Set screen recording FPS to {self.fps}")

    def set_quality(self, quality: int):
        """Set the quality (CRF value)"""
        self.quality = max(0, min(51, quality))
        logger.info(f"Set screen recording quality (CRF) to {self.quality}")

    @staticmethod
    def get_monitors() -> List[MonitorInfo]:
        """Get list of available monitors"""
        if mss is None:
            return []

        monitors = []
        try:
            with mss.mss() as sct:
                # Skip monitor 0 as it's the virtual "all monitors" monitor
                for i, mon in enumerate(sct.monitors):
                    if i == 0:
                        # All monitors combined
                        monitors.append(MonitorInfo(
                            index=-1,
                            name="All Monitors",
                            width=mon["width"],
                            height=mon["height"],
                            left=mon["left"],
                            top=mon["top"],
                            is_primary=False,
                        ))
                    else:
                        monitors.append(MonitorInfo(
                            index=i - 1,  # Convert to 0-based for non-combined
                            name=f"Monitor {i}",
                            width=mon["width"],
                            height=mon["height"],
                            left=mon["left"],
                            top=mon["top"],
                            is_primary=(i == 1),  # First actual monitor is usually primary
                        ))
        except Exception as e:
            logger.error(f"Error getting monitors: {e}")

        return monitors

    def _get_monitor_region(self) -> dict:
        """Get the region to capture based on monitor selection"""
        if mss is None:
            raise RuntimeError("mss not installed")

        with mss.mss() as sct:
            if self.monitor_index == -1:
                # All monitors
                return sct.monitors[0]
            else:
                # Specific monitor (1-based in mss)
                mss_index = self.monitor_index + 1
                if mss_index < len(sct.monitors):
                    return sct.monitors[mss_index]
                else:
                    logger.warning(f"Monitor {self.monitor_index} not found, using primary")
                    return sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]

    def start_recording(self, output_path: Path):
        """
        Start recording screen to file.

        Args:
            output_path: Output video file path (should be .mp4)
        """
        if mss is None:
            raise RuntimeError("mss not installed")

        if self._ffmpeg_path is None:
            raise RuntimeError("FFmpeg not found")

        if self._state != ScreenRecordingState.IDLE:
            logger.warning("Screen recording already in progress")
            return

        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get monitor region
        monitor = self._get_monitor_region()
        width = monitor["width"]
        height = monitor["height"]

        # Ensure dimensions are even (required by many codecs)
        width = width - (width % 2)
        height = height - (height % 2)

        logger.info(f"Starting screen recording: {width}x{height} @ {self.fps}fps")

        # Build FFmpeg command
        ffmpeg_cmd = [
            self._ffmpeg_path,
            "-y",  # Overwrite output
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgra",  # mss uses BGRA format
            "-s", f"{width}x{height}",
            "-r", str(self.fps),
            "-i", "-",  # Read from stdin
            "-c:v", self.codec,
            "-pix_fmt", "yuv420p",
            "-crf", str(self.quality),
            "-preset", "ultrafast",  # Fast encoding for real-time
            str(self._output_path),
        ]

        try:
            # Start FFmpeg process
            # Use DEVNULL for stderr to prevent buffer deadlock on Windows
            # (if stderr buffer fills up, FFmpeg blocks and wait() hangs)
            self._ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Start capture thread
            self._stop_event.clear()  # Reset stop event
            self._state = ScreenRecordingState.RECORDING
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                args=(monitor, width, height),
                daemon=True,
            )
            self._capture_thread.start()

            logger.info(f"Screen recording started: {self._output_path}")

        except Exception as e:
            logger.error(f"Failed to start screen recording: {e}")
            self._state = ScreenRecordingState.IDLE
            if self._on_error_callback:
                self._on_error_callback(str(e))
            raise

    def _capture_loop(self, monitor: dict, width: int, height: int):
        """Main capture loop - runs in separate thread"""
        frame_interval = 1.0 / self.fps

        try:
            with mss.mss() as sct:
                while not self._stop_event.is_set() and self._ffmpeg_process is not None:
                    current_time = time.time()

                    if self._state == ScreenRecordingState.RECORDING:
                        # Capture frame
                        try:
                            screenshot = sct.grab(monitor)

                            # Convert to raw bytes (BGRA format)
                            frame_data = screenshot.raw

                            # Resize if needed to match expected dimensions
                            if screenshot.width != width or screenshot.height != height:
                                # Skip this frame if dimensions don't match
                                logger.warning(f"Frame size mismatch: {screenshot.width}x{screenshot.height} vs {width}x{height}")
                            else:
                                # Write to FFmpeg - check stop conditions first
                                ffmpeg_proc = self._ffmpeg_process
                                if ffmpeg_proc and ffmpeg_proc.stdin and not self._stop_event.is_set():
                                    try:
                                        ffmpeg_proc.stdin.write(frame_data)
                                    except (BrokenPipeError, OSError, ValueError) as e:
                                        # ValueError can occur if stdin is closed
                                        if not self._stop_event.is_set():
                                            logger.debug(f"FFmpeg pipe closed: {e}")
                                        break
                                else:
                                    # FFmpeg process gone or stopping
                                    break
                        except Exception as e:
                            if not self._stop_event.is_set():
                                logger.error(f"Error capturing frame: {e}")

                    # Calculate sleep time to maintain FPS
                    elapsed = time.time() - current_time
                    sleep_time = frame_interval - elapsed
                    if sleep_time > 0:
                        # Use event wait for interruptible sleep
                        self._stop_event.wait(timeout=sleep_time)

        except Exception as e:
            if not self._stop_event.is_set():
                logger.error(f"Error in capture loop: {e}")
                if self._on_error_callback:
                    self._on_error_callback(str(e))

        logger.debug("Capture loop ended")

    def pause_recording(self):
        """Pause the recording (frames will not be captured)"""
        if self._state == ScreenRecordingState.RECORDING:
            self._state = ScreenRecordingState.PAUSED
            logger.info("Screen recording paused")

    def resume_recording(self):
        """Resume the recording"""
        if self._state == ScreenRecordingState.PAUSED:
            self._state = ScreenRecordingState.RECORDING
            logger.info("Screen recording resumed")

    def stop_recording(self) -> Optional[Path]:
        """
        Stop recording and finalize the file.

        Returns:
            Path to the output video file, or None if failed
        """
        if self._state == ScreenRecordingState.IDLE:
            logger.warning("No screen recording in progress")
            return None

        logger.info("Stopping screen recording...")
        self._state = ScreenRecordingState.STOPPING

        # Signal the capture thread to stop
        self._stop_event.set()

        # Close FFmpeg stdin FIRST to unblock any pending writes in capture thread
        ffmpeg_proc = self._ffmpeg_process
        self._ffmpeg_process = None  # Clear reference to signal capture thread

        if ffmpeg_proc and ffmpeg_proc.stdin:
            try:
                ffmpeg_proc.stdin.close()
                logger.debug("Closed FFmpeg stdin to unblock capture thread")
            except Exception as e:
                logger.debug(f"Error closing FFmpeg stdin early: {e}")

        # Now wait for capture thread to finish (should exit quickly now)
        if self._capture_thread and self._capture_thread.is_alive():
            logger.debug("Waiting for capture thread to finish...")
            self._capture_thread.join(timeout=2.0)
            if self._capture_thread.is_alive():
                logger.warning("Capture thread did not finish in time, continuing anyway")

        if ffmpeg_proc:
            try:
                # Wait for FFmpeg to finish encoding (stdin already closed above)
                logger.debug("Waiting for FFmpeg to finish encoding...")
                try:
                    ffmpeg_proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    logger.warning("FFmpeg did not finish in time, terminating...")
                    ffmpeg_proc.terminate()
                    try:
                        ffmpeg_proc.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        logger.warning("FFmpeg terminate timeout, killing...")
                        ffmpeg_proc.kill()

                # Check for errors
                if ffmpeg_proc.returncode and ffmpeg_proc.returncode != 0:
                    logger.error(f"FFmpeg exited with error code {ffmpeg_proc.returncode}")

            except Exception as e:
                logger.error(f"Error closing FFmpeg: {e}")
                # Force kill if anything goes wrong
                try:
                    ffmpeg_proc.kill()
                except Exception:
                    pass

        self._state = ScreenRecordingState.IDLE
        self._capture_thread = None

        # Verify output file exists
        if self._output_path and self._output_path.exists():
            file_size = self._output_path.stat().st_size
            logger.info(f"Screen recording saved: {self._output_path} ({file_size} bytes)")
            return self._output_path
        else:
            logger.error("Screen recording file not created")
            return None

    def is_available(self) -> bool:
        """Check if screen recording is available (dependencies installed)"""
        return mss is not None and self._ffmpeg_path is not None


# Singleton instance
_recorder_instance: Optional[ScreenRecorder] = None


def get_screen_recorder(
    fps: int = 10,
    monitor_index: int = 0,
    quality: int = 23,
    codec: str = "libx264",
) -> ScreenRecorder:
    """Get the singleton screen recorder instance"""
    global _recorder_instance
    if _recorder_instance is None:
        _recorder_instance = ScreenRecorder(
            fps=fps,
            monitor_index=monitor_index,
            quality=quality,
            codec=codec,
        )
    return _recorder_instance
