"""Frame extractor - extracts frames from video at specific timestamps.

Uses FFmpeg to extract individual frames from screen recordings
for use in document generation.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
from loguru import logger

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None


@dataclass
class ExtractedFrame:
    """Represents an extracted video frame."""
    timestamp: float  # Time in seconds
    image_path: Path  # Path to saved image
    width: int
    height: int


def find_ffmpeg() -> Optional[str]:
    """Find FFmpeg executable path."""
    import shutil
    import sys
    import os

    # Check if ffmpeg is in PATH
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    # Check common locations
    if sys.platform == "win32":
        common_paths = [
            Path(os.environ.get("CONDA_PREFIX", "")) / "Library" / "bin" / "ffmpeg.exe",
            Path(os.environ.get("USERPROFILE", "")) / "anaconda3" / "Library" / "bin" / "ffmpeg.exe",
            Path("C:/ffmpeg/bin/ffmpeg.exe"),
            Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
        ]
    else:
        common_paths = [
            Path("/usr/local/bin/ffmpeg"),
            Path("/opt/homebrew/bin/ffmpeg"),
        ]

    for path in common_paths:
        if path.exists():
            return str(path)

    return None


class FrameExtractor:
    """Extracts frames from video files using FFmpeg."""

    def __init__(self, video_path: Path):
        """Initialize frame extractor.

        Args:
            video_path: Path to the video file
        """
        self._video_path = video_path
        self._ffmpeg_path = find_ffmpeg()
        self._video_duration: Optional[float] = None
        self._video_size: Optional[Tuple[int, int]] = None

        if not self._ffmpeg_path:
            logger.warning("FFmpeg not found, frame extraction will not work")

        if video_path.exists():
            self._probe_video()

    def _probe_video(self):
        """Get video metadata using ffprobe."""
        if not self._ffmpeg_path:
            return

        ffprobe_path = self._ffmpeg_path.replace("ffmpeg", "ffprobe")

        try:
            # Get duration
            result = subprocess.run(
                [
                    ffprobe_path,
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(self._video_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                self._video_duration = float(result.stdout.strip())

            # Get video size
            result = subprocess.run(
                [
                    ffprobe_path,
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "csv=s=x:p=0",
                    str(self._video_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split("x")
                if len(parts) == 2:
                    self._video_size = (int(parts[0]), int(parts[1]))

            logger.debug(f"Video probe: duration={self._video_duration}s, size={self._video_size}")

        except Exception as e:
            logger.error(f"Failed to probe video: {e}")

    @property
    def is_available(self) -> bool:
        """Check if frame extraction is available."""
        return self._ffmpeg_path is not None and self._video_path.exists()

    @property
    def duration(self) -> Optional[float]:
        """Get video duration in seconds."""
        return self._video_duration

    @property
    def size(self) -> Optional[Tuple[int, int]]:
        """Get video dimensions (width, height)."""
        return self._video_size

    def extract_frame(
        self,
        timestamp: float,
        output_path: Optional[Path] = None,
        quality: int = 2,  # 1-31, lower is better
    ) -> Optional[ExtractedFrame]:
        """Extract a single frame at the specified timestamp.

        Args:
            timestamp: Time in seconds
            output_path: Where to save the frame (auto-generated if None)
            quality: JPEG quality (1-31, lower is better)

        Returns:
            ExtractedFrame object, or None if extraction failed
        """
        if not self.is_available:
            logger.error("Frame extraction not available")
            return None

        # Clamp timestamp to valid range
        if self._video_duration and timestamp > self._video_duration:
            timestamp = self._video_duration - 0.1

        if timestamp < 0:
            timestamp = 0

        # Generate output path if not provided
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=".jpg"))

        try:
            result = subprocess.run(
                [
                    self._ffmpeg_path,
                    "-ss", str(timestamp),
                    "-i", str(self._video_path),
                    "-vframes", "1",
                    "-q:v", str(quality),
                    "-y",  # Overwrite
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.error(f"FFmpeg failed: {result.stderr}")
                return None

            if not output_path.exists():
                logger.error("Output frame not created")
                return None

            # Get actual image dimensions
            width, height = self._video_size or (0, 0)
            if HAS_PIL:
                try:
                    with Image.open(output_path) as img:
                        width, height = img.size
                except Exception:
                    pass

            return ExtractedFrame(
                timestamp=timestamp,
                image_path=output_path,
                width=width,
                height=height,
            )

        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timed out")
            return None
        except Exception as e:
            logger.error(f"Failed to extract frame: {e}")
            return None

    def extract_frame_pair(
        self,
        timestamp: float,
        before_offset: float = 1.0,
        after_offset: float = 2.0,
        output_dir: Optional[Path] = None,
    ) -> Tuple[Optional[ExtractedFrame], Optional[ExtractedFrame]]:
        """Extract before and after frames around a timestamp.

        Args:
            timestamp: Center timestamp (usually a click event)
            before_offset: Seconds before timestamp for "before" frame
            after_offset: Seconds after timestamp for "after" frame
            output_dir: Directory to save frames (temp dir if None)

        Returns:
            Tuple of (before_frame, after_frame)
        """
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp())
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        before_time = max(0, timestamp - before_offset)
        after_time = timestamp + after_offset

        before_path = output_dir / f"before_{timestamp:.2f}.jpg"
        after_path = output_dir / f"after_{timestamp:.2f}.jpg"

        before_frame = self.extract_frame(before_time, before_path)
        after_frame = self.extract_frame(after_time, after_path)

        return before_frame, after_frame

    def extract_frames_batch(
        self,
        timestamps: List[float],
        output_dir: Path,
        prefix: str = "frame",
    ) -> List[Optional[ExtractedFrame]]:
        """Extract multiple frames efficiently.

        Args:
            timestamps: List of timestamps to extract
            output_dir: Directory to save frames
            prefix: Filename prefix

        Returns:
            List of ExtractedFrame objects (None for failed extractions)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for i, ts in enumerate(timestamps):
            output_path = output_dir / f"{prefix}_{i:04d}_{ts:.2f}.jpg"
            frame = self.extract_frame(ts, output_path)
            results.append(frame)

        return results


def create_frame_extractor(video_path: Path) -> Optional[FrameExtractor]:
    """Factory function to create a frame extractor.

    Args:
        video_path: Path to the video file

    Returns:
        FrameExtractor instance, or None if video doesn't exist
    """
    if not video_path.exists():
        logger.warning(f"Video file not found: {video_path}")
        return None

    return FrameExtractor(video_path)
