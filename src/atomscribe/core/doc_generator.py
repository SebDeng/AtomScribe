"""Document generator - orchestrates the document generation workflow.

This is the main controller that coordinates:
1. Transcript analysis (key point extraction)
2. Video frame extraction
3. VLM smart cropping and change detection
4. Markdown document generation
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from .session import Session
from .transcript_analyzer import (
    TranscriptAnalyzer,
    KeyPoint,
    KeyPointType,
    create_transcript_analyzer,
)
from .frame_extractor import FrameExtractor, create_frame_extractor, ExtractedFrame
from .vlm_processor import VLMProcessor, get_vlm_processor, CropRegion
from .markdown_writer import MarkdownWriter, ImageReference, create_markdown_writer
from .config import get_config_manager

from ..signals import get_app_signals


class GenerationMode(str, Enum):
    """Document generation mode."""
    TRAINING = "training"  # Step-by-step tutorial with screenshots
    EXPERIMENT_LOG = "experiment_log"  # Findings-focused log


@dataclass
class GenerationProgress:
    """Progress information for document generation."""
    phase: str
    current: int
    total: int
    description: str


class DocumentGenerator:
    """
    Main controller for document generation.

    Orchestrates the full pipeline:
    Session → Transcript Analysis → Frame Extraction → VLM Processing → Markdown
    """

    def __init__(
        self,
        use_vlm: bool = True,
        vlm_server_url: Optional[str] = None,
    ):
        """Initialize document generator.

        Args:
            use_vlm: Whether to use VLM for smart cropping
            vlm_server_url: URL of VLM server (uses config if None)
        """
        config = get_config_manager().config

        self.use_vlm = use_vlm
        self.vlm_server_url = vlm_server_url or config.vlm_server_url

        self._signals = get_app_signals()
        self._cancelled = False
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Components (created during generation)
        self._transcript_analyzer: Optional[TranscriptAnalyzer] = None
        self._frame_extractor: Optional[FrameExtractor] = None
        self._vlm_processor: Optional[VLMProcessor] = None
        self._markdown_writer: Optional[MarkdownWriter] = None

        # Progress callback
        self._progress_callback: Optional[Callable[[int, int, str], None]] = None

    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set progress callback.

        Args:
            callback: Function(current, total, description)
        """
        self._progress_callback = callback

    def _emit_progress(self, current: int, total: int, description: str):
        """Emit progress update."""
        if self._progress_callback:
            self._progress_callback(current, total, description)
        self._signals.doc_generation_progress.emit(current, total, description)

    def generate_async(
        self,
        session: Session,
        mode: GenerationMode = GenerationMode.TRAINING,
    ):
        """Start document generation in background thread.

        Args:
            session: Session to generate document for
            mode: Generation mode (training or experiment_log)
        """
        if self._running:
            logger.warning("Document generation already in progress")
            return

        self._cancelled = False
        self._running = True

        self._worker_thread = threading.Thread(
            target=self._generation_worker,
            args=(session, mode),
            daemon=True,
        )
        self._worker_thread.start()

    def _generation_worker(self, session: Session, mode: GenerationMode):
        """Worker thread for document generation."""
        try:
            self._signals.doc_generation_started.emit(mode.value)
            result_path = self.generate(session, mode)

            if result_path:
                self._signals.doc_generation_completed.emit(str(result_path))
            elif self._cancelled:
                self._signals.doc_generation_cancelled.emit()
            else:
                self._signals.doc_generation_error.emit("Unknown error during generation")

        except Exception as e:
            logger.error(f"Document generation failed: {e}")
            import traceback
            traceback.print_exc()
            self._signals.doc_generation_error.emit(str(e))

        finally:
            self._running = False

    def cancel(self):
        """Cancel ongoing generation."""
        self._cancelled = True
        logger.info("Document generation cancellation requested")

    def generate(
        self,
        session: Session,
        mode: GenerationMode = GenerationMode.TRAINING,
    ) -> Optional[Path]:
        """Generate document for a session (blocking).

        Args:
            session: Session to generate document for
            mode: Generation mode

        Returns:
            Path to generated markdown file, or None if failed/cancelled
        """
        logger.info(f"Starting document generation for session: {session.metadata.name}")
        logger.info(f"Mode: {mode.value}")

        # Phase 1: Load and analyze transcript
        self._emit_progress(0, 100, "Loading transcript...")

        if self._cancelled:
            return None

        transcript_path = session.get_transcript_path()
        if not transcript_path.exists():
            logger.error(f"Transcript not found: {transcript_path}")
            self._signals.doc_generation_error.emit("Transcript file not found")
            return None

        # Create transcript analyzer (using existing LLM if available)
        self._transcript_analyzer = create_transcript_analyzer()

        # Try to share LLM model with the post-processor
        try:
            from .llm_processor import get_llm_processor
            llm_proc = get_llm_processor()
            if llm_proc.is_model_loaded() and llm_proc._model:
                self._transcript_analyzer.set_model(llm_proc._model)
                logger.info("Sharing LLM model with transcript analyzer")
        except Exception as e:
            logger.debug(f"Could not share LLM model: {e}")

        transcript_data = self._transcript_analyzer.load_transcript(transcript_path)
        if not transcript_data:
            logger.error("Failed to load transcript")
            self._signals.doc_generation_error.emit("Failed to load transcript")
            return None

        self._emit_progress(10, 100, "Analyzing transcript...")

        if self._cancelled:
            return None

        # Analyze transcript to extract key points
        key_points = self._transcript_analyzer.analyze_with_llm(
            transcript_data,
            progress_callback=lambda c, t, d: self._emit_progress(
                10 + int((c / t) * 20), 100, d
            ),
        )

        if self._cancelled:
            return None

        if not key_points:
            logger.warning("No key points extracted from transcript")
            # Continue anyway, generate minimal document

        logger.info(f"Extracted {len(key_points)} key points")

        # Phase 2: Load input events for mouse positions
        self._emit_progress(30, 100, "Loading input events...")

        events = self._load_events(session.get_events_path())
        click_events = self._extract_click_events(events)
        logger.info(f"Found {len(click_events)} click events")

        # Phase 3: Setup frame extractor
        self._emit_progress(35, 100, "Preparing frame extraction...")

        video_path = session.get_video_path()
        if video_path.exists():
            self._frame_extractor = create_frame_extractor(video_path)
            if self._frame_extractor and self._frame_extractor.is_available:
                logger.info(f"Frame extractor ready for {video_path}")
            else:
                logger.warning("Frame extractor not available")
                self._frame_extractor = None
        else:
            logger.warning(f"Video file not found: {video_path}")
            self._frame_extractor = None

        # Phase 4: Setup VLM processor (will auto-start llama-server if needed)
        if self.use_vlm:
            self._emit_progress(38, 100, "Starting VLM server (this may take a minute)...")
            self._vlm_processor = get_vlm_processor(self.vlm_server_url)
            # ensure_available() will auto-start the server if model files are present
            if not self._vlm_processor.ensure_available():
                logger.warning("VLM server not available, using fallback cropping")
                self._vlm_processor = None
            else:
                logger.info("VLM server is available")

        # Phase 5: Process key points with frames and VLM
        self._emit_progress(40, 100, "Processing key points...")

        images_dict = {}
        frames_dir = session.directory / "frames"
        frames_dir.mkdir(exist_ok=True)

        total_kp = len(key_points)
        for idx, kp in enumerate(key_points):
            if self._cancelled:
                return None

            progress = 40 + int((idx / max(total_kp, 1)) * 40)
            self._emit_progress(progress, 100, f"Processing key point {idx + 1}/{total_kp}")

            # Get mouse position near this timestamp
            mouse_pos = self._find_nearest_click(click_events, kp.start_time)
            if mouse_pos:
                kp.mouse_position = mouse_pos

            # Process frames for this key point
            img_refs = self._process_key_point_frames(
                kp, idx, frames_dir, mouse_pos
            )
            if img_refs:
                images_dict[idx] = img_refs

        if self._cancelled:
            return None

        # Phase 6: Generate markdown document
        self._emit_progress(85, 100, "Generating markdown document...")

        self._markdown_writer = create_markdown_writer(session.directory)

        title = session.metadata.name or "Recording Session"
        duration = transcript_data.total_duration or session.metadata.duration_seconds

        if mode == GenerationMode.TRAINING:
            content = self._markdown_writer.generate_training_document(
                key_points=key_points,
                title=title,
                duration=duration,
                images=images_dict,
            )
        else:
            content = self._markdown_writer.generate_experiment_log(
                key_points=key_points,
                title=title,
                duration=duration,
                images=images_dict,
            )

        self._emit_progress(95, 100, "Writing document...")

        output_path = self._markdown_writer.write_document(content)

        # Update session metadata
        session.metadata.summary_file = str(output_path)
        session.save_metadata()

        self._emit_progress(100, 100, "Document generation complete!")

        logger.info(f"Document generated: {output_path}")
        return output_path

    def _load_events(self, events_path: Path) -> List[dict]:
        """Load input events from JSON file.

        Args:
            events_path: Path to events.json

        Returns:
            List of event dictionaries
        """
        if not events_path.exists():
            return []

        try:
            with open(events_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else data.get("events", [])
        except Exception as e:
            logger.error(f"Failed to load events: {e}")
            return []

    def _extract_click_events(
        self,
        events: List[dict],
    ) -> List[Tuple[float, int, int]]:
        """Extract mouse click events with timestamps and positions.

        Args:
            events: List of input events

        Returns:
            List of (timestamp, x, y) tuples for clicks
        """
        clicks = []
        for event in events:
            if event.get("event_type") == "mouse_click":
                x = event.get("x")
                y = event.get("y")
                ts = event.get("timestamp", 0)
                pressed = event.get("pressed", True)

                if x is not None and y is not None and pressed:
                    clicks.append((ts, x, y))

        return sorted(clicks, key=lambda c: c[0])

    def _find_nearest_click(
        self,
        click_events: List[Tuple[float, int, int]],
        timestamp: float,
        max_distance: float = 3.0,
    ) -> Optional[Tuple[int, int]]:
        """Find the click event nearest to a timestamp.

        Args:
            click_events: List of (timestamp, x, y) tuples
            timestamp: Target timestamp
            max_distance: Maximum time difference in seconds

        Returns:
            (x, y) position of nearest click, or None
        """
        if not click_events:
            return None

        nearest = None
        min_dist = float('inf')

        for ts, x, y in click_events:
            dist = abs(ts - timestamp)
            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                nearest = (x, y)

        return nearest

    def _process_key_point_frames(
        self,
        kp: KeyPoint,
        idx: int,
        frames_dir: Path,
        mouse_pos: Optional[Tuple[int, int]],
    ) -> dict:
        """Process frames for a key point.

        Args:
            kp: Key point to process
            idx: Key point index
            frames_dir: Directory to save frames
            mouse_pos: Mouse position at this timestamp

        Returns:
            Dict with image references: {"main": ..., "before": ..., "after": ...}
        """
        result = {}

        if not self._frame_extractor or not self._frame_extractor.is_available:
            return result

        cropped_dir = frames_dir / "cropped"
        cropped_dir.mkdir(exist_ok=True)

        # Determine what frames to extract
        if kp.needs_comparison:
            # Extract before/after frames
            before_offset = 1.0
            after_offset = 2.0

            before_frame, after_frame = self._frame_extractor.extract_frame_pair(
                timestamp=kp.start_time,
                before_offset=before_offset,
                after_offset=after_offset,
                output_dir=frames_dir,
            )

            if before_frame and after_frame:
                # Try VLM change detection
                if self._vlm_processor:
                    change_result = self._vlm_processor.detect_changes(
                        before_frame.image_path,
                        after_frame.image_path,
                        kp.text,
                    )

                    if change_result.changed_region and change_result.is_significant:
                        # Crop both images to changed region
                        before_cropped = cropped_dir / f"before_{idx:04d}.jpg"
                        after_cropped = cropped_dir / f"after_{idx:04d}.jpg"

                        self._vlm_processor.crop_image(
                            before_frame.image_path,
                            change_result.changed_region,
                            before_cropped,
                        )
                        self._vlm_processor.crop_image(
                            after_frame.image_path,
                            change_result.changed_region,
                            after_cropped,
                        )

                        result["before"] = self._create_image_ref(
                            before_cropped, "Before", f"Before step {idx + 1}"
                        )
                        result["after"] = self._create_image_ref(
                            after_cropped,
                            change_result.change_description or "After",
                            f"After step {idx + 1}",
                        )
                    else:
                        # Use full frames
                        result["before"] = self._create_image_ref(
                            before_frame.image_path, "Before", f"Before step {idx + 1}"
                        )
                        result["after"] = self._create_image_ref(
                            after_frame.image_path, "After", f"After step {idx + 1}"
                        )
                else:
                    # No VLM, use full frames
                    result["before"] = self._create_image_ref(
                        before_frame.image_path, "Before", f"Before step {idx + 1}"
                    )
                    result["after"] = self._create_image_ref(
                        after_frame.image_path, "After", f"After step {idx + 1}"
                    )

        elif kp.needs_screenshot:
            # Extract single frame with smart cropping
            frame = self._frame_extractor.extract_frame(
                kp.start_time,
                frames_dir / f"frame_{idx:04d}_{kp.start_time:.2f}.jpg",
            )

            if frame:
                if self._vlm_processor:
                    # Smart crop using VLM
                    crop_region = self._vlm_processor.smart_crop(
                        frame.image_path,
                        context=kp.text,
                        mouse_position=mouse_pos,
                    )

                    cropped_path = cropped_dir / f"crop_{idx:04d}.jpg"
                    if self._vlm_processor.crop_image(
                        frame.image_path, crop_region, cropped_path
                    ):
                        result["main"] = self._create_image_ref(
                            cropped_path,
                            crop_region.description or "",
                            f"Screenshot for step {idx + 1}",
                        )
                    else:
                        result["main"] = self._create_image_ref(
                            frame.image_path, "", f"Screenshot for step {idx + 1}"
                        )
                else:
                    result["main"] = self._create_image_ref(
                        frame.image_path, "", f"Screenshot for step {idx + 1}"
                    )

        return result

    def _create_image_ref(
        self,
        path: Path,
        caption: str,
        alt_text: str,
    ) -> ImageReference:
        """Create an image reference.

        Args:
            path: Path to image
            caption: Image caption
            alt_text: Alt text

        Returns:
            ImageReference object
        """
        # Calculate relative path from session directory
        try:
            rel_path = f"./frames/{path.parent.name}/{path.name}" if path.parent.name != "frames" else f"./frames/{path.name}"
        except Exception:
            rel_path = f"./{path.name}"

        return ImageReference(
            path=path,
            relative_path=rel_path,
            caption=caption,
            alt_text=alt_text,
        )

    @property
    def is_running(self) -> bool:
        """Check if generation is in progress."""
        return self._running


# Singleton instance
_generator_instance: Optional[DocumentGenerator] = None


def get_document_generator(
    use_vlm: bool = True,
    vlm_server_url: Optional[str] = None,
) -> DocumentGenerator:
    """Get the singleton document generator instance.

    Args:
        use_vlm: Whether to use VLM for smart cropping
        vlm_server_url: Optional VLM server URL

    Returns:
        DocumentGenerator instance
    """
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = DocumentGenerator(
            use_vlm=use_vlm,
            vlm_server_url=vlm_server_url,
        )
    return _generator_instance
