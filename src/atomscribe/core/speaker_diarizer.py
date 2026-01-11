"""Speaker diarization using SpeechBrain ECAPA-TDNN + Agglomerative Clustering

Supports sub-segment diarization to handle multiple speakers within a single
transcript segment.
"""

import os
import threading
import time
import numpy as np
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, Callable, Dict, List, Tuple
from dataclasses import dataclass, field
from loguru import logger

# Set SpeechBrain to use COPY strategy instead of SYMLINK (Windows compatibility)
os.environ.setdefault("SPEECHBRAIN_LOCAL_STRATEGY", "copy")

# Deferred imports for optional dependencies
TORCH_AVAILABLE = False
SPEECHBRAIN_AVAILABLE = False
SKLEARN_AVAILABLE = False

torch = None
EncoderClassifier = None
AgglomerativeClustering = None

try:
    import torch as _torch
    torch = _torch
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not installed. Speaker diarization will not work.")
except Exception as e:
    logger.warning(f"Failed to import torch: {e}")

try:
    # Patch torchaudio for compatibility with SpeechBrain
    # torchaudio 2.1+ removed list_audio_backends(), but SpeechBrain may still use it
    import torchaudio
    if not hasattr(torchaudio, 'list_audio_backends'):
        # Provide a stub function that returns available backends
        def _list_audio_backends():
            """Stub for removed torchaudio.list_audio_backends()"""
            try:
                # Try to get backends from the new API
                backends = []
                if hasattr(torchaudio, '_backend') and hasattr(torchaudio._backend, 'get_audio_backend'):
                    backends.append(torchaudio._backend.get_audio_backend())
                return backends if backends else ['soundfile']
            except Exception:
                return ['soundfile']
        torchaudio.list_audio_backends = _list_audio_backends
        logger.debug("Patched torchaudio.list_audio_backends for SpeechBrain compatibility")

    from speechbrain.inference.speaker import EncoderClassifier as _EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    EncoderClassifier = _EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    logger.warning("SpeechBrain not installed. Speaker diarization will not work.")
    LocalStrategy = None
except Exception as e:
    logger.warning(f"Failed to import SpeechBrain: {e}. Speaker diarization disabled.")
    LocalStrategy = None

try:
    from sklearn.cluster import AgglomerativeClustering as _AgglomerativeClustering
    AgglomerativeClustering = _AgglomerativeClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not installed. Speaker clustering will not work.")
except Exception as e:
    logger.warning(f"Failed to import sklearn: {e}")


@dataclass
class SpeakerSegment:
    """A sub-segment with speaker assignment within a transcript segment"""
    start_time: float  # Absolute start time in seconds
    end_time: float    # Absolute end time in seconds
    speaker_label: str # "Speaker A", "Speaker B", etc.


@dataclass
class SpeakerResult:
    """Result of speaker identification for a segment"""
    segment_id: int
    speakers: List[SpeakerSegment] = field(default_factory=list)  # Sub-segment speakers
    primary_speaker: str = ""  # Most frequent speaker in the segment

    @property
    def has_multiple_speakers(self) -> bool:
        """Check if segment has multiple different speakers"""
        if len(self.speakers) <= 1:
            return False
        unique_speakers = set(s.speaker_label for s in self.speakers)
        return len(unique_speakers) > 1

    def get_speaker_sequence(self) -> str:
        """Get speaker sequence string like 'A → B → A'"""
        if not self.speakers:
            return ""

        # Merge consecutive same speakers
        merged = []
        for seg in self.speakers:
            label_short = seg.speaker_label.replace("Speaker ", "")
            if not merged or merged[-1] != label_short:
                merged.append(label_short)

        return " → ".join(merged)


class SpeakerDiarizer:
    """
    Speaker diarization using ECAPA-TDNN embeddings + clustering.

    Supports sub-segment diarization: splits long segments into windows
    to detect speaker changes within a single transcript segment.
    """

    # Model source - can be downloaded without HuggingFace token
    DEFAULT_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

    # Local model path for offline use
    LOCAL_MODEL_DIR = Path(__file__).parent.parent.parent.parent / "models" / "ecapa_tdnn"

    # Speaker labels
    SPEAKER_LABELS = ["Speaker A", "Speaker B", "Speaker C", "Speaker D",
                      "Speaker E", "Speaker F", "Speaker G", "Speaker H"]

    # Sub-segment window settings
    WINDOW_DURATION = 1.5  # Window size in seconds
    WINDOW_STEP = 1.0      # Step size (overlap = WINDOW_DURATION - WINDOW_STEP)
    MIN_WINDOW_DURATION = 0.5  # Minimum window duration for embedding

    # Minimum number of windows before clustering
    MIN_WINDOWS_FOR_CLUSTERING = 2

    def __init__(
        self,
        num_speakers: Optional[int] = None,  # None = auto-detect
        min_speakers: int = 2,
        max_speakers: int = 4,
        sample_rate: int = 16000,  # SpeechBrain expects 16kHz
    ):
        """
        Initialize the speaker diarizer.

        Args:
            num_speakers: Fixed number of speakers (None = auto-detect)
            min_speakers: Minimum speakers for auto-detect
            max_speakers: Maximum speakers for auto-detect
            sample_rate: Expected audio sample rate
        """
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.sample_rate = sample_rate

        self._model: Optional[EncoderClassifier] = None
        self._model_loaded = False
        self._loading_model = False

        # Global embedding storage for clustering across all segments
        # Key: (segment_id, window_idx), Value: embedding
        self._window_embeddings: Dict[Tuple[int, int], np.ndarray] = {}
        # Key: (segment_id, window_idx), Value: (abs_start, abs_end)
        self._window_times: Dict[Tuple[int, int], Tuple[float, float]] = {}

        # Segment info storage
        self._segment_window_count: Dict[int, int] = {}  # segment_id -> num windows

        # Processing queue
        self._queue: Queue = Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Callbacks
        self._on_speaker_callback: Optional[Callable[[SpeakerResult], None]] = None
        self._on_model_loaded_callback: Optional[Callable[[], None]] = None

        # Current speaker assignments per window
        self._window_assignments: Dict[Tuple[int, int], str] = {}

        # Batch processing settings
        self._batch_delay = 1.0  # Wait before processing (to batch segments)
        self._last_segment_time = 0.0

    def set_on_speaker_callback(self, callback: Callable[[SpeakerResult], None]):
        """Set callback for when a segment's speaker is identified"""
        self._on_speaker_callback = callback

    def set_on_model_loaded_callback(self, callback: Callable[[], None]):
        """Set callback for when model finishes loading"""
        self._on_model_loaded_callback = callback

    def load_model(self, blocking: bool = False):
        """
        Load the ECAPA-TDNN model.

        Args:
            blocking: If True, wait for model to load
        """
        if self._model_loaded or self._loading_model:
            logger.debug(f"Model already loaded or loading: loaded={self._model_loaded}, loading={self._loading_model}")
            return

        if not SPEECHBRAIN_AVAILABLE:
            logger.error("SpeechBrain not installed. Run: pip install speechbrain")
            return

        if not TORCH_AVAILABLE:
            logger.error("PyTorch not installed. Run: pip install torch torchaudio")
            return

        self._loading_model = True

        def _load():
            try:
                logger.info(f"Loading ECAPA-TDNN speaker embedding model...")
                start_time = time.time()

                # Check if local model files exist (hyperparams.yaml is required)
                hyperparams_file = self.LOCAL_MODEL_DIR / "hyperparams.yaml"
                if hyperparams_file.exists():
                    logger.info(f"Loading from local path: {self.LOCAL_MODEL_DIR}")
                    source = str(self.LOCAL_MODEL_DIR)
                    savedir = str(self.LOCAL_MODEL_DIR)
                else:
                    logger.info(f"Downloading model from: {self.DEFAULT_MODEL_SOURCE}")
                    source = self.DEFAULT_MODEL_SOURCE
                    savedir = str(self.LOCAL_MODEL_DIR)
                    # Create directory if it doesn't exist
                    self.LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

                # Load model (CPU only, use COPY strategy for Windows compatibility)
                self._model = EncoderClassifier.from_hparams(
                    source=source,
                    savedir=savedir,
                    run_opts={"device": "cpu"},  # Force CPU
                    local_strategy=LocalStrategy.COPY if LocalStrategy else None,
                )

                load_time = time.time() - start_time
                logger.info(f"ECAPA-TDNN model loaded in {load_time:.1f}s")

                self._model_loaded = True
                self._loading_model = False

                if self._on_model_loaded_callback:
                    self._on_model_loaded_callback()

            except Exception as e:
                import traceback
                logger.error(f"Failed to load ECAPA-TDNN model: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                self._loading_model = False

        if blocking:
            _load()
        else:
            threading.Thread(target=_load, daemon=True).start()

    def start(self):
        """Start the diarization worker"""
        if self._running:
            return

        if not self._model_loaded:
            logger.warning("Model not loaded, diarization disabled")
            return

        self._running = True

        # Reset state
        with self._lock:
            self._window_embeddings.clear()
            self._window_times.clear()
            self._segment_window_count.clear()
            self._window_assignments.clear()

        # Clear queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        logger.info("Speaker diarizer started (sub-segment mode)")

    def stop(self):
        """Stop the diarization worker"""
        if not self._running:
            return

        self._running = False
        self._queue.put(None)  # Signal to stop

        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

        # Process any remaining segments
        self._process_batch()

        logger.info("Speaker diarizer stopped")

    def queue_segment(self, segment_id: int, audio: np.ndarray, start_time: float, end_time: float):
        """
        Queue a segment for speaker identification with sub-segment analysis.

        Args:
            segment_id: Unique segment identifier
            audio: Audio data (expected at self.sample_rate, int16 or float32)
            start_time: Absolute start time in seconds
            end_time: Absolute end time in seconds
        """
        if not self._running:
            return

        duration = end_time - start_time
        if duration < self.MIN_WINDOW_DURATION:
            logger.debug(f"Segment {segment_id} too short ({duration:.2f}s), skipping")
            return

        self._queue.put((segment_id, audio, start_time, end_time))
        self._last_segment_time = time.time()

    def _worker_loop(self):
        """Main worker loop"""
        logger.debug("Speaker diarization worker started")

        pending_segments: List[tuple] = []

        while self._running:
            try:
                # Get segment from queue with timeout
                item = self._queue.get(timeout=0.5)

                if item is None:
                    break

                pending_segments.append(item)

            except Empty:
                # Check if we should process pending segments
                if pending_segments:
                    time_since_last = time.time() - self._last_segment_time
                    if time_since_last >= self._batch_delay:
                        self._process_segments(pending_segments)
                        pending_segments.clear()
                continue
            except Exception as e:
                logger.error(f"Error in diarization worker: {e}")
                continue

        # Process remaining segments
        if pending_segments:
            self._process_segments(pending_segments)

        logger.debug("Speaker diarization worker stopped")

    def _split_into_windows(self, audio: np.ndarray, start_time: float, end_time: float) -> List[Tuple[np.ndarray, float, float]]:
        """
        Split audio into overlapping windows for sub-segment analysis.

        Args:
            audio: Audio data
            start_time: Absolute start time
            end_time: Absolute end time

        Returns:
            List of (audio_chunk, abs_start, abs_end) tuples
        """
        duration = end_time - start_time
        window_samples = int(self.WINDOW_DURATION * self.sample_rate)
        step_samples = int(self.WINDOW_STEP * self.sample_rate)
        min_samples = int(self.MIN_WINDOW_DURATION * self.sample_rate)

        windows = []
        pos = 0

        while pos < len(audio):
            end_pos = min(pos + window_samples, len(audio))
            chunk = audio[pos:end_pos]

            # Skip if chunk is too short
            if len(chunk) < min_samples:
                break

            # Calculate absolute times
            chunk_start = start_time + (pos / self.sample_rate)
            chunk_end = start_time + (end_pos / self.sample_rate)

            windows.append((chunk, chunk_start, chunk_end))

            pos += step_samples

            # For very short segments, just use one window
            if duration < self.WINDOW_DURATION * 1.5:
                break

        return windows

    def _process_segments(self, segments: List[tuple]):
        """Process a batch of segments with sub-segment windowing"""
        new_segment_ids = []

        for segment_id, audio, start_time, end_time in segments:
            try:
                # Split into windows
                windows = self._split_into_windows(audio, start_time, end_time)

                if not windows:
                    logger.debug(f"No valid windows for segment {segment_id}")
                    continue

                logger.debug(f"Segment {segment_id}: {len(windows)} windows")

                # Extract embedding for each window
                window_idx = 0
                for chunk, chunk_start, chunk_end in windows:
                    embedding = self._extract_embedding(chunk)

                    if embedding is not None:
                        key = (segment_id, window_idx)
                        with self._lock:
                            self._window_embeddings[key] = embedding
                            self._window_times[key] = (chunk_start, chunk_end)
                        window_idx += 1

                with self._lock:
                    self._segment_window_count[segment_id] = window_idx

                if window_idx > 0:
                    new_segment_ids.append(segment_id)

            except Exception as e:
                logger.error(f"Error processing segment {segment_id}: {e}")

        # Run clustering if we have enough windows
        if new_segment_ids:
            self._process_batch(new_segment_ids)

    def _process_batch(self, new_segment_ids: List[int] = None):
        """Cluster all window embeddings and assign speakers"""
        with self._lock:
            if len(self._window_embeddings) < self.MIN_WINDOWS_FOR_CLUSTERING:
                return

            # Get all window keys and embeddings
            window_keys = list(self._window_embeddings.keys())
            embeddings = np.array([self._window_embeddings[k] for k in window_keys])

        # Perform clustering
        try:
            labels = self._cluster_embeddings(embeddings)

            # Assign speaker labels to windows
            with self._lock:
                for i, key in enumerate(window_keys):
                    label_idx = labels[i]
                    speaker_label = self.SPEAKER_LABELS[label_idx] if label_idx < len(self.SPEAKER_LABELS) else f"Speaker {label_idx + 1}"
                    self._window_assignments[key] = speaker_label

            # Build results for segments that have new data
            segments_to_report = new_segment_ids if new_segment_ids else set(k[0] for k in window_keys)

            for segment_id in segments_to_report:
                self._emit_segment_result(segment_id)

        except Exception as e:
            logger.error(f"Error clustering embeddings: {e}")

    def _merge_consecutive_speakers(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """
        Merge consecutive segments with the same speaker.

        E.g., [A(0-1), A(1-2), B(2-3), A(3-4)] -> [A(0-2), B(2-3), A(3-4)]
        """
        if len(segments) <= 1:
            return segments

        merged = []
        current = segments[0]

        for seg in segments[1:]:
            if seg.speaker_label == current.speaker_label:
                # Same speaker - extend the current segment
                current = SpeakerSegment(
                    start_time=current.start_time,
                    end_time=seg.end_time,
                    speaker_label=current.speaker_label
                )
            else:
                # Different speaker - save current and start new
                merged.append(current)
                current = seg

        # Don't forget the last segment
        merged.append(current)

        if len(merged) < len(segments):
            logger.debug(f"Merged {len(segments)} windows into {len(merged)} speaker segments")

        return merged

    def _emit_segment_result(self, segment_id: int):
        """Build and emit SpeakerResult for a segment"""
        with self._lock:
            window_count = self._segment_window_count.get(segment_id, 0)
            if window_count == 0:
                return

            # Collect speaker segments
            speaker_segments = []
            speaker_counts = {}

            for window_idx in range(window_count):
                key = (segment_id, window_idx)
                if key in self._window_assignments and key in self._window_times:
                    speaker = self._window_assignments[key]
                    start, end = self._window_times[key]
                    speaker_segments.append(SpeakerSegment(
                        start_time=start,
                        end_time=end,
                        speaker_label=speaker
                    ))
                    speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

            if not speaker_segments:
                return

            # Merge consecutive same-speaker segments
            speaker_segments = self._merge_consecutive_speakers(speaker_segments)

            # Recalculate speaker counts after merging
            speaker_counts = {}
            for seg in speaker_segments:
                speaker_counts[seg.speaker_label] = speaker_counts.get(seg.speaker_label, 0) + 1

            # Determine primary speaker
            primary_speaker = max(speaker_counts, key=speaker_counts.get)

        # Emit callback
        if self._on_speaker_callback:
            result = SpeakerResult(
                segment_id=segment_id,
                speakers=speaker_segments,
                primary_speaker=primary_speaker
            )
            self._on_speaker_callback(result)

            if result.has_multiple_speakers:
                logger.info(f"Segment {segment_id}: Multiple speakers detected - {result.get_speaker_sequence()}")
            else:
                logger.debug(f"Segment {segment_id}: {primary_speaker}")

    def _extract_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract speaker embedding from audio.

        Args:
            audio: Audio data (int16 or float32)

        Returns:
            192-dimensional embedding vector, or None if failed
        """
        if self._model is None:
            return None

        try:
            # Convert to float32 if needed
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Ensure 1D
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Convert to torch tensor
            waveform = torch.tensor(audio).unsqueeze(0)

            # Extract embedding
            with torch.no_grad():
                embedding = self._model.encode_batch(waveform)

            # Convert to numpy
            embedding = embedding.squeeze().cpu().numpy()

            return embedding

        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster embeddings to assign speaker labels.

        Args:
            embeddings: Array of shape (n_windows, embedding_dim)

        Returns:
            Array of cluster labels (0, 1, 2, ...)
        """
        if not SKLEARN_AVAILABLE:
            # Fallback: assign all to Speaker A
            return np.zeros(len(embeddings), dtype=int)

        n_samples = len(embeddings)

        # Determine number of clusters
        if self.num_speakers is not None:
            n_clusters = self.num_speakers
        else:
            # Auto-detect: use min_speakers as default, or fewer if not enough samples
            n_clusters = min(self.min_speakers, n_samples)

        # Need at least 2 samples for clustering
        if n_samples < 2:
            return np.zeros(n_samples, dtype=int)

        # Can't have more clusters than samples
        n_clusters = min(n_clusters, n_samples)

        # Perform Agglomerative Clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )

        labels = clustering.fit_predict(embeddings)

        return labels

    def get_speaker_for_segment(self, segment_id: int) -> Optional[str]:
        """Get the primary speaker for a segment"""
        with self._lock:
            window_count = self._segment_window_count.get(segment_id, 0)
            if window_count == 0:
                return None

            # Count speakers across windows
            speaker_counts = {}
            for window_idx in range(window_count):
                key = (segment_id, window_idx)
                if key in self._window_assignments:
                    speaker = self._window_assignments[key]
                    speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

            if not speaker_counts:
                return None

            return max(speaker_counts, key=speaker_counts.get)

    def get_all_assignments(self) -> Dict[int, str]:
        """Get all primary speaker assignments"""
        result = {}
        with self._lock:
            for segment_id in self._segment_window_count:
                speaker = self.get_speaker_for_segment(segment_id)
                if speaker:
                    result[segment_id] = speaker
        return result

    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model_loaded

    def is_running(self) -> bool:
        """Check if diarizer is running"""
        return self._running


# Singleton instance
_diarizer_instance: Optional[SpeakerDiarizer] = None


def get_speaker_diarizer(
    num_speakers: Optional[int] = None,
    min_speakers: int = 2,
    max_speakers: int = 4,
) -> SpeakerDiarizer:
    """
    Get the singleton speaker diarizer instance.

    Args:
        num_speakers: Fixed number of speakers (None = auto-detect)
        min_speakers: Minimum speakers for auto-detect
        max_speakers: Maximum speakers for auto-detect
    """
    global _diarizer_instance
    if _diarizer_instance is None:
        _diarizer_instance = SpeakerDiarizer(
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    else:
        # Update settings if changed
        if num_speakers is not None:
            _diarizer_instance.num_speakers = num_speakers
        _diarizer_instance.min_speakers = min_speakers
        _diarizer_instance.max_speakers = max_speakers
    return _diarizer_instance
