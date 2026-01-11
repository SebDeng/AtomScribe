"""Real-time transcription service using faster-whisper"""

import json
import time
import threading
import numpy as np
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass, asdict, field
from loguru import logger

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None
    logger.warning("faster-whisper not installed. Transcription will not work.")


@dataclass
class Word:
    """A single word with timing information"""
    word: str
    start: float  # seconds from recording start
    end: float
    probability: Optional[float] = None


@dataclass
class TranscriptSegment:
    """A single transcription segment"""
    id: int
    start: float  # seconds from recording start
    end: float
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    is_partial: bool = False  # True if this is an interim result that may be updated
    speaker: Optional[str] = None  # Speaker label (e.g., "Speaker A", "Speaker B")
    words: List["Word"] = field(default_factory=list)  # Word-level timestamps for conversation splitting


class RealtimeTranscriber:
    """
    Real-time transcription service using faster-whisper.

    Processes audio chunks as they arrive, buffers them, and runs
    transcription when enough audio has accumulated.
    """

    # Minimum audio duration (seconds) before processing
    # Longer buffer = better accuracy, shorter = faster response
    MIN_BUFFER_DURATION = 3.0

    # Maximum audio duration (seconds) to process at once
    MAX_BUFFER_DURATION = 30.0

    # Overlap with previous chunk (seconds) for context
    OVERLAP_DURATION = 0.5

    # Flush transcript to file every N seconds
    FILE_FLUSH_INTERVAL = 30.0

    # Flush transcript to file every N segments
    FILE_FLUSH_SEGMENTS = 20

    # Streaming mode settings
    STREAMING_WORD_DELAY = 0.03  # Delay between emitting words (seconds)

    # Gap threshold (seconds) to start a new paragraph/segment
    # Longer gap = more natural paragraphs, shorter = more segments
    PARAGRAPH_GAP_THRESHOLD = 1.0

    # Silence timeout (seconds) - finalize segment if no new words for this duration
    SILENCE_TIMEOUT = 1.2

    # Default scientific vocabulary for electron microscopy
    DEFAULT_SCIENTIFIC_PROMPT = (
        "electron microscope, TEM, SEM, high tension, stigmator, aperture, "
        "astigmatism, focus, alignment, specimen, vacuum, filament, "
        "千伏, 电子显微镜, 透射电镜, 扫描电镜, 高压, 消像散, 光阑, 聚焦, 对中, 样品"
    )

    # Default text replacements for common transcription errors
    DEFAULT_REPLACEMENTS = {
        # Scientific terms
        "hi-tension": "high tension",
        "Hi-tension": "High tension",
        "hi tension": "high tension",
        "千幅": "千伏",
        "千福": "千伏",
        "stigma": "stigmator",
        "stickmate": "stigmator",
        "stickma": "stigmator",
        # Common Chinese homophones
        "负镜": "复镜",
        "电景": "电镜",
    }

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        sample_rate: int = 16000,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        hotwords: Optional[List[str]] = None,
        replacements: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the transcriber.

        Args:
            model_size: Whisper model size (large-v3, medium, small, etc.)
            device: Device to use (cuda, cpu)
            compute_type: Computation type (float16, int8, etc.)
            sample_rate: Expected audio sample rate (Whisper uses 16kHz)
            language: Language code (e.g., "zh", "en", None for auto-detect)
            initial_prompt: Prompt to guide transcription style and vocabulary
            hotwords: List of words to boost during transcription
            replacements: Dict of {wrong: correct} text replacements
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.sample_rate = sample_rate
        self.language = language  # None = auto-detect

        # Use default scientific prompt if none provided
        self.initial_prompt = initial_prompt or self.DEFAULT_SCIENTIFIC_PROMPT

        # Hotwords for boosting specific terms
        self.hotwords = hotwords or []

        # Text replacements (merge with defaults)
        self.replacements = {**self.DEFAULT_REPLACEMENTS}
        if replacements:
            self.replacements.update(replacements)

        self._model: Optional[WhisperModel] = None
        self._model_loaded = False
        self._loading_model = False

        # Audio buffering
        self._audio_queue: Queue = Queue()
        self._audio_buffer: List[np.ndarray] = []
        self._buffer_duration = 0.0

        # Transcript storage
        self._segments: List[TranscriptSegment] = []
        self._segment_counter = 0
        self._recording_start_time = 0.0
        self._last_processed_time = 0.0

        # File output
        self._output_path: Optional[Path] = None
        self._last_flush_time = 0.0
        self._segments_since_flush = 0

        # Threading
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Callbacks
        self._on_segment_callback: Optional[Callable[[TranscriptSegment], None]] = None
        self._on_segment_update_callback: Optional[Callable[[TranscriptSegment], None]] = None
        self._on_model_loaded_callback: Optional[Callable[[], None]] = None

        # Streaming state
        self._current_streaming_segment: Optional[TranscriptSegment] = None
        self._streaming_words: List[str] = []
        self._previous_text: str = ""  # For condition_on_previous_text
        self._last_emitted_end_time: float = 0.0  # Track to avoid duplicate words
        self._last_word_time: float = 0.0  # Track last time we emitted a word (for silence timeout)

    def _apply_replacements(self, text: str) -> str:
        """Apply text replacements for common transcription errors"""
        if not self.replacements:
            return text
        for wrong, correct in self.replacements.items():
            text = text.replace(wrong, correct)
        return text

    def _is_cjk_char(self, char: str) -> bool:
        """Check if a character is CJK (Chinese/Japanese/Korean)"""
        if not char:
            return False
        code = ord(char)
        # CJK Unified Ideographs and common ranges
        return (
            0x4E00 <= code <= 0x9FFF or  # CJK Unified Ideographs
            0x3400 <= code <= 0x4DBF or  # CJK Unified Ideographs Extension A
            0x3000 <= code <= 0x303F or  # CJK Symbols and Punctuation
            0xFF00 <= code <= 0xFFEF or  # Fullwidth Forms
            0x3040 <= code <= 0x309F or  # Hiragana
            0x30A0 <= code <= 0x30FF     # Katakana
        )

    def _should_add_space(self, prev_text: str, new_word: str) -> bool:
        """Determine if a space should be added between previous text and new word"""
        if not prev_text or not new_word:
            return False

        # Don't add space before punctuation
        if new_word[0] in "',.!?;:，。！？；：、":
            return False

        # Get last char of previous text and first char of new word
        last_char = prev_text[-1] if prev_text else ""
        first_char = new_word[0] if new_word else ""

        # Don't add space between CJK characters
        if self._is_cjk_char(last_char) and self._is_cjk_char(first_char):
            return False

        # Don't add space after CJK punctuation
        if last_char in "，。！？；：、（）【】":
            return False

        return True

    def add_replacement(self, wrong: str, correct: str):
        """Add a custom replacement rule"""
        self.replacements[wrong] = correct
        logger.debug(f"Added replacement: '{wrong}' -> '{correct}'")

    def set_on_segment_callback(self, callback: Callable[[TranscriptSegment], None]):
        """Set callback for when a new segment is transcribed"""
        self._on_segment_callback = callback

    def set_on_segment_update_callback(self, callback: Callable[[TranscriptSegment], None]):
        """Set callback for when an existing segment is updated (streaming mode)"""
        self._on_segment_update_callback = callback

    def set_on_model_loaded_callback(self, callback: Callable[[], None]):
        """Set callback for when model finishes loading"""
        self._on_model_loaded_callback = callback

    def load_model(self, blocking: bool = False):
        """
        Load the Whisper model.

        Args:
            blocking: If True, wait for model to load. If False, load in background.
        """
        if self._model_loaded or self._loading_model:
            return

        if WhisperModel is None:
            logger.error("faster-whisper not installed")
            return

        self._loading_model = True

        def _load():
            try:
                logger.info(f"Loading Whisper model: {self.model_size} on {self.device}")
                start_time = time.time()

                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )

                load_time = time.time() - start_time
                logger.info(f"Whisper model loaded in {load_time:.1f}s")

                self._model_loaded = True
                self._loading_model = False

                if self._on_model_loaded_callback:
                    self._on_model_loaded_callback()

            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                self._loading_model = False

        if blocking:
            _load()
        else:
            threading.Thread(target=_load, daemon=True).start()

    def start(self, output_path: Optional[Path] = None):
        """
        Start the transcription service.

        Args:
            output_path: Path to save transcript JSON file
        """
        if self._running:
            logger.warning("Transcriber already running")
            return

        # Wait for model if it's currently loading
        if self._loading_model:
            logger.info("Waiting for transcription model to finish loading...")
            # Wait up to 60 seconds for model to load
            wait_time = 0
            while self._loading_model and wait_time < 60:
                time.sleep(0.5)
                wait_time += 0.5

            if self._loading_model:
                logger.error("Timeout waiting for model to load")
                return

        # Load model if not loaded
        if not self._model_loaded:
            self.load_model(blocking=True)

        if not self._model_loaded:
            logger.error("Cannot start transcriber: model not loaded")
            return

        self._output_path = output_path
        self._running = True
        self._recording_start_time = time.time()
        self._last_flush_time = time.time()
        self._last_processed_time = 0.0
        self._segments = []
        self._segment_counter = 0
        self._audio_buffer = []
        self._buffer_duration = 0.0
        self._current_streaming_segment = None
        self._streaming_words = []
        self._previous_text = ""
        self._last_emitted_end_time = 0.0
        self._last_word_time = 0.0

        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except Empty:
                break

        # Start worker thread
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        logger.info("Transcriber started")

    def stop(self) -> List[TranscriptSegment]:
        """
        Stop the transcription service.

        Returns:
            List of all transcribed segments
        """
        if not self._running:
            return self._segments

        self._running = False

        # Signal worker to stop
        self._audio_queue.put(None)

        # Wait for worker to finish
        if self._worker_thread:
            self._worker_thread.join(timeout=10.0)

        # Process any remaining audio
        self._process_remaining()

        # Finalize any ongoing streaming segment
        if self._current_streaming_segment is not None:
            self._current_streaming_segment.is_partial = False
            self._segments.append(self._current_streaming_segment)
            self._segments_since_flush += 1

            if self._on_segment_update_callback:
                self._on_segment_update_callback(self._current_streaming_segment)

            logger.debug(f"Final segment {self._current_streaming_segment.id}: [{self._current_streaming_segment.start:.1f}s] {self._current_streaming_segment.text}")
            self._current_streaming_segment = None

        # Final flush to file
        self._flush_to_file(force=True)

        logger.info(f"Transcriber stopped. Total segments: {len(self._segments)}")
        return self._segments

    def feed_audio(self, audio_data: np.ndarray, source_sample_rate: int = 44100):
        """
        Feed audio data to the transcriber.

        Args:
            audio_data: Audio samples (int16 or float32)
            source_sample_rate: Sample rate of the input audio
        """
        if not self._running:
            return

        # Convert to float32 if needed
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0

        # Resample if needed (Whisper expects 16kHz)
        if source_sample_rate != self.sample_rate:
            # Simple resampling - for better quality, use librosa or scipy
            ratio = self.sample_rate / source_sample_rate
            new_length = int(len(audio_data) * ratio)
            indices = np.linspace(0, len(audio_data) - 1, new_length).astype(int)
            audio_data = audio_data[indices]

        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        self._audio_queue.put(audio_data)

    def _worker_loop(self):
        """Main worker loop - processes audio queue"""
        logger.debug("Transcription worker started")

        while self._running:
            try:
                # Get audio from queue with timeout
                audio_chunk = self._audio_queue.get(timeout=0.1)

                if audio_chunk is None:
                    # Stop signal
                    break

                # Add to buffer
                with self._lock:
                    self._audio_buffer.append(audio_chunk)
                    self._buffer_duration += len(audio_chunk) / self.sample_rate

                # Check if we have enough audio to process
                if self._buffer_duration >= self.MIN_BUFFER_DURATION:
                    self._process_buffer()

            except Empty:
                # No audio available, check if we should process what we have
                if self._buffer_duration >= self.MIN_BUFFER_DURATION:
                    self._process_buffer()

                # Check for silence timeout - finalize segment if no new words for a while
                self._check_silence_timeout()
                continue
            except Exception as e:
                logger.error(f"Error in transcription worker: {e}")
                continue

        logger.debug("Transcription worker stopped")

    def _check_silence_timeout(self):
        """Check if we should finalize the current segment due to silence"""
        if self._current_streaming_segment is None:
            return

        # Only check if we've emitted at least one word
        if self._last_word_time <= 0:
            return

        time_since_last_word = time.time() - self._last_word_time

        if time_since_last_word >= self.SILENCE_TIMEOUT:
            # Finalize the current segment
            self._current_streaming_segment.is_partial = False
            self._segments.append(self._current_streaming_segment)
            self._segments_since_flush += 1

            # Update previous text for context
            self._previous_text += " " + self._current_streaming_segment.text

            # Emit the finalized segment
            if self._on_segment_update_callback:
                self._on_segment_update_callback(self._current_streaming_segment)

            logger.debug(f"Segment {self._current_streaming_segment.id} finalized due to silence timeout: [{self._current_streaming_segment.start:.1f}s] {self._current_streaming_segment.text}")

            self._current_streaming_segment = None
            self._last_word_time = 0.0

    def _process_buffer(self):
        """Process the accumulated audio buffer"""
        with self._lock:
            if not self._audio_buffer:
                return

            # Concatenate all audio
            audio = np.concatenate(self._audio_buffer)

            # Limit to max duration
            max_samples = int(self.MAX_BUFFER_DURATION * self.sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            # Calculate time offset
            time_offset = self._last_processed_time

            # Keep overlap for context
            overlap_samples = int(self.OVERLAP_DURATION * self.sample_rate)
            if len(audio) > overlap_samples:
                self._audio_buffer = [audio[-overlap_samples:]]
                self._buffer_duration = self.OVERLAP_DURATION
                self._last_processed_time += (len(audio) - overlap_samples) / self.sample_rate
            else:
                self._audio_buffer = []
                self._buffer_duration = 0.0
                self._last_processed_time += len(audio) / self.sample_rate

        # Transcribe with word timestamps for streaming
        try:
            # Build transcription options
            transcribe_opts = dict(
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,  # 500ms silence to split
                    speech_pad_ms=200,  # Padding around speech
                    threshold=0.5,  # VAD threshold
                ),
                word_timestamps=True,  # Enable word-level timestamps
                condition_on_previous_text=True,
            )

            # Set language if specified (otherwise auto-detect)
            if self.language:
                transcribe_opts["language"] = self.language

            # Set initial prompt for domain vocabulary and style
            if self.initial_prompt:
                transcribe_opts["initial_prompt"] = self.initial_prompt
            elif self._previous_text:
                # Use previous transcription as context
                transcribe_opts["initial_prompt"] = self._previous_text[-500:]

            segments, info = self._model.transcribe(audio, **transcribe_opts)

            # Process segments with word-level streaming
            for seg in segments:
                seg_text = seg.text.strip()
                if not seg_text:
                    continue

                # Get words from segment
                words = list(seg.words) if hasattr(seg, 'words') and seg.words else []

                if words:
                    # Stream words one by one
                    self._stream_words(
                        words=words,
                        time_offset=time_offset,
                        language=info.language if hasattr(info, 'language') else None,
                        confidence=seg.avg_logprob if hasattr(seg, 'avg_logprob') else None,
                    )
                else:
                    # Fallback: emit full segment if no word timestamps
                    segment = TranscriptSegment(
                        id=self._segment_counter,
                        start=time_offset + seg.start,
                        end=time_offset + seg.end,
                        text=seg_text,
                        language=info.language if hasattr(info, 'language') else None,
                        confidence=seg.avg_logprob if hasattr(seg, 'avg_logprob') else None,
                        is_partial=False,
                    )
                    self._segment_counter += 1
                    self._segments.append(segment)
                    self._segments_since_flush += 1

                    if self._on_segment_callback:
                        self._on_segment_callback(segment)

                    logger.debug(f"Segment {segment.id}: [{segment.start:.1f}s] {segment.text}")

            # Check if should flush to file
            self._check_flush()

        except Exception as e:
            logger.error(f"Transcription error: {e}")

    def _stream_words(self, words, time_offset: float, language: str, confidence: float):
        """Stream words one by one for real-time display"""
        if not words:
            return

        first_word_start = time_offset + words[0].start if words else time_offset

        # Check if we should continue the current segment or start a new one
        # Start new segment if:
        # 1. No current segment exists
        # 2. There's a significant pause (> PARAGRAPH_GAP_THRESHOLD)
        should_start_new = (
            self._current_streaming_segment is None or
            (first_word_start - self._current_streaming_segment.end) > self.PARAGRAPH_GAP_THRESHOLD
        )

        if should_start_new and self._current_streaming_segment is not None:
            # Finalize previous segment before starting new one
            self._current_streaming_segment.is_partial = False
            self._segments.append(self._current_streaming_segment)
            self._segments_since_flush += 1

            # Update previous text for context in next transcription
            self._previous_text += " " + self._current_streaming_segment.text

            if self._on_segment_update_callback:
                self._on_segment_update_callback(self._current_streaming_segment)

            logger.debug(f"Segment {self._current_streaming_segment.id}: [{self._current_streaming_segment.start:.1f}s] {self._current_streaming_segment.text}")
            self._current_streaming_segment = None

        for i, word in enumerate(words):
            word_text = word.word.strip()
            if not word_text:
                continue

            # Calculate absolute time for this word
            word_start_time = time_offset + word.start
            word_end_time = time_offset + word.end

            # Skip words that have already been emitted (from overlap)
            # Use a small tolerance to avoid floating point issues
            if word_start_time < self._last_emitted_end_time - 0.1:
                continue

            # Apply replacements to individual word
            word_text = self._apply_replacements(word_text)

            is_last_word = (i == len(words) - 1)

            # Create Word object for this word
            word_obj = Word(
                word=word_text,
                start=word_start_time,
                end=word_end_time,
                probability=word.probability if hasattr(word, 'probability') else None,
            )

            if self._current_streaming_segment is None:
                # Create new streaming segment
                self._current_streaming_segment = TranscriptSegment(
                    id=self._segment_counter,
                    start=word_start_time,
                    end=word_end_time,
                    text=word_text,
                    language=language,
                    confidence=confidence,
                    is_partial=True,
                    words=[word_obj],  # Initialize with first word
                )
                self._segment_counter += 1

                # Emit new segment
                if self._on_segment_callback:
                    self._on_segment_callback(self._current_streaming_segment)
            else:
                # Add word to existing streaming segment with smart spacing
                current_text = self._current_streaming_segment.text

                # Use smart spacing (no space between CJK characters)
                if self._should_add_space(current_text, word_text):
                    current_text += " " + word_text
                else:
                    current_text += word_text

                # Apply replacements to catch multi-word patterns
                current_text = self._apply_replacements(current_text)

                self._current_streaming_segment.text = current_text
                self._current_streaming_segment.end = word_end_time
                self._current_streaming_segment.words.append(word_obj)  # Add word to list

                # Emit update
                if self._on_segment_update_callback:
                    self._on_segment_update_callback(self._current_streaming_segment)

            # Update last emitted time to avoid duplicates
            self._last_emitted_end_time = word_end_time

            # Update last word time for silence timeout detection
            self._last_word_time = time.time()

            # Small delay between words for visual effect
            if not is_last_word and self._running:
                time.sleep(self.STREAMING_WORD_DELAY)

    def _process_remaining(self):
        """Process any remaining audio in the buffer"""
        with self._lock:
            if not self._audio_buffer:
                return
            audio = np.concatenate(self._audio_buffer)
            self._audio_buffer = []
            self._buffer_duration = 0.0

        if len(audio) < self.sample_rate * 0.5:  # Less than 0.5s
            return

        try:
            time_offset = self._last_processed_time

            segments, info = self._model.transcribe(
                audio,
                vad_filter=True,
                word_timestamps=True,  # Use word timestamps for streaming
            )

            for seg in segments:
                seg_text = seg.text.strip()
                if not seg_text:
                    continue

                # Get words from segment
                words = list(seg.words) if hasattr(seg, 'words') and seg.words else []

                if words:
                    # Stream words
                    self._stream_words(
                        words=words,
                        time_offset=time_offset,
                        language=info.language if hasattr(info, 'language') else None,
                        confidence=seg.avg_logprob if hasattr(seg, 'avg_logprob') else None,
                    )
                else:
                    # Fallback: add to current segment or create new one
                    if self._current_streaming_segment:
                        self._current_streaming_segment.text += " " + seg_text
                        self._current_streaming_segment.end = time_offset + seg.end
                        if self._on_segment_update_callback:
                            self._on_segment_update_callback(self._current_streaming_segment)
                    else:
                        segment = TranscriptSegment(
                            id=self._segment_counter,
                            start=time_offset + seg.start,
                            end=time_offset + seg.end,
                            text=seg_text,
                            language=info.language if hasattr(info, 'language') else None,
                            is_partial=False,
                        )
                        self._segment_counter += 1
                        self._segments.append(segment)

                        if self._on_segment_callback:
                            self._on_segment_callback(segment)

        except Exception as e:
            logger.error(f"Error processing remaining audio: {e}")

    def _check_flush(self):
        """Check if we should flush segments to file"""
        now = time.time()
        should_flush = (
            (now - self._last_flush_time >= self.FILE_FLUSH_INTERVAL) or
            (self._segments_since_flush >= self.FILE_FLUSH_SEGMENTS)
        )

        if should_flush:
            self._flush_to_file()

    def _flush_to_file(self, force: bool = False):
        """Flush segments to JSON file"""
        if not self._output_path:
            return

        if not self._segments and not force:
            return

        try:
            transcript_data = {
                "model": self.model_size,
                "segments": [asdict(seg) for seg in self._segments],
            }

            with open(self._output_path, "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=2)

            self._last_flush_time = time.time()
            self._segments_since_flush = 0

            logger.debug(f"Flushed {len(self._segments)} segments to {self._output_path}")

        except Exception as e:
            logger.error(f"Error flushing transcript to file: {e}")

    def get_segments(self) -> List[TranscriptSegment]:
        """Get all transcribed segments"""
        return self._segments.copy()

    def is_model_loaded(self) -> bool:
        """Check if the model is loaded"""
        return self._model_loaded

    def is_running(self) -> bool:
        """Check if the transcriber is running"""
        return self._running


# Singleton instance
_transcriber_instance: Optional[RealtimeTranscriber] = None


def get_transcriber(
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    hotwords: Optional[List[str]] = None,
    replacements: Optional[Dict[str, str]] = None,
) -> RealtimeTranscriber:
    """
    Get the singleton transcriber instance.

    Args:
        language: Language code to use (only applied on first call or if changed)
        initial_prompt: Initial prompt for domain vocabulary (only applied on first call or if changed)
        hotwords: List of words to boost during transcription
        replacements: Dict of {wrong: correct} text replacements
    """
    global _transcriber_instance
    if _transcriber_instance is None:
        _transcriber_instance = RealtimeTranscriber(
            language=language,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            replacements=replacements,
        )
    else:
        # Update settings if changed
        if language is not None:
            _transcriber_instance.language = language
        if initial_prompt is not None:
            _transcriber_instance.initial_prompt = initial_prompt
        if hotwords is not None:
            _transcriber_instance.hotwords = hotwords
        if replacements is not None:
            _transcriber_instance.replacements.update(replacements)
    return _transcriber_instance
