"""LLM post-processor for transcript correction using Qwen3-4B"""

import threading
import time
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, Callable, List
from dataclasses import dataclass
from loguru import logger

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None
    logger.warning("llama-cpp-python not installed. LLM post-processing will not work.")


@dataclass
class CorrectionResult:
    """Result of LLM correction"""
    segment_id: int
    original_text: str
    corrected_text: str
    is_corrected: bool  # True if text was modified
    merge_with_previous: bool = False  # True if should merge with previous segment
    merged_text: str = ""  # Combined text if merge_with_previous is True


class LLMPostProcessor:
    """
    LLM-based post-processor for transcript correction.

    Uses Qwen3-4B-Instruct to:
    - Fix transcription errors
    - Remove filler words (嗯, 啊, uh, um, etc.)
    - Improve readability
    """

    # Default model path
    DEFAULT_MODEL_PATH = Path("D:/AEG/AtomE_Corp/Projects_MVP_Demos/AtomScribe/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf")

    # System prompt for correction
    SYSTEM_PROMPT = """You are a transcript correction assistant. Your task is to clean up speech-to-text transcripts.

Rules:
1. Fix obvious transcription errors (e.g., "hi-tension" → "high tension", "千幅" → "千伏")
2. Remove filler words: 嗯, 啊, 呃, 那个, 就是说, uh, um, like, you know, I mean
3. Keep the original meaning and language (don't translate)
4. Keep scientific/technical terms accurate
5. Output ONLY the corrected text, nothing else
6. If no correction needed, output the original text exactly"""

    # System prompt for context-aware merge checking
    MERGE_CHECK_PROMPT = """You are a transcript merge checker. Given two consecutive transcript segments, determine if they should be merged into one sentence.

Rules:
1. Answer "MERGE" if the second segment clearly continues the first (incomplete sentence)
2. Answer "NO" if they are separate complete thoughts
3. If MERGE, provide the combined corrected text on the next line
4. Common merge cases:
   - First ends with incomplete phrase: "I think the" + "problem is here" → MERGE
   - First ends mid-sentence without punctuation: "Let's see how" + "this works" → MERGE
5. Do NOT merge if both are complete sentences"""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,  # -1 = use all GPU layers
        n_threads: int = 4,
    ):
        """
        Initialize the LLM post-processor.

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            n_threads: Number of CPU threads
        """
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads

        self._model: Optional[Llama] = None
        self._model_loaded = False
        self._loading_model = False

        # Processing queue
        self._queue: Queue = Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

        # Callbacks
        self._on_correction_callback: Optional[Callable[[CorrectionResult], None]] = None
        self._on_model_loaded_callback: Optional[Callable[[], None]] = None

        # Batch processing settings
        self._batch_delay = 2.0  # Wait this long before processing (to batch segments)
        self._pending_segments: List[tuple] = []  # (segment_id, text)
        self._last_segment_time = 0.0

        # Context history for merge checking
        self._segment_history: List[tuple] = []  # [(segment_id, corrected_text), ...] last N segments
        self._max_history = 3  # Keep last N segments for context

    def set_on_correction_callback(self, callback: Callable[[CorrectionResult], None]):
        """Set callback for when a segment is corrected"""
        self._on_correction_callback = callback

    def set_on_model_loaded_callback(self, callback: Callable[[], None]):
        """Set callback for when model finishes loading"""
        self._on_model_loaded_callback = callback

    def load_model(self, blocking: bool = False):
        """
        Load the LLM model.

        Args:
            blocking: If True, wait for model to load
        """
        if self._model_loaded or self._loading_model:
            logger.debug(f"LLM model already loaded or loading: loaded={self._model_loaded}, loading={self._loading_model}")
            return

        if Llama is None:
            logger.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
            return

        logger.info(f"LLM model path: {self.model_path}")
        logger.info(f"LLM model path exists: {self.model_path.exists()}")

        if not self.model_path.exists():
            logger.warning(f"Model not found at {self.model_path}")
            logger.info("Please download Qwen3-4B-Instruct GGUF model:")
            logger.info("  https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF")
            logger.info(f"  Save to: {self.model_path}")
            return

        self._loading_model = True

        def _load():
            try:
                logger.info(f"Loading LLM model: {self.model_path.name}")
                start_time = time.time()

                self._model = Llama(
                    model_path=str(self.model_path),
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    n_threads=self.n_threads,
                    verbose=True,  # Enable verbose to see detailed errors
                )

                load_time = time.time() - start_time
                logger.info(f"LLM model loaded in {load_time:.1f}s")

                self._model_loaded = True
                self._loading_model = False

                if self._on_model_loaded_callback:
                    self._on_model_loaded_callback()

            except Exception as e:
                import traceback
                logger.error(f"Failed to load LLM model: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                self._loading_model = False

        if blocking:
            _load()
        else:
            threading.Thread(target=_load, daemon=True).start()

    def start(self):
        """Start the post-processor worker"""
        if self._running:
            return

        if not self._model_loaded:
            logger.warning("LLM model not loaded, post-processing disabled")
            return

        self._running = True
        self._pending_segments = []
        self._last_segment_time = 0.0
        self._segment_history = []  # Clear history on start

        # Clear queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        logger.info("LLM post-processor started")

    def stop(self):
        """Stop the post-processor"""
        if not self._running:
            return

        self._running = False
        self._queue.put(None)  # Signal to stop

        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

        # Process any remaining segments
        if self._pending_segments:
            self._process_batch()

        logger.info("LLM post-processor stopped")

    def queue_segment(self, segment_id: int, text: str):
        """Queue a segment for correction"""
        if not self._running:
            return
        self._queue.put((segment_id, text))

    def _worker_loop(self):
        """Main worker loop"""
        logger.debug("LLM post-processor worker started")

        while self._running:
            try:
                # Get segment from queue with timeout
                item = self._queue.get(timeout=0.5)

                if item is None:
                    break

                segment_id, text = item
                self._pending_segments.append((segment_id, text))
                self._last_segment_time = time.time()

            except Empty:
                # Check if we should process pending segments
                if self._pending_segments:
                    time_since_last = time.time() - self._last_segment_time
                    if time_since_last >= self._batch_delay:
                        self._process_batch()
                continue
            except Exception as e:
                logger.error(f"Error in LLM worker: {e}")
                continue

        logger.debug("LLM post-processor worker stopped")

    def _process_batch(self):
        """Process pending segments as a batch"""
        if not self._pending_segments:
            return

        segments = self._pending_segments.copy()
        self._pending_segments.clear()

        for segment_id, text in segments:
            try:
                # First, correct the text
                corrected = self._correct_text(text)

                # Check if should merge with previous segment
                merge_with_previous = False
                merged_text = ""

                if self._segment_history:
                    prev_id, prev_text = self._segment_history[-1]
                    should_merge, combined = self._check_merge(prev_text, corrected)

                    if should_merge and combined:
                        merge_with_previous = True
                        merged_text = combined
                        logger.info(f"Merge suggested: segment {prev_id} + {segment_id}")

                result = CorrectionResult(
                    segment_id=segment_id,
                    original_text=text,
                    corrected_text=corrected,
                    is_corrected=(corrected != text),
                    merge_with_previous=merge_with_previous,
                    merged_text=merged_text,
                )

                if self._on_correction_callback:
                    self._on_correction_callback(result)

                # Update history
                self._segment_history.append((segment_id, corrected))
                if len(self._segment_history) > self._max_history:
                    self._segment_history.pop(0)

                if result.is_corrected:
                    logger.debug(f"Corrected segment {segment_id}: '{text[:30]}...' -> '{corrected[:30]}...'")

            except Exception as e:
                logger.error(f"Error correcting segment {segment_id}: {e}")

    def _correct_text(self, text: str) -> str:
        """Correct a single text using LLM"""
        if not self._model:
            return text

        # Skip very short texts
        if len(text.strip()) < 5:
            return text

        prompt = f"""<|im_start|>system
{self.SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
Correct this transcript:
{text}<|im_end|>
<|im_start|>assistant
"""

        try:
            response = self._model(
                prompt,
                max_tokens=len(text) * 2,  # Allow some expansion
                temperature=0.1,  # Low temperature for consistent output
                top_p=0.9,
                stop=["<|im_end|>", "<|im_start|>"],
                echo=False,
            )

            corrected = response["choices"][0]["text"].strip()

            # Sanity check: if response is empty or too different, return original
            if not corrected or len(corrected) < len(text) * 0.3:
                return text

            return corrected

        except Exception as e:
            logger.error(f"LLM inference error: {e}")
            return text

    def _check_merge(self, prev_text: str, curr_text: str) -> tuple:
        """
        Check if current segment should be merged with previous.

        Args:
            prev_text: Previous segment's corrected text
            curr_text: Current segment's corrected text

        Returns:
            (should_merge: bool, merged_text: str)
        """
        if not self._model:
            return False, ""

        # Quick heuristic check first - skip LLM if obviously not a merge case
        prev_stripped = prev_text.rstrip()

        # If previous ends with strong punctuation, likely complete
        if prev_stripped and prev_stripped[-1] in '.!?。！？':
            # But check if current starts with lowercase (continuation hint)
            curr_stripped = curr_text.lstrip()
            if curr_stripped and curr_stripped[0].isupper():
                return False, ""

        # If previous is very short or current is very short, check for merge
        # Otherwise skip (too expensive to call LLM for every pair)
        if len(prev_text) > 100 and len(curr_text) > 50:
            return False, ""

        prompt = f"""<|im_start|>system
{self.MERGE_CHECK_PROMPT}<|im_end|>
<|im_start|>user
Segment 1: {prev_text}
Segment 2: {curr_text}

Should these be merged?<|im_end|>
<|im_start|>assistant
"""

        try:
            response = self._model(
                prompt,
                max_tokens=len(prev_text) + len(curr_text) + 50,
                temperature=0.1,
                top_p=0.9,
                stop=["<|im_end|>", "<|im_start|>"],
                echo=False,
            )

            result = response["choices"][0]["text"].strip()

            if result.startswith("MERGE"):
                # Extract merged text from response
                lines = result.split('\n')
                if len(lines) > 1:
                    merged = '\n'.join(lines[1:]).strip()
                    if merged:
                        return True, merged

                # Fallback: simple concatenation
                return True, f"{prev_text.rstrip()} {curr_text.lstrip()}"

            return False, ""

        except Exception as e:
            logger.error(f"Merge check error: {e}")
            return False, ""

    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model_loaded

    def is_running(self) -> bool:
        """Check if processor is running"""
        return self._running


# Singleton instance
_processor_instance: Optional[LLMPostProcessor] = None


def get_llm_processor(model_path: Optional[Path] = None) -> LLMPostProcessor:
    """Get the singleton LLM processor instance"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = LLMPostProcessor(model_path=model_path)
    return _processor_instance
