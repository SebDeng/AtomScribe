"""Transcript analyzer - extracts key points from transcripts for document generation.

Uses existing Qwen3-4B LLM to analyze transcripts and identify:
- Key steps and actions
- Moments that need screenshots (deictic words: "here", "this")
- Operations that need before/after comparison (action words: "click", "apply")
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, List, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from loguru import logger

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


class KeyPointType(str, Enum):
    """Types of key points extracted from transcript."""
    ACTION = "action"  # User performed an action (click, type, etc.)
    OBSERVATION = "observation"  # User observed/noticed something
    INSTRUCTION = "instruction"  # User gave instruction/explanation
    DISCOVERY = "discovery"  # User discovered something important


@dataclass
class KeyPoint:
    """A key point extracted from the transcript."""
    text: str  # The extracted key text
    start_time: float  # Start timestamp in seconds
    end_time: float  # End timestamp in seconds
    type: KeyPointType  # Type of key point
    needs_screenshot: bool = False  # Should extract a frame at this timestamp
    needs_comparison: bool = False  # Should extract before/after frames
    mouse_position: Optional[tuple] = None  # (x, y) if available from events
    segment_ids: List[int] = field(default_factory=list)  # Source segment IDs

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["type"] = self.type.value
        return d


@dataclass
class TranscriptData:
    """Loaded transcript data."""
    segments: List[dict]
    total_duration: float
    language: Optional[str] = None


class TranscriptAnalyzer:
    """Analyzes transcripts to extract key points for document generation."""

    # Words/phrases indicating need for screenshot (deictic references)
    DEICTIC_PATTERNS_EN = [
        r"\bhere\b", r"\bthis\b", r"\bthat\b", r"\bthese\b", r"\bthose\b",
        r"\bthis one\b", r"\bright here\b", r"\bover here\b", r"\bsee\b",
        r"\blook at\b", r"\bnotice\b", r"\byou can see\b",
    ]

    DEICTIC_PATTERNS_ZH = [
        r"这里", r"这个", r"那个", r"这些", r"那些",
        r"看这里", r"看这个", r"你看", r"可以看到", r"注意",
    ]

    # Words/phrases indicating operations that need before/after comparison
    ACTION_PATTERNS_EN = [
        r"\bclick\b", r"\bpress\b", r"\bapply\b", r"\bchange\b",
        r"\bmodify\b", r"\badjust\b", r"\bset\b", r"\benable\b",
        r"\bdisable\b", r"\bturn on\b", r"\bturn off\b", r"\bselect\b",
        r"\bopen\b", r"\bclose\b", r"\bstart\b", r"\bstop\b",
    ]

    ACTION_PATTERNS_ZH = [
        r"点击", r"按", r"应用", r"更改", r"修改", r"调整",
        r"设置", r"启用", r"禁用", r"打开", r"关闭",
        r"开始", r"停止", r"选择", r"点一下",
    ]

    # System prompt for key point extraction
    EXTRACTION_PROMPT = """You are a transcript analyzer. Extract key instructional points from the transcript.

For each key point, determine:
1. Type: action (user did something), observation (user noticed something), instruction (user explained something), discovery (important finding)
2. Whether it needs a screenshot (references "here", "this", "look at", etc.)
3. Whether it needs before/after comparison (involves clicking, applying, changing something)

Input: Transcript segments with timestamps
Output: JSON array of key points

Rules:
- Focus on instructive content, skip filler and repetition
- Keep original language (don't translate)
- Merge related consecutive segments into single key points
- Mark significant UI operations for before/after screenshots
- Mark deictic references ("here", "this") for screenshots

Output format (JSON array):
[
  {
    "text": "concise summary of the key point",
    "start_time": 10.5,
    "end_time": 15.2,
    "type": "action|observation|instruction|discovery",
    "needs_screenshot": true/false,
    "needs_comparison": true/false,
    "segment_ids": [1, 2, 3]
  }
]"""

    def __init__(self, model: Optional[Llama] = None):
        """Initialize the transcript analyzer.

        Args:
            model: Existing Llama model instance (shared with LLM processor)
        """
        self._model = model
        self._all_deictic_patterns = (
            [re.compile(p, re.IGNORECASE) for p in self.DEICTIC_PATTERNS_EN] +
            [re.compile(p) for p in self.DEICTIC_PATTERNS_ZH]
        )
        self._all_action_patterns = (
            [re.compile(p, re.IGNORECASE) for p in self.ACTION_PATTERNS_EN] +
            [re.compile(p) for p in self.ACTION_PATTERNS_ZH]
        )

    def set_model(self, model: Llama):
        """Set the LLM model for analysis."""
        self._model = model

    def load_transcript(self, transcript_path: Path) -> Optional[TranscriptData]:
        """Load transcript from JSON file.

        Args:
            transcript_path: Path to transcript.json

        Returns:
            TranscriptData object, or None if loading failed
        """
        if not transcript_path.exists():
            logger.warning(f"Transcript file not found: {transcript_path}")
            return None

        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            segments = data if isinstance(data, list) else data.get("segments", [])

            if not segments:
                logger.warning("No segments found in transcript")
                return None

            # Calculate total duration
            total_duration = max(seg.get("end", 0) for seg in segments) if segments else 0

            # Detect primary language
            language = None
            if segments:
                langs = [seg.get("language") for seg in segments if seg.get("language")]
                if langs:
                    from collections import Counter
                    language = Counter(langs).most_common(1)[0][0]

            return TranscriptData(
                segments=segments,
                total_duration=total_duration,
                language=language,
            )

        except Exception as e:
            logger.error(f"Failed to load transcript: {e}")
            return None

    def _has_deictic_reference(self, text: str) -> bool:
        """Check if text contains deictic references (need screenshot)."""
        for pattern in self._all_deictic_patterns:
            if pattern.search(text):
                return True
        return False

    def _has_action_reference(self, text: str) -> bool:
        """Check if text contains action references (need before/after)."""
        for pattern in self._all_action_patterns:
            if pattern.search(text):
                return True
        return False

    def analyze_with_llm(
        self,
        transcript_data: TranscriptData,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[KeyPoint]:
        """Analyze transcript using LLM to extract key points.

        Args:
            transcript_data: Loaded transcript data
            progress_callback: Optional callback (current, total, description)

        Returns:
            List of KeyPoint objects
        """
        if not self._model:
            logger.warning("No LLM model available, using rule-based extraction")
            return self.analyze_rule_based(transcript_data, progress_callback)

        if progress_callback:
            progress_callback(0, 100, "Preparing transcript for analysis...")

        # Prepare transcript text for LLM
        segments = transcript_data.segments
        transcript_text = self._format_segments_for_llm(segments)

        if progress_callback:
            progress_callback(10, 100, "Analyzing transcript with LLM...")

        # Build prompt
        prompt = f"""<|im_start|>system
{self.EXTRACTION_PROMPT}<|im_end|>
<|im_start|>user
Analyze this transcript and extract key instructional points:

{transcript_text}<|im_end|>
<|im_start|>assistant
"""

        try:
            response = self._model(
                prompt,
                max_tokens=4096,
                temperature=0.3,
                top_p=0.9,
                stop=["<|im_end|>", "<|im_start|>"],
                echo=False,
            )

            if progress_callback:
                progress_callback(70, 100, "Parsing LLM response...")

            result_text = response["choices"][0]["text"].strip()
            key_points = self._parse_llm_response(result_text, segments)

            if progress_callback:
                progress_callback(90, 100, "Validating key points...")

            # Post-process: apply rule-based checks for screenshot/comparison needs
            for kp in key_points:
                if not kp.needs_screenshot:
                    kp.needs_screenshot = self._has_deictic_reference(kp.text)
                if not kp.needs_comparison:
                    kp.needs_comparison = self._has_action_reference(kp.text)

            if progress_callback:
                progress_callback(100, 100, f"Extracted {len(key_points)} key points")

            logger.info(f"Extracted {len(key_points)} key points using LLM")
            return key_points

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}, falling back to rule-based")
            return self.analyze_rule_based(transcript_data, progress_callback)

    def _format_segments_for_llm(self, segments: List[dict], max_chars: int = 8000) -> str:
        """Format transcript segments for LLM input.

        Args:
            segments: List of segment dictionaries
            max_chars: Maximum characters to include

        Returns:
            Formatted transcript text
        """
        lines = []
        total_chars = 0

        for seg in segments:
            seg_id = seg.get("id", 0)
            start = seg.get("start", 0)
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker", "")

            if not text:
                continue

            # Format: [ID:1 T:10.5s Speaker A] Text here
            speaker_str = f" {speaker}" if speaker else ""
            line = f"[ID:{seg_id} T:{start:.1f}s{speaker_str}] {text}"

            if total_chars + len(line) > max_chars:
                lines.append("... (transcript truncated)")
                break

            lines.append(line)
            total_chars += len(line) + 1

        return "\n".join(lines)

    def _parse_llm_response(self, response: str, segments: List[dict]) -> List[KeyPoint]:
        """Parse LLM response into KeyPoint objects.

        Args:
            response: LLM response text (expected JSON array)
            segments: Original transcript segments for fallback timestamps

        Returns:
            List of KeyPoint objects
        """
        key_points = []

        # Try to extract JSON from response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if not json_match:
            logger.warning("No JSON array found in LLM response")
            return key_points

        try:
            items = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON: {e}")
            return key_points

        # Create segment ID to segment map for timestamp lookup
        seg_map = {seg.get("id"): seg for seg in segments}

        for item in items:
            if not isinstance(item, dict):
                continue

            text = item.get("text", "").strip()
            if not text:
                continue

            # Get timestamps
            start_time = item.get("start_time", 0)
            end_time = item.get("end_time", start_time + 1)

            # Fallback: use segment IDs to get timestamps
            segment_ids = item.get("segment_ids", [])
            if segment_ids and (start_time == 0 or end_time == 0):
                seg_starts = [seg_map.get(sid, {}).get("start", 0) for sid in segment_ids if sid in seg_map]
                seg_ends = [seg_map.get(sid, {}).get("end", 0) for sid in segment_ids if sid in seg_map]
                if seg_starts:
                    start_time = min(seg_starts)
                if seg_ends:
                    end_time = max(seg_ends)

            # Parse type
            type_str = item.get("type", "instruction").lower()
            try:
                kp_type = KeyPointType(type_str)
            except ValueError:
                kp_type = KeyPointType.INSTRUCTION

            key_points.append(KeyPoint(
                text=text,
                start_time=start_time,
                end_time=end_time,
                type=kp_type,
                needs_screenshot=item.get("needs_screenshot", False),
                needs_comparison=item.get("needs_comparison", False),
                segment_ids=segment_ids,
            ))

        return key_points

    def analyze_rule_based(
        self,
        transcript_data: TranscriptData,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[KeyPoint]:
        """Fallback rule-based analysis when LLM is unavailable.

        Args:
            transcript_data: Loaded transcript data
            progress_callback: Optional callback (current, total, description)

        Returns:
            List of KeyPoint objects
        """
        if progress_callback:
            progress_callback(0, 100, "Starting rule-based analysis...")

        segments = transcript_data.segments
        key_points = []

        # Group consecutive segments and detect key points
        current_group = []

        for i, seg in enumerate(segments):
            if progress_callback:
                progress_callback(
                    int((i / len(segments)) * 80),
                    100,
                    f"Analyzing segment {i + 1}/{len(segments)}"
                )

            text = seg.get("text", "").strip()
            if not text or len(text) < 5:
                continue

            # Check for key indicators
            has_deictic = self._has_deictic_reference(text)
            has_action = self._has_action_reference(text)

            # If significant content, create a key point
            if has_deictic or has_action or len(text) > 30:
                # Determine type based on content
                if has_action:
                    kp_type = KeyPointType.ACTION
                elif has_deictic:
                    kp_type = KeyPointType.OBSERVATION
                else:
                    kp_type = KeyPointType.INSTRUCTION

                key_points.append(KeyPoint(
                    text=text,
                    start_time=seg.get("start", 0),
                    end_time=seg.get("end", 0),
                    type=kp_type,
                    needs_screenshot=has_deictic,
                    needs_comparison=has_action,
                    segment_ids=[seg.get("id", i)],
                ))

        if progress_callback:
            progress_callback(90, 100, "Merging adjacent key points...")

        # Merge adjacent key points of the same type
        merged = self._merge_adjacent_keypoints(key_points)

        if progress_callback:
            progress_callback(100, 100, f"Extracted {len(merged)} key points")

        logger.info(f"Extracted {len(merged)} key points using rule-based analysis")
        return merged

    def _merge_adjacent_keypoints(
        self,
        key_points: List[KeyPoint],
        time_threshold: float = 3.0,
    ) -> List[KeyPoint]:
        """Merge adjacent key points that are close in time.

        Args:
            key_points: List of key points
            time_threshold: Maximum gap (seconds) to merge

        Returns:
            Merged list of key points
        """
        if not key_points:
            return []

        merged = [key_points[0]]

        for kp in key_points[1:]:
            prev = merged[-1]

            # Merge if same type and close in time
            if (kp.type == prev.type and
                kp.start_time - prev.end_time < time_threshold):
                # Merge into previous
                prev.text = f"{prev.text} {kp.text}"
                prev.end_time = kp.end_time
                prev.needs_screenshot = prev.needs_screenshot or kp.needs_screenshot
                prev.needs_comparison = prev.needs_comparison or kp.needs_comparison
                prev.segment_ids.extend(kp.segment_ids)
            else:
                merged.append(kp)

        return merged

    def save_key_points(self, key_points: List[KeyPoint], output_path: Path):
        """Save extracted key points to JSON file.

        Args:
            key_points: List of KeyPoint objects
            output_path: Path to save the JSON file
        """
        try:
            data = [kp.to_dict() for kp in key_points]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(key_points)} key points to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save key points: {e}")


def create_transcript_analyzer(model: Optional[Llama] = None) -> TranscriptAnalyzer:
    """Factory function to create a transcript analyzer.

    Args:
        model: Optional Llama model instance

    Returns:
        TranscriptAnalyzer instance
    """
    return TranscriptAnalyzer(model=model)
