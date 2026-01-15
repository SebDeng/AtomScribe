"""Claude-based transcript analyzer - uses Claude API for document generation.

This module provides an alternative to the local LLM (Qwen3-4B) for analyzing
transcripts and generating documentation. It uses the Anthropic Claude API
for higher quality analysis and more sophisticated document generation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any
from dataclasses import asdict
from loguru import logger

from .transcript_analyzer import (
    KeyPoint,
    KeyPointType,
    TranscriptData,
    TranscriptAnalyzer,
)
from .config import get_config_manager

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    anthropic = None


class ClaudeTranscriptAnalyzer(TranscriptAnalyzer):
    """Transcript analyzer using Claude API.

    Extends the base TranscriptAnalyzer to use Claude for:
    1. Key point extraction from transcripts
    2. Enhanced document generation with better summaries
    """

    # System prompt for key point extraction (optimized for Claude)
    CLAUDE_EXTRACTION_PROMPT = """You are an expert transcript analyzer for technical tutorials and lab recordings.

Your task is to analyze a transcript and extract key instructional points that would be valuable for creating a step-by-step tutorial or documentation.

For each key point, determine:
1. **Type**:
   - action: User performed a specific operation (clicking, typing, adjusting settings)
   - observation: User observed or noticed something important
   - instruction: User explained or taught something
   - discovery: User made an important finding or realization

2. **needs_screenshot**: Set to true if the text references something visible that needs a screenshot:
   - Deictic words: "here", "this", "that", "see this", "look at"
   - Visual references: "you can see", "notice", "observe"

3. **needs_comparison**: Set to true if the text describes a state change that needs before/after screenshots:
   - Action verbs: "click", "apply", "change", "adjust", "enable", "disable"
   - Transformations: "becomes", "turns into", "changes to"

Rules:
- Focus on instructive content - skip filler words and repetition
- Keep the original language (Chinese, English, or mixed) - do not translate
- Merge related consecutive segments into coherent key points
- Extract concise but complete descriptions
- Preserve technical terminology accurately

Output format: Return ONLY a valid JSON array, no other text:
[
  {
    "text": "Concise description of the key point in original language",
    "start_time": 10.5,
    "end_time": 15.2,
    "type": "action",
    "needs_screenshot": true,
    "needs_comparison": false,
    "segment_ids": [1, 2, 3]
  }
]"""

    # System prompt for document enhancement
    CLAUDE_DOCUMENT_PROMPT = """You are a technical documentation expert. Given extracted key points from a tutorial/lab session, create a polished, professional document.

Guidelines:
- Write clear, step-by-step instructions
- CRITICAL: You MUST write the document in the SAME language as the key points. If key points are in English, write in English. If in Chinese, write in Chinese. Do NOT translate.
- Add helpful context and explanations
- Organize logically with clear section headings
- Include tips and warnings where appropriate
- Format for readability with proper markdown

Output a well-structured markdown document in the SAME language as the input."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Claude-based analyzer.

        Args:
            api_key: Anthropic API key. If None, reads from config.
        """
        super().__init__(model=None)  # Don't use local LLM

        self._api_key = api_key
        self._client: Optional[anthropic.Anthropic] = None

        if not self._api_key:
            config = get_config_manager()
            self._api_key = config.get_anthropic_api_key()

        if HAS_ANTHROPIC and self._api_key:
            self._client = anthropic.Anthropic(api_key=self._api_key)
            logger.info("Claude transcript analyzer initialized")
        else:
            if not HAS_ANTHROPIC:
                logger.warning("anthropic package not installed")
            if not self._api_key:
                logger.warning("No Anthropic API key configured")

    @property
    def is_available(self) -> bool:
        """Check if Claude API is available."""
        return HAS_ANTHROPIC and self._client is not None

    def analyze_with_llm(
        self,
        transcript_data: TranscriptData,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[KeyPoint]:
        """Analyze transcript using Claude API.

        Args:
            transcript_data: Loaded transcript data
            progress_callback: Optional callback (current, total, description)

        Returns:
            List of KeyPoint objects
        """
        if not self.is_available:
            logger.warning("Claude API not available, falling back to rule-based")
            return self.analyze_rule_based(transcript_data, progress_callback)

        if progress_callback:
            progress_callback(0, 100, "Preparing transcript for Claude analysis...")

        # Format transcript for Claude
        segments = transcript_data.segments
        transcript_text = self._format_segments_for_claude(segments)

        if progress_callback:
            progress_callback(10, 100, "Sending to Claude API...")

        try:
            response = self._client.messages.create(
                model="claude-opus-4-5-20251101",
                max_tokens=4096,
                system=self.CLAUDE_EXTRACTION_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": f"Analyze this transcript and extract key instructional points:\n\n{transcript_text}"
                    }
                ]
            )

            if progress_callback:
                progress_callback(70, 100, "Parsing Claude response...")

            result_text = response.content[0].text.strip()
            key_points = self._parse_claude_response(result_text, segments)

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

            logger.info(f"Extracted {len(key_points)} key points using Claude")
            return key_points

        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            return self.analyze_rule_based(transcript_data, progress_callback)
        except Exception as e:
            logger.error(f"Claude analysis failed: {e}, falling back to rule-based")
            return self.analyze_rule_based(transcript_data, progress_callback)

    def _format_segments_for_claude(
        self,
        segments: List[dict],
        max_chars: int = 30000,
    ) -> str:
        """Format transcript segments for Claude input.

        Claude has larger context, so we can include more content.

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
            end = seg.get("end", start)
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker", "")

            if not text:
                continue

            # Format: [ID:1 T:10.5-15.2s Speaker A] Text here
            speaker_str = f" {speaker}" if speaker else ""
            line = f"[ID:{seg_id} T:{start:.1f}-{end:.1f}s{speaker_str}] {text}"

            if total_chars + len(line) > max_chars:
                lines.append("... (transcript truncated due to length)")
                break

            lines.append(line)
            total_chars += len(line) + 1

        return "\n".join(lines)

    def _parse_claude_response(
        self,
        response: str,
        segments: List[dict],
    ) -> List[KeyPoint]:
        """Parse Claude response into KeyPoint objects.

        Args:
            response: Claude response text (expected JSON array)
            segments: Original transcript segments for fallback timestamps

        Returns:
            List of KeyPoint objects
        """
        key_points = []

        # Try to extract JSON from response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if not json_match:
            logger.warning("No JSON array found in Claude response")
            logger.debug(f"Response was: {response[:500]}")
            return key_points

        try:
            items = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude JSON: {e}")
            logger.debug(f"JSON string was: {json_match.group()[:500]}")
            return key_points

        # Create segment ID to segment map
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
                seg_starts = [
                    seg_map.get(sid, {}).get("start", 0)
                    for sid in segment_ids if sid in seg_map
                ]
                seg_ends = [
                    seg_map.get(sid, {}).get("end", 0)
                    for sid in segment_ids if sid in seg_map
                ]
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

    def enhance_document(
        self,
        key_points: List[KeyPoint],
        mode: str = "training",
        title: str = "Document",
        duration: float = 0,
        images: Optional[Dict[int, Any]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Optional[str]:
        """Use Claude to generate an enhanced markdown document.

        Args:
            key_points: Extracted key points
            mode: Document mode ("training" or "experiment_log")
            title: Document title
            duration: Recording duration in seconds
            images: Dict mapping key point index to image references
                   Format: {idx: {"main": ImageRef, "before": ImageRef, "after": ImageRef}}
            progress_callback: Optional progress callback

        Returns:
            Enhanced markdown content, or None if failed
        """
        if not self.is_available:
            logger.warning("Claude API not available for document enhancement")
            return None

        if progress_callback:
            progress_callback(0, 100, "Preparing content for Claude...")

        images = images or {}

        # Format key points with their associated images
        kp_data = []
        for i, kp in enumerate(key_points):
            kp_dict = kp.to_dict()
            kp_dict["index"] = i

            # Add image paths if available
            if i in images:
                img_info = images[i]
                kp_dict["images"] = {}
                if "main" in img_info:
                    kp_dict["images"]["main"] = img_info["main"].relative_path
                if "before" in img_info:
                    kp_dict["images"]["before"] = img_info["before"].relative_path
                if "after" in img_info:
                    kp_dict["images"]["after"] = img_info["after"].relative_path

            kp_data.append(kp_dict)

        key_points_json = json.dumps(kp_data, ensure_ascii=False, indent=2)

        # Build prompt based on mode
        if mode == "training":
            mode_instruction = """Create a step-by-step training tutorial document.
Structure:
1. Overview section summarizing what will be learned
2. Prerequisites if applicable
3. Numbered steps with clear instructions
4. Tips and important notes highlighted
5. Summary of what was covered"""
        else:
            mode_instruction = """Create an experiment log document.
Structure:
1. Session information (date, duration)
2. Key findings/discoveries highlighted at the top
3. Chronological log of observations and actions
4. Conclusions and next steps"""

        prompt = f"""Title: {title}
Duration: {self._format_duration(duration)}
Mode: {mode}

Key points extracted from the recording (with actual image paths if available):
{key_points_json}

{mode_instruction}

Generate the markdown document. Use the original language of the key points (do not translate).

IMPORTANT - Image handling rules:
- Each key point may have an "images" field with actual image paths
- If a key point has "images.main", include it: ![Description](path_from_images.main)
- If a key point has "images.before" and "images.after", show them as a comparison
- ONLY use the exact paths provided in the "images" field
- Do NOT invent or guess image paths - only use paths from the data
- If no images field exists for a key point, do not include any image for it"""

        if progress_callback:
            progress_callback(20, 100, "Generating document with Claude...")

        try:
            response = self._client.messages.create(
                model="claude-opus-4-5-20251101",
                max_tokens=8192,
                system=self.CLAUDE_DOCUMENT_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            if progress_callback:
                progress_callback(90, 100, "Processing response...")

            content = response.content[0].text.strip()

            if progress_callback:
                progress_callback(100, 100, "Document generated")

            return content

        except Exception as e:
            logger.error(f"Claude document enhancement failed: {e}")
            return None

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    def translate_document(
        self,
        content: str,
        target_language: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Optional[str]:
        """Translate a markdown document to another language.

        Args:
            content: Original markdown content
            target_language: Target language ("en" for English, "zh" for Chinese)
            progress_callback: Optional progress callback

        Returns:
            Translated markdown content, or None if failed
        """
        if not self.is_available:
            logger.warning("Claude API not available for translation")
            return None

        if progress_callback:
            progress_callback(0, 100, f"Translating to {target_language}...")

        lang_name = "English" if target_language == "en" else "Chinese (简体中文)"

        prompt = f"""Translate this markdown document to {lang_name}.

Rules:
- Translate ALL text content to {lang_name}
- Keep all markdown formatting intact (headers, lists, tables, bold, etc.)
- Keep all image paths exactly as they are (do not modify ![...](...) paths)
- Keep technical terms that are commonly used in English (e.g., Gaussian filter, Fourier transform)
- Maintain the same document structure

Original document:
{content}

Output the translated markdown document only, no explanations."""

        try:
            if progress_callback:
                progress_callback(20, 100, "Calling Claude API...")

            response = self._client.messages.create(
                model="claude-opus-4-5-20251101",
                max_tokens=8192,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            if progress_callback:
                progress_callback(90, 100, "Processing response...")

            translated = response.content[0].text.strip()

            if progress_callback:
                progress_callback(100, 100, "Translation complete")

            return translated

        except Exception as e:
            logger.error(f"Claude translation failed: {e}")
            return None

    def detect_language(self, content: str) -> str:
        """Detect the primary language of content.

        Args:
            content: Text content to analyze

        Returns:
            Language code: "zh" for Chinese, "en" for English
        """
        # Simple heuristic: count Chinese characters
        chinese_chars = sum(1 for c in content if '\u4e00' <= c <= '\u9fff')
        total_chars = len(content.replace(" ", "").replace("\n", ""))

        if total_chars == 0:
            return "en"

        chinese_ratio = chinese_chars / total_chars
        return "zh" if chinese_ratio > 0.1 else "en"


def create_claude_analyzer(api_key: Optional[str] = None) -> Optional[ClaudeTranscriptAnalyzer]:
    """Factory function to create a Claude-based analyzer.

    Args:
        api_key: Optional API key (reads from config if None)

    Returns:
        ClaudeTranscriptAnalyzer instance, or None if not available
    """
    analyzer = ClaudeTranscriptAnalyzer(api_key=api_key)
    if analyzer.is_available:
        return analyzer
    return None


def is_claude_available() -> bool:
    """Check if Claude API is available.

    Returns:
        True if anthropic package is installed and API key is configured
    """
    if not HAS_ANTHROPIC:
        return False

    config = get_config_manager()
    api_key = config.get_anthropic_api_key()
    return bool(api_key)
