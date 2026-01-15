"""Claude Vision processor - uses Claude API for image analysis and smart cropping.

Replaces local VLM (Qwen3-VL-8B) with Claude's vision capabilities for:
1. Smart cropping - identify relevant UI regions in screenshots
2. Change detection - compare before/after images to find differences
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from .config import get_config_manager

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    anthropic = None


@dataclass
class CropRegion:
    """Region to crop from an image."""
    x: int  # Left coordinate
    y: int  # Top coordinate
    width: int  # Width of region
    height: int  # Height of region
    description: str = ""  # Description of what's in the region


@dataclass
class ChangeDetectionResult:
    """Result of change detection between two images."""
    is_significant: bool  # Whether there's a significant change
    changed_region: Optional[CropRegion]  # Region where change occurred
    change_description: str  # Description of the change


class ClaudeVisionProcessor:
    """Uses Claude API for image analysis and smart cropping."""

    SMART_CROP_PROMPT = """Analyze this screenshot and identify the most relevant UI region based on the context.

Context: {context}
{mouse_info}

The image dimensions are {width}x{height} pixels.

Return a JSON object with the crop region that best captures the relevant UI element or area:
{{
    "x": <left coordinate>,
    "y": <top coordinate>,
    "width": <region width>,
    "height": <region height>,
    "description": "<brief description of what's in this region>"
}}

Guidelines:
- Focus on the area most relevant to the context description
- If mouse position is given, include that area
- Include enough context around the target (don't crop too tight)
- Minimum size should be 512x512 pixels
- Maximum size should be 1200x900 pixels for readability
- Return ONLY the JSON object, no other text"""

    CHANGE_DETECTION_PROMPT = """Compare these two screenshots (before and after) and identify what changed.

Context: {context}
{mouse_info}

The images are {width}x{height} pixels.

Analyze the differences and return a JSON object:
{{
    "is_significant": true/false,
    "changed_region": {{
        "x": <left coordinate of changed area>,
        "y": <top coordinate>,
        "width": <region width>,
        "height": <region height>
    }},
    "change_description": "<brief description of what changed>"
}}

Guidelines:
- Include enough context around the changed area (don't crop too tight)
- MINIMUM region size: 512x512 pixels
- MAXIMUM region size: 1200x900 pixels
- If mouse position is given, ensure that area is included in the region
- Focus on UI elements that changed (menus, dialogs, buttons, values)

If no significant change is detected, set is_significant to false and changed_region to null.
Return ONLY the JSON object, no other text."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Claude vision processor.

        Args:
            api_key: Anthropic API key. If None, reads from config.
        """
        self._api_key = api_key
        self._client: Optional[anthropic.Anthropic] = None

        if not self._api_key:
            config = get_config_manager()
            self._api_key = config.get_anthropic_api_key()

        if HAS_ANTHROPIC and self._api_key:
            self._client = anthropic.Anthropic(api_key=self._api_key)
            logger.info("Claude vision processor initialized")
        else:
            if not HAS_ANTHROPIC:
                logger.warning("anthropic package not installed")
            if not self._api_key:
                logger.warning("No Anthropic API key configured")

    @property
    def is_available(self) -> bool:
        """Check if Claude vision is available."""
        return HAS_ANTHROPIC and HAS_PIL and self._client is not None

    def _encode_image(self, image_path: Path) -> Tuple[str, str]:
        """Encode image to base64 for Claude API.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (base64_data, media_type)
        """
        suffix = image_path.suffix.lower()
        if suffix in [".jpg", ".jpeg"]:
            media_type = "image/jpeg"
        elif suffix == ".png":
            media_type = "image/png"
        elif suffix == ".gif":
            media_type = "image/gif"
        elif suffix == ".webp":
            media_type = "image/webp"
        else:
            # Convert to JPEG
            media_type = "image/jpeg"
            img = Image.open(image_path)
            import io
            buffer = io.BytesIO()
            img.convert("RGB").save(buffer, format="JPEG", quality=85)
            return base64.standard_b64encode(buffer.getvalue()).decode("utf-8"), media_type

        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8"), media_type

    def _get_image_size(self, image_path: Path) -> Tuple[int, int]:
        """Get image dimensions.

        Args:
            image_path: Path to image

        Returns:
            Tuple of (width, height)
        """
        with Image.open(image_path) as img:
            return img.size

    def smart_crop(
        self,
        image_path: Path,
        context: str = "",
        mouse_position: Optional[Tuple[int, int]] = None,
    ) -> CropRegion:
        """Use Claude to identify the most relevant region to crop.

        Args:
            image_path: Path to screenshot
            context: Text context describing what to focus on
            mouse_position: Optional (x, y) mouse position

        Returns:
            CropRegion with coordinates and description
        """
        if not self.is_available:
            logger.warning("Claude vision not available, returning full image")
            width, height = self._get_image_size(image_path)
            return CropRegion(0, 0, width, height, "Full image")

        try:
            width, height = self._get_image_size(image_path)
            image_data, media_type = self._encode_image(image_path)

            mouse_info = ""
            if mouse_position:
                mouse_info = f"Mouse cursor is at position ({mouse_position[0]}, {mouse_position[1]})."

            prompt = self.SMART_CROP_PROMPT.format(
                context=context or "General UI interaction",
                mouse_info=mouse_info,
                width=width,
                height=height,
            )

            response = self._client.messages.create(
                model="claude-opus-4-5-20251101",
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )

            result_text = response.content[0].text.strip()
            return self._parse_crop_response(result_text, width, height)

        except Exception as e:
            logger.error(f"Claude smart crop failed: {e}")
            width, height = self._get_image_size(image_path)
            return CropRegion(0, 0, width, height, "Full image (fallback)")

    def _parse_crop_response(
        self,
        response: str,
        image_width: int,
        image_height: int,
    ) -> CropRegion:
        """Parse Claude's crop response.

        Args:
            response: Claude's response text
            image_width: Original image width
            image_height: Original image height

        Returns:
            CropRegion object
        """
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                raise ValueError("No JSON found in response")

            data = json.loads(json_match.group())

            x = max(0, int(data.get("x", 0)))
            y = max(0, int(data.get("y", 0)))
            w = int(data.get("width", image_width))
            h = int(data.get("height", image_height))

            # Ensure minimum size (512x512)
            min_size = 512
            w = max(w, min_size)
            h = max(h, min_size)

            # Clamp to image bounds (ensure room for minimum size)
            x = min(x, max(0, image_width - min_size))
            y = min(y, max(0, image_height - min_size))
            w = min(w, image_width - x)
            h = min(h, image_height - y)

            return CropRegion(
                x=x,
                y=y,
                width=w,
                height=h,
                description=data.get("description", ""),
            )

        except Exception as e:
            logger.error(f"Failed to parse crop response: {e}")
            return CropRegion(0, 0, image_width, image_height, "Full image")

    def detect_changes(
        self,
        before_image: Path,
        after_image: Path,
        context: str = "",
        mouse_position: Optional[Tuple[int, int]] = None,
    ) -> ChangeDetectionResult:
        """Detect changes between two images using Claude.

        Args:
            before_image: Path to before screenshot
            after_image: Path to after screenshot
            context: Text context describing expected change
            mouse_position: Optional (x, y) mouse position

        Returns:
            ChangeDetectionResult with change information
        """
        if not self.is_available:
            logger.warning("Claude vision not available")
            return ChangeDetectionResult(
                is_significant=False,
                changed_region=None,
                change_description="Vision analysis not available",
            )

        try:
            width, height = self._get_image_size(before_image)
            before_data, before_type = self._encode_image(before_image)
            after_data, after_type = self._encode_image(after_image)

            mouse_info = ""
            if mouse_position:
                mouse_info = f"Mouse cursor is at position ({mouse_position[0]}, {mouse_position[1]})."

            prompt = self.CHANGE_DETECTION_PROMPT.format(
                context=context or "UI operation",
                mouse_info=mouse_info,
                width=width,
                height=height,
            )

            response = self._client.messages.create(
                model="claude-opus-4-5-20251101",
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Before image:",
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": before_type,
                                    "data": before_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": "After image:",
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": after_type,
                                    "data": after_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )

            result_text = response.content[0].text.strip()
            return self._parse_change_response(result_text, width, height)

        except Exception as e:
            logger.error(f"Claude change detection failed: {e}")
            return ChangeDetectionResult(
                is_significant=False,
                changed_region=None,
                change_description=f"Error: {str(e)}",
            )

    def _parse_change_response(
        self,
        response: str,
        image_width: int,
        image_height: int,
    ) -> ChangeDetectionResult:
        """Parse Claude's change detection response.

        Args:
            response: Claude's response text
            image_width: Image width
            image_height: Image height

        Returns:
            ChangeDetectionResult object
        """
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                raise ValueError("No JSON found in response")

            data = json.loads(json_match.group())

            is_significant = data.get("is_significant", False)
            change_description = data.get("change_description", "")

            changed_region = None
            if is_significant and data.get("changed_region"):
                region = data["changed_region"]
                x = max(0, int(region.get("x", 0)))
                y = max(0, int(region.get("y", 0)))
                w = int(region.get("width", image_width))
                h = int(region.get("height", image_height))

                # Enforce minimum size (512x512)
                min_w, min_h = 512, 512
                if w < min_w:
                    # Expand width, center around original
                    expand = (min_w - w) // 2
                    x = max(0, x - expand)
                    w = min_w
                if h < min_h:
                    # Expand height, center around original
                    expand = (min_h - h) // 2
                    y = max(0, y - expand)
                    h = min_h

                # Clamp to image bounds
                x = min(x, image_width - min_w)
                y = min(y, image_height - min_h)
                w = min(w, image_width - x)
                h = min(h, image_height - y)

                changed_region = CropRegion(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    description=change_description,
                )

            return ChangeDetectionResult(
                is_significant=is_significant,
                changed_region=changed_region,
                change_description=change_description,
            )

        except Exception as e:
            logger.error(f"Failed to parse change response: {e}")
            return ChangeDetectionResult(
                is_significant=False,
                changed_region=None,
                change_description="Failed to parse response",
            )

    def crop_image(
        self,
        image_path: Path,
        region: CropRegion,
        output_path: Path,
        quality: int = 85,
    ) -> bool:
        """Crop an image to the specified region.

        Args:
            image_path: Source image path
            region: Region to crop
            output_path: Output path for cropped image
            quality: JPEG quality (1-100)

        Returns:
            True if successful
        """
        if not HAS_PIL:
            logger.error("PIL not available for cropping")
            return False

        try:
            with Image.open(image_path) as img:
                # Calculate crop box (left, upper, right, lower)
                box = (
                    region.x,
                    region.y,
                    region.x + region.width,
                    region.y + region.height,
                )

                # Ensure box is within image bounds
                box = (
                    max(0, box[0]),
                    max(0, box[1]),
                    min(img.width, box[2]),
                    min(img.height, box[3]),
                )

                cropped = img.crop(box)

                # Save as JPEG
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if cropped.mode in ("RGBA", "P"):
                    cropped = cropped.convert("RGB")
                cropped.save(output_path, "JPEG", quality=quality)

                logger.debug(f"Cropped image saved to {output_path}")
                return True

        except Exception as e:
            logger.error(f"Failed to crop image: {e}")
            return False


# Singleton instance
_claude_vision_instance: Optional[ClaudeVisionProcessor] = None


def get_claude_vision_processor() -> Optional[ClaudeVisionProcessor]:
    """Get the singleton Claude vision processor.

    Returns:
        ClaudeVisionProcessor instance, or None if not available
    """
    global _claude_vision_instance
    if _claude_vision_instance is None:
        _claude_vision_instance = ClaudeVisionProcessor()
    if _claude_vision_instance.is_available:
        return _claude_vision_instance
    return None


def is_claude_vision_available() -> bool:
    """Check if Claude vision is available.

    Returns:
        True if Claude API key is configured and anthropic is installed
    """
    if not HAS_ANTHROPIC or not HAS_PIL:
        return False

    config = get_config_manager()
    api_key = config.get_anthropic_api_key()
    return bool(api_key)
