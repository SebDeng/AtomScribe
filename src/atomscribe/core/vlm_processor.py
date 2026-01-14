"""VLM processor - smart cropping and change detection using Qwen3-VL.

Uses llama.cpp server for vision-language model inference.
Can auto-start the server when needed if model files are available.

Tasks:
- Smart cropping: Identify the most instructive region to crop from a screenshot
- Change detection: Find what changed between before/after frames
"""

from __future__ import annotations

import atexit
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
import requests
from loguru import logger

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None


@dataclass
class CropRegion:
    """Represents a crop region in an image."""
    x: int
    y: int
    width: int
    height: int
    description: str = ""
    ui_element: str = ""
    confidence: float = 0.0

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return as (left, top, right, bottom) for PIL cropping."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def is_valid(self, image_width: int, image_height: int) -> bool:
        """Check if crop region is valid within image bounds."""
        return (
            self.x >= 0 and self.y >= 0 and
            self.width > 0 and self.height > 0 and
            self.x + self.width <= image_width and
            self.y + self.height <= image_height
        )

    def clamp(self, image_width: int, image_height: int) -> "CropRegion":
        """Clamp crop region to fit within image bounds."""
        # Clamp x and y to valid range
        x = max(0, min(self.x, image_width - 1))
        y = max(0, min(self.y, image_height - 1))

        # Ensure width and height don't exceed image bounds
        width = min(self.width, image_width - x)
        height = min(self.height, image_height - y)

        # Ensure minimum size
        width = max(100, width)
        height = max(100, height)

        return CropRegion(
            x=x,
            y=y,
            width=width,
            height=height,
            description=self.description,
            ui_element=self.ui_element,
            confidence=self.confidence,
        )


@dataclass
class ChangeDetectionResult:
    """Result of change detection between two frames."""
    changed_region: Optional[CropRegion]
    change_description: str
    is_significant: bool = True


def find_llama_server() -> Optional[str]:
    """Find llama-server executable."""
    # Check if in PATH
    llama_server = shutil.which("llama-server")
    if llama_server:
        return llama_server

    # Check common locations
    if sys.platform == "win32":
        # Get conda prefix if in a conda environment
        conda_prefix = os.environ.get("CONDA_PREFIX", "")

        common_paths = [
            # Conda environment
            Path(conda_prefix) / "Library" / "bin" / "llama-server.exe" if conda_prefix else None,
            # Common Windows install locations
            Path(os.environ.get("LOCALAPPDATA", "")) / "llama.cpp" / "llama-server.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "llama.cpp" / "llama-server.exe",
            Path(os.environ.get("PROGRAMFILES", "")) / "llama.cpp" / "llama-server.exe",
            # Common build output locations
            Path("C:/llama.cpp/build/bin/Release/llama-server.exe"),
            Path("D:/llama.cpp/build/bin/Release/llama-server.exe"),
            Path("C:/llama.cpp/llama-server.exe"),
            Path("D:/llama.cpp/llama-server.exe"),
            # User home
            Path.home() / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe",
            Path.home() / "llama.cpp" / "llama-server.exe",
            # Downloads folder (pre-compiled releases)
            Path.home() / "Downloads" / "llama-b7708-bin-win-cuda-12.4-x64" / "llama-server.exe",
        ]

        # Also search for any llama-*-bin-* folders in Downloads
        downloads_dir = Path.home() / "Downloads"
        if downloads_dir.exists():
            for folder in downloads_dir.glob("llama-*-bin-*"):
                if folder.is_dir():
                    server_path = folder / "llama-server.exe"
                    if server_path.exists():
                        common_paths.append(server_path)
        # Filter None values
        common_paths = [p for p in common_paths if p is not None]
    else:
        common_paths = [
            Path("/usr/local/bin/llama-server"),
            Path("/opt/llama.cpp/llama-server"),
            Path.home() / "llama.cpp" / "build" / "bin" / "llama-server",
            Path.home() / "llama.cpp" / "llama-server",
        ]

    for path in common_paths:
        if path.exists():
            return str(path)

    return None


def find_vlm_models(models_dir: Optional[Path] = None) -> Tuple[Optional[Path], Optional[Path]]:
    """Find VLM model and mmproj files.

    Args:
        models_dir: Directory to search (auto-detected if None)

    Returns:
        (model_path, mmproj_path) tuple, either may be None
    """
    # Auto-detect models directory
    if models_dir is None:
        # Try relative to this file's location
        this_file = Path(__file__).resolve()
        project_root = this_file.parent.parent.parent.parent  # core -> atomscribe -> src -> project
        models_dir = project_root / "models"

    if not models_dir.exists():
        return None, None

    model_path = None
    mmproj_path = None

    # Search for model files
    for f in models_dir.glob("*.gguf"):
        name_lower = f.name.lower()
        if "mmproj" in name_lower:
            mmproj_path = f
        elif "qwen" in name_lower and "vl" in name_lower and "mmproj" not in name_lower:
            model_path = f

    return model_path, mmproj_path


class VLMProcessor:
    """
    VLM processor using llama.cpp server for Qwen3-VL-8B.

    Can auto-start the server when needed if model files are available.
    """

    # Default server URL
    DEFAULT_SERVER_URL = "http://localhost:8080"

    # Smart cropping prompt template
    SMART_CROP_PROMPT = """You are analyzing a screenshot from a software tutorial.

Context from transcript: "{context}"
Mouse click position: ({mouse_x}, {mouse_y})
Image dimensions: {width} x {height}

Task: Identify the most instructive region to crop for documentation.

Consider:
1. The area around the mouse click (if provided)
2. Related UI elements (buttons, panels, dialogs, menus)
3. What would help a reader understand this step

Output ONLY a JSON object (no other text):
{{"crop_region": {{"x": <int>, "y": <int>, "width": <int>, "height": <int>}}, "description": "<brief description>", "ui_element": "<UI element name if identifiable>"}}"""

    # Change detection prompt template
    CHANGE_DETECTION_PROMPT = """You are comparing two screenshots: BEFORE (first image) and AFTER (second image) an operation.

Operation described: "{operation}"

Task: Identify what changed between the two images.

Focus on:
1. UI state changes (buttons, selections, values)
2. Content changes (text, numbers, graphics)
3. Visual feedback (highlights, selections, cursors)

Output ONLY a JSON object (no other text):
{{"changed_region": {{"x": <int>, "y": <int>, "width": <int>, "height": <int>}}, "change_description": "<what specifically changed>", "is_significant": <true/false>}}

If no significant change detected, set is_significant to false."""

    def __init__(
        self,
        server_url: Optional[str] = None,
        model_path: Optional[Path] = None,
        mmproj_path: Optional[Path] = None,
        auto_start: bool = True,
        gpu_layers: int = 99,
        port: int = 8080,
        timeout: int = 120,
    ):
        """Initialize VLM processor.

        Args:
            server_url: URL of llama.cpp server
            model_path: Path to VLM model GGUF (auto-detected if None)
            mmproj_path: Path to mmproj GGUF (auto-detected if None)
            auto_start: Auto-start server if not running
            gpu_layers: GPU layers for server (-1 = all)
            port: Port for auto-started server
            timeout: Request timeout in seconds
        """
        self.server_url = server_url or self.DEFAULT_SERVER_URL
        self.auto_start = auto_start
        self.gpu_layers = gpu_layers
        self.port = port
        self.timeout = timeout

        # Auto-detect model paths if not provided
        if model_path is None or mmproj_path is None:
            detected_model, detected_mmproj = find_vlm_models()
            self.model_path = model_path or detected_model
            self.mmproj_path = mmproj_path or detected_mmproj
        else:
            self.model_path = model_path
            self.mmproj_path = mmproj_path

        self._server_process: Optional[subprocess.Popen] = None
        self._server_started_by_us = False
        self._available: Optional[bool] = None

        # Log detected paths
        if self.model_path:
            logger.info(f"VLM model: {self.model_path}")
        if self.mmproj_path:
            logger.info(f"VLM mmproj: {self.mmproj_path}")

    def _check_server_available(self, check_model_loaded: bool = False) -> bool:
        """Check if VLM server is responding.

        Args:
            check_model_loaded: If True, also verify the model is fully loaded
        """
        try:
            response = requests.get(
                f"{self.server_url}/health",
                timeout=5,
            )
            if response.status_code != 200:
                return False

            # If we need to check model is loaded, parse the response
            if check_model_loaded:
                try:
                    data = response.json()
                    # llama-server returns {"status": "ok"} when model is loaded
                    # and {"status": "loading model"} when still loading
                    status = data.get("status", "")
                    if status == "ok":
                        return True
                    elif "loading" in status.lower():
                        return False
                    # If status is not clear, assume loaded
                    return True
                except Exception:
                    # If can't parse JSON, just check if endpoint responds
                    return True

            return True
        except Exception:
            return False

    def _check_server_process_alive(self) -> bool:
        """Check if the server process is still running."""
        if self._server_process is None:
            return False
        return self._server_process.poll() is None

    def _start_server(self) -> bool:
        """Start llama-server with VLM model.

        Returns:
            True if server started successfully
        """
        if self._server_process is not None:
            logger.debug("Server process already exists")
            return True

        # Check if model files exist
        if not self.model_path or not self.model_path.exists():
            logger.error(f"VLM model not found: {self.model_path}")
            return False

        if not self.mmproj_path or not self.mmproj_path.exists():
            logger.error(f"VLM mmproj not found: {self.mmproj_path}")
            logger.error("Download mmproj from: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF")
            return False

        # Find llama-server
        llama_server = find_llama_server()
        if not llama_server:
            logger.error("llama-server not found. Please install llama.cpp and add to PATH")
            return False

        logger.info(f"Starting VLM server with {self.model_path.name}...")

        try:
            # Build command
            # Note: -ub 2048 is required for vision models to avoid
            # "non-causal attention requires n_ubatch >= n_tokens" assertion
            cmd = [
                llama_server,
                "-m", str(self.model_path),
                "--mmproj", str(self.mmproj_path),
                "-ngl", str(self.gpu_layers),
                "--port", str(self.port),
                "--host", "127.0.0.1",
                "-ub", "2048",  # Larger ubatch for vision tokens
            ]

            logger.debug(f"Server command: {' '.join(cmd)}")

            # Start server process
            # Use DEVNULL for stdout/stderr to avoid buffer filling issues on Windows
            self._server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )

            # Register cleanup
            atexit.register(self._stop_server)
            self._server_started_by_us = True

            # Wait for server to be ready (up to 120 seconds for model loading)
            # VLM models are large (~9GB) and need time to load
            logger.info("Waiting for VLM server to load model (this may take 30-60 seconds)...")
            for i in range(120):
                time.sleep(1)

                # Check if process died
                if self._server_process.poll() is not None:
                    exit_code = self._server_process.returncode
                    logger.error(f"VLM server failed to start (exit code: {exit_code})")
                    self._server_process = None
                    return False

                # Check if model is fully loaded (not just server responding)
                if self._check_server_available(check_model_loaded=True):
                    logger.info(f"VLM server ready after {i + 1} seconds")
                    return True

                # Log progress every 10 seconds
                if (i + 1) % 10 == 0:
                    logger.info(f"Still loading VLM model... ({i + 1}s)")

            logger.error("VLM server startup timed out after 120 seconds")
            self._stop_server()
            return False

        except Exception as e:
            logger.error(f"Failed to start VLM server: {e}")
            return False

    def _stop_server(self):
        """Stop the auto-started server."""
        if self._server_process is not None and self._server_started_by_us:
            logger.info("Stopping VLM server...")
            try:
                self._server_process.terminate()
                self._server_process.wait(timeout=5)
            except Exception:
                self._server_process.kill()
            self._server_process = None
            self._server_started_by_us = False

    def ensure_available(self) -> bool:
        """Ensure VLM is available, starting server if needed.

        Returns:
            True if VLM is available
        """
        # Check if already available
        if self._check_server_available():
            self._available = True
            return True

        # Try to start server if auto-start enabled
        if self.auto_start and self.model_path and self.mmproj_path:
            if self._start_server():
                self._available = True
                return True

        self._available = False
        return False

    @property
    def is_available(self) -> bool:
        """Check if VLM server is available."""
        if self._available is not None:
            return self._available

        # Just check, don't auto-start yet
        self._available = self._check_server_available()
        return self._available

    @property
    def has_model_files(self) -> bool:
        """Check if model files are available for auto-start."""
        return (
            self.model_path is not None and
            self.model_path.exists() and
            self.mmproj_path is not None and
            self.mmproj_path.exists()
        )

    def _encode_image(
        self,
        image_path: Path,
        max_size: int = 1024,
        return_dims: bool = False
    ) -> Optional[str] | Optional[Tuple[str, Tuple[int, int], Tuple[int, int]]]:
        """Encode image to base64 for API, resizing if too large.

        Args:
            image_path: Path to image
            max_size: Maximum dimension (width or height) in pixels
            return_dims: If True, return (base64, original_dims, resized_dims)

        Returns:
            Base64 encoded image string, or tuple with dimensions if return_dims=True
        """
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return None

        try:
            if not HAS_PIL:
                # No PIL, just encode raw file
                with open(image_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                    if return_dims:
                        return (b64, (1920, 1080), (1920, 1080))  # Unknown dims
                    return b64

            # Load and resize if needed
            with Image.open(image_path) as img:
                original_dims = img.size
                resized_dims = original_dims

                # Check if resizing is needed
                if img.width > max_size or img.height > max_size:
                    # Calculate new size maintaining aspect ratio
                    ratio = min(max_size / img.width, max_size / img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    logger.debug(f"Resizing image from {img.size} to {new_size} for VLM")
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    resized_dims = new_size

                # Convert to RGB if needed (remove alpha channel)
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')

                # Encode to JPEG bytes
                import io
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                if return_dims:
                    return (b64, original_dims, resized_dims)
                return b64

        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None

    def _get_image_size(self, image_path: Path) -> Tuple[int, int]:
        """Get image dimensions."""
        if not HAS_PIL:
            logger.warning("PIL not available, cannot get image size")
            return (1920, 1080)

        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception as e:
            logger.error(f"Failed to get image size: {e}")
            return (1920, 1080)

    def _call_vlm(
        self,
        prompt: str,
        images: List[str],
        max_tokens: int = 512,
        retry_count: int = 1,
    ) -> Optional[str]:
        """Call VLM server with prompt and images.

        Args:
            prompt: Text prompt
            images: List of base64-encoded images
            max_tokens: Maximum tokens in response
            retry_count: Number of retries if server crashes
        """
        if not self.ensure_available():
            logger.warning("VLM not available")
            return None

        # Build content array with images and text
        content = []
        for img_b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}"
                }
            })
        content.append({
            "type": "text",
            "text": prompt
        })

        # Build request body (OpenAI-compatible format)
        body = {
            "model": "qwen3-vl",
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }

        try:
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=body,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            if "choices" in result and result["choices"]:
                return result["choices"][0]["message"]["content"].strip()

            logger.error(f"Unexpected VLM response format: {result}")
            return None

        except requests.exceptions.Timeout:
            logger.error("VLM request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"VLM request failed: {e}")

            # Check if server crashed
            if self._server_started_by_us and not self._check_server_process_alive():
                logger.warning("VLM server process died, attempting restart...")
                self._server_process = None
                self._available = None

                if retry_count > 0:
                    # Try to restart and retry
                    if self._start_server():
                        logger.info("VLM server restarted, retrying request...")
                        return self._call_vlm(prompt, images, max_tokens, retry_count - 1)

            return None
        except Exception as e:
            logger.error(f"VLM processing error: {e}")
            return None

    def _call_vlm_with_tool(
        self,
        prompt: str,
        images: List[str],
        tool: dict,
        max_tokens: int = 512,
        retry_count: int = 1,
    ) -> Optional[dict]:
        """Call VLM server with a tool definition.

        Args:
            prompt: Text prompt
            images: List of base64-encoded images
            tool: Tool definition dict with name, description, parameters
            max_tokens: Maximum tokens in response
            retry_count: Number of retries if server crashes

        Returns:
            Parsed tool call arguments as dict, or None if failed
        """
        if not self.ensure_available():
            logger.warning("VLM not available")
            return None

        # Build content array with images and text
        content = []
        for img_b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}"
                }
            })
        content.append({
            "type": "text",
            "text": prompt
        })

        # Build request body with tool
        # Note: llama.cpp server expects tool_choice as a string (function name),
        # not OpenAI-style object format
        body = {
            "model": "qwen3-vl",
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "tools": [{"type": "function", "function": tool}],
            "tool_choice": tool["name"],  # Force use of this tool
        }

        try:
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=body,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            logger.debug(f"VLM tool response: {result}")

            if "choices" in result and result["choices"]:
                choice = result["choices"][0]
                message = choice.get("message", {})

                # Check for tool calls
                tool_calls = message.get("tool_calls", [])
                if tool_calls:
                    tool_call = tool_calls[0]
                    func = tool_call.get("function", {})
                    args_str = func.get("arguments", "{}")
                    try:
                        return json.loads(args_str)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse tool arguments: {e}")
                        return None

                # Fallback: try to parse content as JSON (some servers don't use tool_calls)
                content_text = message.get("content", "")
                if content_text:
                    logger.debug(f"No tool_calls, trying to parse content: {content_text[:200]}")
                    return self._parse_json_response(content_text)

            logger.error(f"Unexpected VLM response format: {result}")
            return None

        except requests.exceptions.Timeout:
            logger.error("VLM request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"VLM request failed: {e}")

            # Check if server crashed
            if self._server_started_by_us and not self._check_server_process_alive():
                logger.warning("VLM server process died, attempting restart...")
                self._server_process = None
                self._available = None

                if retry_count > 0:
                    if self._start_server():
                        logger.info("VLM server restarted, retrying request...")
                        return self._call_vlm_with_tool(prompt, images, tool, max_tokens, retry_count - 1)

            return None
        except Exception as e:
            logger.error(f"VLM tool processing error: {e}")
            return None

    def _parse_json_response(self, response: str) -> Optional[dict]:
        """Parse JSON from VLM response."""
        if not response:
            return None

        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            logger.warning(f"No JSON found in VLM response: {response[:100]}")
            return None

        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse VLM JSON: {e}")
            return None

    def smart_crop(
        self,
        image_path: Path,
        context: str = "",
        mouse_position: Optional[Tuple[int, int]] = None,
        fallback_size: Tuple[int, int] = (800, 600),
    ) -> CropRegion:
        """Identify the most instructive region to crop using VLM tool calling."""
        # Get original image size
        orig_width, orig_height = self._get_image_size(image_path)
        mouse_x, mouse_y = mouse_position or (orig_width // 2, orig_height // 2)

        # Try VLM-based cropping
        encode_result = self._encode_image(image_path, return_dims=True)
        if encode_result:
            img_b64, (orig_w, orig_h), (resized_w, resized_h) = encode_result

            # Calculate scale factor for coordinate conversion
            scale_x = orig_w / resized_w
            scale_y = orig_h / resized_h

            # Scale mouse position to resized image coordinates
            scaled_mouse_x = int(mouse_x / scale_x)
            scaled_mouse_y = int(mouse_y / scale_y)

            # Build prompt for tool-based cropping
            prompt = f"""You are analyzing a screenshot from a software tutorial.

Context from transcript: "{context[:500] if context else 'No context available'}"
Mouse click position: ({scaled_mouse_x}, {scaled_mouse_y})
Image dimensions: {resized_w} x {resized_h}

Use the crop_image tool to select the most instructive region for documentation.
Consider the area around the mouse click and related UI elements."""

            # Define crop tool with bounds constraints
            crop_tool = {
                "name": "crop_image",
                "description": "Select a rectangular region from the screenshot to crop for documentation. The region should highlight the most relevant UI element or area for the tutorial step.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "integer",
                            "description": f"Left edge of crop region (0 to {resized_w - 100})",
                            "minimum": 0,
                            "maximum": resized_w - 100
                        },
                        "y": {
                            "type": "integer",
                            "description": f"Top edge of crop region (0 to {resized_h - 100})",
                            "minimum": 0,
                            "maximum": resized_h - 100
                        },
                        "width": {
                            "type": "integer",
                            "description": f"Width of crop region (100 to {resized_w})",
                            "minimum": 100,
                            "maximum": resized_w
                        },
                        "height": {
                            "type": "integer",
                            "description": f"Height of crop region (100 to {resized_h})",
                            "minimum": 100,
                            "maximum": resized_h
                        },
                        "description": {
                            "type": "string",
                            "description": "Brief description of what this cropped region shows"
                        },
                        "ui_element": {
                            "type": "string",
                            "description": "Name of the main UI element in the crop (e.g., 'Save Button', 'Settings Panel')"
                        }
                    },
                    "required": ["x", "y", "width", "height", "description"]
                }
            }

            result = self._call_vlm_with_tool(prompt, [img_b64], crop_tool)

            if result:
                # Scale VLM coordinates back to original image size
                crop = CropRegion(
                    x=int(int(result.get("x", 0)) * scale_x),
                    y=int(int(result.get("y", 0)) * scale_y),
                    width=int(int(result.get("width", fallback_size[0])) * scale_x),
                    height=int(int(result.get("height", fallback_size[1])) * scale_y),
                    description=result.get("description", ""),
                    ui_element=result.get("ui_element", ""),
                    confidence=0.8,
                )

                # Clamp to image bounds if needed
                if not crop.is_valid(orig_width, orig_height):
                    logger.debug(f"Clamping VLM crop region to image bounds: {crop}")
                    crop = crop.clamp(orig_width, orig_height)

                logger.debug(f"VLM smart crop: {crop}")
                return crop

        # Fallback: center crop around mouse position
        logger.info("Using fallback center crop around mouse position")
        return self._create_fallback_crop(
            orig_width, orig_height, mouse_x, mouse_y, fallback_size
        )

    def _create_fallback_crop(
        self,
        image_width: int,
        image_height: int,
        center_x: int,
        center_y: int,
        crop_size: Tuple[int, int],
    ) -> CropRegion:
        """Create a fallback crop region centered on a point."""
        crop_w, crop_h = crop_size
        crop_w = min(crop_w, image_width)
        crop_h = min(crop_h, image_height)

        x = max(0, min(center_x - crop_w // 2, image_width - crop_w))
        y = max(0, min(center_y - crop_h // 2, image_height - crop_h))

        return CropRegion(
            x=x,
            y=y,
            width=crop_w,
            height=crop_h,
            description="Fallback crop around mouse position",
            confidence=0.3,
        )

    def detect_changes(
        self,
        before_image: Path,
        after_image: Path,
        operation_description: str = "",
    ) -> ChangeDetectionResult:
        """Detect changes between before and after screenshots using VLM tool calling."""
        before_result = self._encode_image(before_image, return_dims=True)
        after_result = self._encode_image(after_image, return_dims=True)

        if not before_result or not after_result:
            logger.error("Failed to encode images for change detection")
            return ChangeDetectionResult(
                changed_region=None,
                change_description="Failed to load images",
                is_significant=False,
            )

        before_b64, (orig_w, orig_h), (resized_w, resized_h) = before_result
        after_b64, _, _ = after_result

        # Calculate scale factor for coordinate conversion
        scale_x = orig_w / resized_w
        scale_y = orig_h / resized_h

        # Build prompt for tool-based change detection
        prompt = f"""You are comparing two screenshots: BEFORE (first image) and AFTER (second image) an operation.

Operation described: "{operation_description[:300] if operation_description else 'Unknown operation'}"
Image dimensions: {resized_w} x {resized_h}

Use the detect_change tool to identify what changed between the two images.
Focus on UI state changes, content changes, and visual feedback."""

        # Define change detection tool with bounds constraints
        change_tool = {
            "name": "detect_change",
            "description": "Identify the region where changes occurred between the before and after screenshots, and describe the change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "has_change": {
                        "type": "boolean",
                        "description": "Whether a significant visual change was detected"
                    },
                    "x": {
                        "type": "integer",
                        "description": f"Left edge of changed region (0 to {resized_w - 100})",
                        "minimum": 0,
                        "maximum": resized_w - 100
                    },
                    "y": {
                        "type": "integer",
                        "description": f"Top edge of changed region (0 to {resized_h - 100})",
                        "minimum": 0,
                        "maximum": resized_h - 100
                    },
                    "width": {
                        "type": "integer",
                        "description": f"Width of changed region (100 to {resized_w})",
                        "minimum": 100,
                        "maximum": resized_w
                    },
                    "height": {
                        "type": "integer",
                        "description": f"Height of changed region (100 to {resized_h})",
                        "minimum": 100,
                        "maximum": resized_h
                    },
                    "change_description": {
                        "type": "string",
                        "description": "Description of what specifically changed between the images"
                    }
                },
                "required": ["has_change", "change_description"]
            }
        }

        result = self._call_vlm_with_tool(prompt, [before_b64, after_b64], change_tool, max_tokens=512)

        if result:
            has_change = result.get("has_change", True)
            change_desc = result.get("change_description", "")

            # Check if we have valid coordinates for the changed region
            if has_change and all(k in result for k in ["x", "y", "width", "height"]):
                # Scale VLM coordinates back to original image size
                crop = CropRegion(
                    x=int(int(result.get("x", 0)) * scale_x),
                    y=int(int(result.get("y", 0)) * scale_y),
                    width=int(int(result.get("width", 400)) * scale_x),
                    height=int(int(result.get("height", 300)) * scale_y),
                    description=change_desc,
                    confidence=0.7,
                )

                # Clamp to image bounds if needed
                if not crop.is_valid(orig_w, orig_h):
                    logger.debug(f"Clamping VLM change region to image bounds: {crop}")
                    crop = crop.clamp(orig_w, orig_h)

                return ChangeDetectionResult(
                    changed_region=crop,
                    change_description=change_desc,
                    is_significant=has_change,
                )

            return ChangeDetectionResult(
                changed_region=None,
                change_description=change_desc,
                is_significant=has_change,
            )

        logger.warning("VLM change detection failed, returning no significant change")
        return ChangeDetectionResult(
            changed_region=None,
            change_description="Could not analyze changes",
            is_significant=False,
        )

    def crop_image(
        self,
        image_path: Path,
        crop_region: CropRegion,
        output_path: Path,
    ) -> bool:
        """Crop an image and save to file."""
        if not HAS_PIL:
            logger.error("PIL not available, cannot crop image")
            return False

        try:
            with Image.open(image_path) as img:
                if not crop_region.is_valid(img.width, img.height):
                    logger.error(f"Invalid crop region: {crop_region}")
                    return False

                cropped = img.crop(crop_region.to_tuple())
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cropped.save(output_path, quality=90)

                logger.debug(f"Cropped image saved to {output_path}")
                return True

        except Exception as e:
            logger.error(f"Failed to crop image: {e}")
            return False

    def shutdown(self):
        """Shutdown the VLM processor and stop any auto-started server."""
        self._stop_server()


# Singleton instance
_processor_instance: Optional[VLMProcessor] = None


def get_vlm_processor(
    server_url: Optional[str] = None,
    auto_start: bool = True,
) -> VLMProcessor:
    """Get the singleton VLM processor instance.

    Args:
        server_url: Optional server URL (only used on first call)
        auto_start: Auto-start server if not running (only used on first call)

    Returns:
        VLMProcessor instance
    """
    global _processor_instance
    if _processor_instance is None:
        # Load config for settings
        try:
            from .config import get_config_manager
            config = get_config_manager().config
            _processor_instance = VLMProcessor(
                server_url=server_url or config.vlm_server_url,
                model_path=Path(config.vlm_model_path) if config.vlm_model_path else None,
                mmproj_path=Path(config.vlm_mmproj_path) if config.vlm_mmproj_path else None,
                auto_start=config.vlm_auto_start_server if auto_start else False,
                gpu_layers=config.vlm_gpu_layers,
                port=config.vlm_server_port,
            )
        except Exception:
            _processor_instance = VLMProcessor(
                server_url=server_url,
                auto_start=auto_start,
            )
    return _processor_instance
