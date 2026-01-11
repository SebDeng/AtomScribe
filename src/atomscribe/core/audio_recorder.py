"""Audio recording service for AtomScribe"""

import wave
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
from loguru import logger

try:
    import sounddevice as sd
except ImportError:
    sd = None
    logger.warning("sounddevice not installed. Audio recording will not work.")

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None
    logger.warning("pydub not installed. MP3 conversion will not work.")


class RecordingState(Enum):
    """Recording state enumeration"""
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPING = "stopping"


@dataclass
class AudioDevice:
    """Audio input device information"""
    index: int
    name: str
    channels: int
    sample_rate: float
    is_default: bool = False


class AudioRecorder:
    """
    Audio recorder that captures audio from microphone and saves to file.
    Supports pause/resume and real-time audio level monitoring.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 1,
        dtype: str = "int16",
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype

        self._state = RecordingState.IDLE
        self._stream: Optional[sd.InputStream] = None
        self._wav_file: Optional[wave.Wave_write] = None
        self._output_path: Optional[Path] = None
        self._temp_wav_path: Optional[Path] = None

        self._frames: List[np.ndarray] = []
        self._lock = threading.Lock()

        # Callback for audio level updates
        self._level_callback: Optional[Callable[[float], None]] = None

        # Callback for raw audio data (for transcription)
        self._audio_data_callback: Optional[Callable[[np.ndarray], None]] = None

        # Selected device
        self._device_index: Optional[int] = None

    @property
    def state(self) -> RecordingState:
        """Get current recording state"""
        return self._state

    @property
    def is_recording(self) -> bool:
        """Check if currently recording"""
        return self._state == RecordingState.RECORDING

    @property
    def is_paused(self) -> bool:
        """Check if recording is paused"""
        return self._state == RecordingState.PAUSED

    def set_level_callback(self, callback: Callable[[float], None]):
        """Set callback for audio level updates (0.0 to 1.0)"""
        self._level_callback = callback

    def set_audio_data_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback for raw audio data (for transcription)"""
        self._audio_data_callback = callback

    def set_device(self, device_index: int):
        """Set the audio input device"""
        self._device_index = device_index
        logger.info(f"Set audio device to index {device_index}")

    @staticmethod
    def get_audio_devices() -> List[AudioDevice]:
        """Get list of available audio input devices"""
        if sd is None:
            return []

        devices = []
        try:
            default_device = sd.query_devices(kind="input")
            default_index = default_device["index"] if default_device else -1

            for i, device in enumerate(sd.query_devices()):
                if device["max_input_channels"] > 0:
                    devices.append(AudioDevice(
                        index=i,
                        name=device["name"],
                        channels=device["max_input_channels"],
                        sample_rate=device["default_samplerate"],
                        is_default=(i == default_index),
                    ))
        except Exception as e:
            logger.error(f"Error getting audio devices: {e}")

        return devices

    def start_recording(self, output_path: Path, convert_to_mp3: bool = True):
        """
        Start recording audio to file.

        Args:
            output_path: Final output file path (with .mp3 or .wav extension)
            convert_to_mp3: If True and output is .mp3, record to temp WAV then convert
        """
        if sd is None:
            raise RuntimeError("sounddevice not installed")

        if self._state != RecordingState.IDLE:
            logger.warning("Recording already in progress")
            return

        self._output_path = Path(output_path)
        self._frames = []

        # If output is MP3, record to temporary WAV first
        if self._output_path.suffix.lower() == ".mp3" and convert_to_mp3:
            self._temp_wav_path = self._output_path.with_suffix(".temp.wav")
        else:
            self._temp_wav_path = self._output_path

        # Open WAV file for writing
        try:
            self._wav_file = wave.open(str(self._temp_wav_path), "wb")
            self._wav_file.setnchannels(self.channels)
            self._wav_file.setsampwidth(2)  # 16-bit
            self._wav_file.setframerate(self.sample_rate)
        except Exception as e:
            logger.error(f"Failed to open WAV file: {e}")
            raise

        # Start audio stream
        try:
            self._stream = sd.InputStream(
                device=self._device_index,
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype=self.dtype,
                callback=self._audio_callback,
                blocksize=1024,
            )
            self._stream.start()
            self._state = RecordingState.RECORDING
            logger.info(f"Started recording to {self._temp_wav_path}")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            if self._wav_file:
                self._wav_file.close()
            raise

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback for audio stream - called on audio thread"""
        if status:
            logger.warning(f"Audio callback status: {status}")

        if self._state == RecordingState.RECORDING:
            # Write to WAV file
            with self._lock:
                if self._wav_file:
                    self._wav_file.writeframes(indata.tobytes())

            # Calculate audio level for visualization
            if self._level_callback:
                # RMS level normalized to 0-1
                rms = np.sqrt(np.mean(indata.astype(np.float32) ** 2))
                level = min(1.0, rms / 10000.0)  # Normalize (adjust divisor as needed)
                self._level_callback(level)

            # Send audio data to transcriber
            if self._audio_data_callback:
                # Make a copy to avoid issues with buffer reuse
                self._audio_data_callback(indata.copy())

        elif self._state == RecordingState.PAUSED:
            # Send zero level when paused
            if self._level_callback:
                self._level_callback(0.0)

    def pause_recording(self):
        """Pause the recording"""
        if self._state == RecordingState.RECORDING:
            self._state = RecordingState.PAUSED
            logger.info("Recording paused")

    def resume_recording(self):
        """Resume the recording"""
        if self._state == RecordingState.PAUSED:
            self._state = RecordingState.RECORDING
            logger.info("Recording resumed")

    def stop_recording(self) -> Optional[Path]:
        """
        Stop recording and finalize the file.

        Returns:
            Path to the final output file, or None if failed
        """
        if self._state == RecordingState.IDLE:
            logger.warning("No recording in progress")
            return None

        self._state = RecordingState.STOPPING

        # Stop and close the stream
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.error(f"Error closing stream: {e}")
            self._stream = None

        # Close the WAV file
        with self._lock:
            if self._wav_file:
                try:
                    self._wav_file.close()
                except Exception as e:
                    logger.error(f"Error closing WAV file: {e}")
                self._wav_file = None

        logger.info(f"Recording stopped, WAV saved to {self._temp_wav_path}")

        # Convert to MP3 if needed
        final_path = self._output_path
        if (
            self._temp_wav_path != self._output_path
            and self._output_path.suffix.lower() == ".mp3"
        ):
            final_path = self._convert_to_mp3()

        self._state = RecordingState.IDLE
        return final_path

    def _convert_to_mp3(self) -> Optional[Path]:
        """Convert the temporary WAV file to MP3"""
        # Check if temp WAV file exists and has content
        if not self._temp_wav_path or not self._temp_wav_path.exists():
            logger.error(f"Temp WAV file does not exist: {self._temp_wav_path}")
            return None

        wav_size = self._temp_wav_path.stat().st_size
        logger.info(f"Temp WAV file size: {wav_size} bytes")

        if wav_size < 100:  # Almost empty WAV file
            logger.error("WAV file is empty or too small")
            return None

        if AudioSegment is None:
            logger.warning("pydub not installed, keeping WAV file")
            # Rename temp WAV to final WAV
            final_wav = self._output_path.with_suffix(".wav")
            try:
                if final_wav.exists():
                    final_wav.unlink()
                self._temp_wav_path.rename(final_wav)
                logger.info(f"Saved as WAV: {final_wav}")
                # Ensure temp file is gone (rename should have moved it)
                if self._temp_wav_path.exists():
                    self._temp_wav_path.unlink()
                    logger.debug("Cleaned up temp WAV file")
                return final_wav
            except Exception as e:
                logger.error(f"Failed to rename WAV: {e}")
                return self._temp_wav_path

        try:
            logger.info(f"Converting {self._temp_wav_path} to MP3...")
            audio = AudioSegment.from_wav(str(self._temp_wav_path))
            audio.export(
                str(self._output_path),
                format="mp3",
                bitrate="128k",
            )
            logger.info(f"MP3 saved to {self._output_path}")

            # Verify MP3 was created
            if self._output_path.exists():
                mp3_size = self._output_path.stat().st_size
                logger.info(f"MP3 file size: {mp3_size} bytes")

                # Remove temporary WAV file
                try:
                    self._temp_wav_path.unlink()
                    logger.debug("Removed temporary WAV file")
                except Exception as e:
                    logger.warning(f"Could not remove temp WAV: {e}")

                return self._output_path
            else:
                logger.error("MP3 file was not created")
                return self._temp_wav_path

        except Exception as e:
            logger.error(f"Failed to convert to MP3: {e}")
            logger.exception("Full traceback:")
            # Keep the WAV file as fallback - rename it
            final_wav = self._output_path.with_suffix(".wav")
            try:
                if self._temp_wav_path.exists():
                    # Remove destination if it exists
                    if final_wav.exists():
                        final_wav.unlink()
                    self._temp_wav_path.rename(final_wav)
                    logger.info(f"Saved as WAV fallback: {final_wav}")

                    # Clean up the failed MP3 file (conversion failed, so delete it)
                    if self._output_path.exists() and self._output_path.suffix.lower() == ".mp3":
                        self._output_path.unlink()
                        logger.debug("Removed failed MP3 file")

                    # Ensure temp file is gone
                    if self._temp_wav_path.exists():
                        self._temp_wav_path.unlink()
                        logger.debug("Cleaned up temp WAV file")

                    return final_wav
            except Exception as rename_error:
                logger.error(f"Failed to rename WAV: {rename_error}")
            return self._temp_wav_path if self._temp_wav_path.exists() else None

    def get_recording_duration(self) -> float:
        """Get the current recording duration in seconds (approximate)"""
        if self._wav_file and self._state in (RecordingState.RECORDING, RecordingState.PAUSED):
            try:
                frames_written = self._wav_file.tell()
                return frames_written / self.sample_rate
            except:
                pass
        return 0.0


# Singleton instance
_recorder_instance: Optional[AudioRecorder] = None


def get_audio_recorder() -> AudioRecorder:
    """Get the singleton audio recorder instance"""
    global _recorder_instance
    if _recorder_instance is None:
        _recorder_instance = AudioRecorder()
    return _recorder_instance
