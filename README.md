# AtomScribe

**AI Lab Scribe** - A modern desktop application for real-time audio transcription in laboratory environments.

Built with PySide6, featuring a clean Notion-inspired interface.

## Features

- **Real-time Recording** - Record lab sessions with live waveform visualization
- **Microphone Selection** - Auto-detect and select from available audio input devices
- **Real-time Transcription** - Live speech-to-text using faster-whisper (large-v3 model)
- **LLM Post-Processing** - AI-powered transcript correction with Qwen3-4B-Instruct
  - Fixes transcription errors (e.g., "hi-tension" → "high tension")
  - Removes filler words (嗯, 啊, uh, um, like, etc.)
  - Visual indicator for AI-corrected segments
- **Scientific Vocabulary** - Built-in support for electron microscopy terminology
- **File Browser** - Browse and manage transcription files (.json, .txt, .md, .wav, .mp3)
- **Session Management** - Organize recordings by date and project
- **Modern UI** - Clean, Notion-style bright theme with intuitive controls

## Screenshots

```
┌──────────────────────────────────────────────────────────────────────────┐
│  AI Lab Scribe                                     [Settings] [About]    │
├──────────────────────────────────────────────────────────────────────────┤
│  ● REC  │ Pause │ Stop │  ▁▂▃▅▃▂▁  │  00:15:32  │  🎤 Microphone ▼     │
├──────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────────────────────────────────────────┐   │
│  │ Files       │  │  Realtime Transcript                             │   │
│  │ 🔍 Search   │  │                                                  │   │
│  │ 📁 data/    │  │  Speaker A  14:30:12                             │   │
│  │   notes.txt │  │  "We need to adjust the stigmator..."            │   │
│  │   audio.wav │  │                                                  │   │
│  │             │  │  Speaker B  14:30:45                             │   │
│  │ [Open...]   │  │  "Try the Y direction"                           │   │
│  └─────────────┘  └──────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────────────────┤
│  Ready                                             Session: Untitled     │
└──────────────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.11+
- Conda (recommended) or pip

### Setup with Conda

```bash
# Clone the repository
git clone https://github.com/your-org/AtomScribe.git
cd AtomScribe

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate AtomScribe
```

### Setup with pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the Application

```bash
conda activate AtomScribe
python -m atomscribe.main
```

### Basic Workflow

1. **Select Microphone** - Choose your audio input device from the dropdown
2. **Start Recording** - Click the REC button to begin recording
3. **Monitor Audio** - Watch the live waveform and timer
4. **Pause/Resume** - Use Pause button to temporarily stop recording
5. **Stop Recording** - Click Stop to end the session
6. **Browse Files** - Use the left sidebar to navigate saved transcriptions

## Project Structure

```
AtomScribe/
├── src/
│   └── atomscribe/
│       ├── main.py              # Application entry point
│       ├── app.py               # QApplication configuration
│       ├── signals/             # Global signal definitions
│       ├── ui/
│       │   ├── main_window.py   # Main window layout
│       │   └── widgets/         # UI components
│       ├── styles/              # QSS themes and colors
│       └── core/                # Business logic
├── models/                      # LLM models (Qwen3-4B GGUF)
├── environment.yml              # Conda environment
├── pyproject.toml              # Project metadata
└── README.md
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PySide6 | >=6.6.0 | Qt GUI framework |
| pydantic | >=2.0 | Data validation |
| loguru | >=0.7.0 | Logging |
| sounddevice | >=0.4.6 | Audio recording |
| numpy | >=1.24.0 | Audio processing |
| pydub | >=0.25.1 | MP3 encoding |
| faster-whisper | >=1.0.0 | Speech-to-text |
| llama-cpp-python | >=0.2.0 | LLM inference |

### LLM Model Setup

To enable AI-powered transcript correction, download the Qwen3-4B model:

1. **Download the model** from [HuggingFace](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF)
   - Recommended: `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` (~3GB, requires ~3GB VRAM)

2. **Place the model** at:
   ```
   AtomScribe/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
   ```

3. **For GPU acceleration** (optional):
   ```bash
   CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
   ```

## Development

### Running in Development Mode

```bash
# With live reload (if using watchdog)
python -m atomscribe.main --dev

# Debug logging
python -m atomscribe.main --debug
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Qt signals use `snake_case` naming
- Widget object names use `camelCase` for QSS targeting

## Configuration

Configuration files are stored in:
- Windows: `%APPDATA%\AtomScribe\`
- macOS: `~/Library/Application Support/AtomScribe/`
- Linux: `~/.config/AtomScribe/`

## Roadmap

- [x] Speech-to-text integration (faster-whisper)
- [x] LLM post-processing (Qwen3-4B-Instruct)
- [ ] Speaker diarization (pyannote-audio)
- [ ] Export to multiple formats (PDF, DOCX, SRT)
- [ ] Settings dialog for configuration
- [ ] Cloud sync support
- [ ] Keyboard shortcuts
- [ ] Dark theme option

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is proprietary software developed by AtomE Corp.

## Acknowledgments

- [PySide6](https://doc.qt.io/qtforpython/) - Qt for Python
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Fast Whisper transcription
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - Python bindings for llama.cpp
- [Qwen3](https://huggingface.co/Qwen) - Alibaba's powerful language model
- [Notion](https://notion.so) - UI design inspiration
- [Loguru](https://github.com/Delgan/loguru) - Python logging made simple
