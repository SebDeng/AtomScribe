# AtomScribe

**AI Lab Scribe** - A modern desktop application for real-time audio transcription in laboratory environments.

Built with PySide6, featuring a clean Notion-inspired interface.

## Features

- **Real-time Recording** - Record lab sessions with live waveform visualization
- **Microphone Selection** - Auto-detect and select from available audio input devices
- **File Browser** - Browse and manage transcription files (.json, .txt, .md, .wav, .mp3)
- **Live Transcript** - View real-time transcription with speaker identification
- **Session Preview** - Preview transcript, summary, and events in tabbed view
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

- [ ] Speech-to-text integration (Whisper, Azure Speech)
- [ ] Speaker diarization
- [ ] Export to multiple formats (PDF, DOCX, SRT)
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
- [Notion](https://notion.so) - UI design inspiration
- [Loguru](https://github.com/Delgan/loguru) - Python logging made simple
