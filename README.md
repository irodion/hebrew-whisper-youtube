# Whisper Transcriber

A CLI tool to download YouTube audio and transcribe it using OpenAI Whisper with GPU acceleration support. **Optimized for Hebrew transcription** using ivrit-ai's fine-tuned models.

## Features

- Download audio from YouTube videos
- Transcribe audio using OpenAI Whisper (via faster-whisper)
- **Hebrew-optimized models** from ivrit-ai for superior Hebrew transcription quality
- **Hebrew-to-English Translation** with choice of local or online translation models
- GPU acceleration support (CUDA)
- Multiple output formats (text, SRT, VTT)
- Progress indicators and rich CLI interface
- Automatic language detection (for standard models)
- Robust error handling

## Installation

### Prerequisites

- Python 3.12+
- UV package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- ffmpeg (for audio processing)
- CUDA toolkit (optional, for GPU acceleration)

### Setup

This project uses UV as the package manager. Make sure you have UV installed, then:

```bash
# Clone the repository
git clone https://github.com/irodion/hebrew-whisper-youtube.git
cd hebrew-whisper-youtube

# Install dependencies
uv sync

# Install in development mode
uv pip install -e .
```

### Verify Installation

```bash
# Test the CLI
.venv/bin/python -m whisper_transcriber --help

# Or if you want to run directly (after setting up entry points)
whisper-transcriber --help
```

## Usage

### Hebrew Content (Recommended)

```bash
# Use Hebrew-optimized model (default) - best for Hebrew videos
whisper-transcriber "https://youtube.com/watch?v=VIDEO_ID" output.txt

# Specify Hebrew model explicitly
whisper-transcriber -m ivrit-turbo "https://youtube.com/watch?v=VIDEO_ID" output.txt

# Use largest Hebrew model for highest quality
whisper-transcriber -m ivrit-large "https://youtube.com/watch?v=VIDEO_ID" output.txt

# Generate Hebrew subtitles (SRT format)
whisper-transcriber -m ivrit-turbo -f srt "https://youtube.com/watch?v=VIDEO_ID" output.srt
```

### Standard Models

```bash
# Use standard OpenAI models for non-Hebrew content
whisper-transcriber -m large "https://youtube.com/watch?v=VIDEO_ID" output.txt

# Specify source language for standard models
whisper-transcriber -m medium -l en "https://youtube.com/watch?v=VIDEO_ID" output.txt
```

### Additional Options

```bash
# Keep the downloaded audio file
whisper-transcriber --keep-audio "https://youtube.com/watch?v=VIDEO_ID" output.txt

# Force CPU usage
whisper-transcriber --device cpu "https://youtube.com/watch?v=VIDEO_ID" output.txt

# Verbose output
whisper-transcriber -v "https://youtube.com/watch?v=VIDEO_ID" output.txt
```

## Model Options

### Hebrew-Optimized Models (Recommended for Hebrew content)
- `ivrit-turbo`: Fast Hebrew transcription, 809M parameters (default)
- `ivrit-large`: Highest quality Hebrew transcription, 1.5B parameters
- `ivrit-small`: Fastest Hebrew transcription, same as ivrit-turbo

### Standard OpenAI Models
- `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`

## Translation Support

### Hebrew-to-English Translation

The tool supports translating Hebrew transcripts to English using two different approaches:

#### Local Translation (Default)
Uses DictaLM 2.0-Instruct model running locally for privacy and offline capability:

```bash
# Translate Hebrew to English using local model
whisper-transcriber --translate "https://youtube.com/watch?v=VIDEO_ID" output.txt
# Creates both output_he.txt and output_en.txt
```

#### Online Translation (Qwen-MT)
Uses Qwen-MT-turbo online model for potentially higher quality translation with full transcript context:

```bash
# Translate using Qwen-MT online model
whisper-transcriber --translate --translator qwen "https://youtube.com/watch?v=VIDEO_ID" output.txt
# Creates both output_he.txt and output_en.txt
```

### Translation Setup

#### For Local Translation (DictaLM)
No additional setup required - the model downloads automatically on first use.

#### For Online Translation (Qwen-MT)
1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Get your API key from [Dashscope Console](https://dashscope.console.aliyun.com/)

3. Add your API key to `.env`:
   ```env
   DASHSCOPE_API_KEY=your_api_key_here
   ```

### Translation Options

```bash
# Local translation (default) - privacy focused, offline capable
whisper-transcriber --translate --translator local "URL" output.txt

# Online translation - higher quality, requires internet and API key
whisper-transcriber --translate --translator qwen "URL" output.txt

# Works with playlists too
whisper-transcriber --translate --translator qwen --playlist "PLAYLIST_URL" output_dir/
```

### Translation Quality Comparison

- **Local (DictaLM)**:
  - ✅ Privacy-focused (runs locally)
  - ✅ No API key required
  - ✅ Works offline
  - ⚠️ Segment-by-segment translation

- **Online (Qwen-MT)**:
  - ✅ Full transcript context for better coherence
  - ✅ Metadata context (video title, playlist, description)
  - ✅ Higher translation quality
  - ⚠️ Requires internet connection and API key
  - ⚠️ Sends data to external service

## CLI Options

- `-m, --model`: Model to use (default: ivrit-turbo)
- `-l, --language`: Source language code (auto-detected for standard models, forced to 'he' for Hebrew models)
- `-f, --format`: Output format: text, srt, vtt (default: text)
- `--device`: Device to use: cuda, cpu, auto (default: auto)
- `--translate`: Translate Hebrew transcripts to English (creates both `_he` and `_en` files)
- `--translator`: Translation model: local (DictaLM) or qwen (Qwen-MT online) (default: local)
- `--playlist`: Process entire YouTube playlist
- `--max-videos`: Maximum number of videos to process from playlist
- `--start-index`: Start processing from this video index (1-based)
- `--keep-audio`: Keep the downloaded audio file
- `-v, --verbose`: Enable verbose output

## Hebrew Model Notes

- Hebrew models are specifically fine-tuned on 5000+ hours of Hebrew audio
- Language detection is disabled for Hebrew models (always uses Hebrew)
- Translation is not supported with Hebrew models (transcription only)
- Hebrew models provide significantly better accuracy for Hebrew content

## GPU Support

The tool automatically detects and uses CUDA-enabled GPUs for faster transcription. On the RTX 4080, transcription is significantly faster than CPU.

### GPU/CUDA Troubleshooting

If you encounter CUDA/cuDNN errors like "Unable to load libcudnn_ops.so", you have several options:

#### Option 1: Force CPU Mode (Recommended for stability)
```bash
# Force CPU usage to avoid GPU compatibility issues
whisper-transcriber --device cpu "https://youtube.com/watch?v=VIDEO_ID" output.txt
```

#### Option 2: Automatic Fallback
The application automatically falls back to CPU if GPU initialization fails:
```bash
# Will try GPU first, fall back to CPU if there are issues
whisper-transcriber --device auto "https://youtube.com/watch?v=VIDEO_ID" output.txt
```

#### Option 3: Fix CUDA Environment (Advanced)
If you need GPU acceleration, ensure compatible versions:
- PyTorch CUDA version matches your system CUDA
- cuDNN version compatibility

**To install cuDNN on Ubuntu 22.04:**
1. Visit [NVIDIA cuDNN Downloads](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)
2. Download the local .deb package for Ubuntu 22.04
3. Install with:
   ```bash
   sudo dpkg -i cudnn-local-repo-ubuntu2204-9.x.x-x_1.0-1_amd64.deb
   sudo apt-get update
   sudo apt-get install libcudnn9-dev
   ```
4. Restart your terminal/environment

## Requirements

- Python 3.12+
- CUDA toolkit (for GPU acceleration)
- ffmpeg (for audio processing)

### Installing ffmpeg

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows (using Chocolatey)
choco install ffmpeg
```

## Architecture

The application is structured with modular components for maintainability:

- `cli.py` - Command-line interface using Click
- `transcriber.py` - Whisper model management and transcription logic
- `downloader.py` - YouTube audio download with progress tracking
- `utils.py` - Utility functions and error handling

## Performance

- **GPU Acceleration**: Up to 4x faster than CPU transcription
- **Hebrew Models**: Significantly better accuracy for Hebrew content vs standard OpenAI models
- **CT2 Format**: Optimized CTranslate2 models for faster inference
- **Memory Efficient**: Lower memory usage with quantized models

## Troubleshooting

### Common Issues

1. **"Command not found"**: Use `.venv/bin/python -m whisper_transcriber` instead of `whisper-transcriber`
2. **GPU errors**: Use `--device cpu` to force CPU mode
3. **Audio download fails**: Check your internet connection and video availability
4. **Empty transcription**: Ensure the video has audio content in the specified language

### Development

```bash
# Run tests (when available)
pytest

# Format code
black src/

# Type checking
mypy src/
```

## Contributing

This project was developed as a helper utility for adding subtitles to Hebrew YouTube videos. Contributions are welcome!

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ivrit-ai](https://huggingface.co/ivrit-ai) for Hebrew-optimized Whisper models
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for optimized inference
- [OpenAI Whisper](https://github.com/openai/whisper) for the original model

## Future Enhancements

- Hebrew to English translation
- Test suite
- Docker support
- Batch processing
- API mode
