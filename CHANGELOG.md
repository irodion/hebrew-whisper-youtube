# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-07-26

### Added

#### Core Features
- **YouTube Audio Transcription**: Download and transcribe YouTube videos using OpenAI Whisper
- **Hebrew-Optimized Models**: Integration with ivrit-ai's fine-tuned Hebrew Whisper models
  - `ivrit-turbo`: Fast Hebrew transcription (809M parameters, default)
  - `ivrit-large`: Highest quality Hebrew transcription (1.5B parameters)
  - `ivrit-small`: Alias for ivrit-turbo
- **GPU Acceleration**: CUDA support with automatic CPU fallback
- **Multiple Output Formats**: Support for text, SRT, and VTT subtitle formats
- **Rich CLI**: Beautiful progress bars, status indicators, and formatted output
- **Robust Error Handling**: Comprehensive error handling with graceful degradation

#### Technical Implementation
- **Modular Architecture**: Separated concerns into cli, transcriber, downloader, and utils modules
- **UV Package Management**: Modern Python packaging with UV
- **CTranslate2 Optimization**: 4x faster inference compared to standard OpenAI Whisper
- **Voice Activity Detection**: Built-in VAD for better transcription quality
- **Automatic Language Detection**: For standard models (Hebrew models force Hebrew language)

#### CLI Features
- Command-line interface using Click framework
- Verbose mode with detailed logging and transcript preview
- Device selection (auto, cuda, cpu)
- Audio file preservation option
- Configuration display in verbose mode

#### Dependencies
- **PyTorch 2.7.1**: Latest stable release with CUDA 12.6 support
- **faster-whisper 1.1.1**: Optimized Whisper implementation
- **yt-dlp 2025.7.21**: Latest YouTube downloader with security fixes
- **rich 14.1.0**: Terminal formatting and progress indicators
- **click 8.2.1**: Modern CLI framework

### Technical Details

#### GPU/CUDA Support
- Automatic CUDA detection and GPU information display
- Intelligent error handling for cuDNN compatibility issues
- Automatic fallback from GPU to CPU on initialization failures
- Support for mixed precision (float16 on GPU, int8 on CPU)

#### Hebrew Model Integration
- Direct integration with HuggingFace ivrit-ai models
- Automatic language forcing for Hebrew models (language detection disabled)
- Translation capability disabled for Hebrew models (transcription only)
- Optimized parameters for Hebrew content transcription

#### Audio Processing
- YouTube audio download with progress tracking
- Automatic audio conversion to WAV format (16kHz)
- FFmpeg integration for audio processing
- Temporary file management with cleanup

### Documentation
- Comprehensive README with installation, usage, and troubleshooting guides
- GPU/CUDA troubleshooting section with cuDNN installation instructions
- Model comparison and selection guidance
- Performance benchmarks and optimization notes

### Testing
- Basic functionality tests for Hebrew model integration
- CPU/GPU mode validation
- VAD parameter compatibility verification
- CLI help and option validation

[0.1.0]: https://github.com/username/whisper-transcriber/releases/tag/v0.1.0