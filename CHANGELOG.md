# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-08-02

### Added
- **Enhanced Translation System**: Major improvements to Hebrew-to-English translation accuracy
  - Context-aware translation with video metadata integration
  - Professional role-playing prompts to prevent conversational responses
  - Playlist title integration for topic consistency across videos
  - Video title and channel context for domain-specific terminology
  - **Translation Context Continuity**: Previous translation context for consistent terminology
    - New CLI options: `--translation-context/--no-translation-context` (default: enabled)
    - New CLI option: `--context-lines` (0-5, default: 2) to control context window size
    - Sliding window of 2-3 recent translations maintains consistency within videos
    - Smart context truncation prevents token overflow
  - Robust fallback mechanisms for translation failures

### Improved
- **Translation Quality**: Significantly reduced context drift and improved accuracy
  - Enhanced prompts with clear instructions to avoid treating subtitles as questions
  - Video/playlist context prevents off-topic translations
  - Better domain awareness leads to more appropriate word choices
  - **Enhanced Terminology Consistency**: Previous translations provide context for consistent word choices
  - **Style Continuity**: Maintains formal/informal tone established in earlier subtitle segments
  - Better coherence across video segments through translation memory

### Technical Changes
- **Translator Module**: Added `translate_with_context()` method with metadata and previous translations support
- **Translation Context Tracking**: Enhanced `translate_segments()` to maintain sliding window of previous translations
- **CLI Enhancements**: New command-line options for controlling translation context behavior
- **Transcriber Integration**: Modified `_perform_translation()` to pass context settings throughout pipeline
- **Playlist Processors**: Updated both sequential and concurrent processors to support context continuity
- **Pipeline Integration**: Complete context flow from CLI → transcriber → playlist processors → translator
- **Metadata Flow**: Enhanced pipeline from extraction to translation with context awareness

## [0.1.1] - 2025-07-27

### Added
- **Development Tooling**: Comprehensive code quality and development workflow setup
  - Ruff for fast linting and code formatting (replaces Black, isort, flake8)
  - Mypy for static type checking with strict configuration
  - Pre-commit hooks for automatic code quality checks on commit
  - Makefile with convenient development commands
- **Code Quality Improvements**: All source code formatted and type-annotated
  - Added type annotations throughout the codebase
  - Fixed linting issues (imports, line length, error handling)
  - Timezone-aware datetime handling
  - Proper exception handling with variable assignment

### Configuration
- Updated `pyproject.toml` with development dependencies and tool configurations
- Created `.pre-commit-config.yaml` for automated quality checks
- Enhanced `.gitignore` with tool-specific exclusions

### Developer Experience
- `make check`: Run all quality checks (format, lint, type-check)
- `make format`: Auto-format code with ruff
- `make lint`: Run comprehensive linting
- `make type-check`: Static type analysis
- `make install-dev`: Install development dependencies
- `make clean`: Clean up cache and build artifacts

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
