# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Running the Application
```bash
# Run the CLI directly with Python module syntax
.venv/bin/python -m whisper_transcriber --help

# Test with sample video (Hebrew-optimized model)
.venv/bin/python -m whisper_transcriber -m ivrit-turbo "https://youtube.com/watch?v=VIDEO_ID" output.txt

# Force CPU mode if GPU issues occur
.venv/bin/python -m whisper_transcriber --device cpu "URL" output.txt
```

### Testing
```bash
# Run the test script
bash test_cli.sh

# The test script uses a public domain video for testing
```

### Development Setup
```bash
# Install dependencies with UV
uv sync

# Install in development mode
uv pip install -e .
```

## Architecture Overview

The codebase implements a YouTube audio transcriber with Hebrew-optimized support:

### Core Components

1. **cli.py** - Entry point and command-line interface
   - Uses Click for CLI parsing
   - Rich console for formatted output
   - Handles GPU fallback to CPU automatically
   - Three-step process: download → transcribe → save

2. **transcriber.py** - Whisper model management
   - Supports both standard OpenAI and Hebrew-optimized ivrit-ai models
   - Automatic GPU/CPU device selection with fallback
   - Output formats: text, SRT, VTT
   - Hebrew models force language to 'he' and don't support translation

3. **downloader.py** - YouTube audio download logic
   - Uses yt-dlp for downloading
   - Progress tracking with Rich
   - Temporary file management

4. **utils.py** - Shared utilities and error handling
   - Custom exception classes (TranscriptionError, DownloadError, GPUError)
   - CUDA availability checking
   - URL validation

### Hebrew Model Integration

The tool integrates ivrit-ai's Hebrew-optimized Whisper models:
- `ivrit-turbo` (default): Fast 809M parameter model
- `ivrit-large`: Highest quality 1.5B parameter model  
- `ivrit-small`: Same as turbo, included for naming consistency

These models map to HuggingFace paths in `transcriber.py:HEBREW_MODELS` dict.

### GPU Handling

The tool has robust GPU fallback logic:
1. Auto-detects CUDA availability
2. Attempts GPU initialization with float16 compute
3. Falls back to CPU with int8 compute on GPU errors
4. User can force CPU with `--device cpu`