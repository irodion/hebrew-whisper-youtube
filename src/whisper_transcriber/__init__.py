"""YouTube audio transcription tool using OpenAI Whisper.

Copyright (c) 2025 Whisper Transcriber Contributors
Licensed under the MIT License - see LICENSE file for details.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__package__ or "whisper-transcriber")
except PackageNotFoundError:  # local editable install
    __version__ = "0.0.0+dev"
