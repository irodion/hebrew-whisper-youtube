"""Utility functions for error handling and system checks.

Copyright (c) 2025 Whisper Transcriber Contributors
Licensed under the MIT License - see LICENSE file for details.
"""

import shutil
from urllib.parse import urlparse

import torch
from rich.console import Console

console = Console()


class TranscriptionError(Exception):
    """Base exception for transcription errors."""


class DownloadError(TranscriptionError):
    """Error during YouTube download."""


class GPUError(TranscriptionError):
    """Error with GPU/CUDA setup."""


def check_cuda_availability() -> tuple[bool, str | None]:
    """Check if CUDA is available and return status with GPU info.

    Returns:
        Tuple of (is_available, gpu_info_string)
    """
    if not torch.cuda.is_available():
        return False, None

    gpus = [
        f"{torch.cuda.get_device_name(i)} "
        f"({torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f} GB)"
        for i in range(torch.cuda.device_count())
    ]
    info = ", ".join(gpus)
    return True, info


def check_disk_space(path: str, required_gb: float = 2.0) -> bool:
    """Check if there's enough disk space.

    Args:
        path: Path to check
        required_gb: Required space in GB

    Returns:
        True if enough space available
    """
    stat = shutil.disk_usage(path)
    available_gb = stat.free / (1024**3)
    return available_gb >= required_gb


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours}h {mins}m"


def validate_url(url: str) -> bool:
    """Basic YouTube URL validation."""
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        return host in {"www.youtube.com", "youtube.com", "youtu.be"} and bool(parsed.path)
    except (ValueError, AttributeError):
        return False
