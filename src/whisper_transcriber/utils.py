"""Utility functions for error handling and system checks.

Copyright (c) 2025 Whisper Transcriber Contributors
Licensed under the MIT License - see LICENSE file for details.
"""
import shutil
import torch
from typing import Tuple, Optional
from rich.console import Console

console = Console()

class TranscriptionError(Exception):
    """Base exception for transcription errors."""
    pass

class DownloadError(TranscriptionError):
    """Error during YouTube download."""
    pass

class GPUError(TranscriptionError):
    """Error with GPU/CUDA setup."""
    pass

def check_cuda_availability() -> Tuple[bool, Optional[str]]:
    """
    Check if CUDA is available and return status with GPU info.
    
    Returns:
        Tuple of (is_available, gpu_info_string)
    """
    if not torch.cuda.is_available():
        return False, None
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    
    info = f"{gpu_name} ({gpu_memory:.1f} GB)"
    return True, info

def check_disk_space(path: str, required_gb: float = 2.0) -> bool:
    """
    Check if there's enough disk space.
    
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
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"

def validate_url(url: str) -> bool:
    """Basic YouTube URL validation."""
    youtube_patterns = [
        'youtube.com/watch?v=',
        'youtu.be/',
        'youtube.com/embed/',
        'youtube.com/v/',
    ]
    return any(pattern in url for pattern in youtube_patterns)