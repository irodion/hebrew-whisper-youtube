"""Transcript file I/O operations.

Copyright (c) 2025 Whisper Transcriber Contributors
Licensed under the MIT License - see LICENSE file for details.
"""

import os
from typing import Any

from .metadata_formatter import MetadataFormatter
from .utils import console


def save_transcript_file(
    transcript: str,
    output_path: str,
    format: str,  # noqa: A002
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save transcript to file with appropriate formatting.

    Args:
        transcript: The transcript text
        output_path: Path to save the file
        format: Output format (text, srt, vtt, json)
        metadata: Optional metadata for text/vtt formats
    """
    final_transcript = transcript

    # Add metadata header for text format
    if format == "text" and metadata:
        header = MetadataFormatter.format_text_header(metadata)
        final_transcript = header + transcript
    elif format == "vtt" and metadata:
        # For VTT, prepend metadata as comments
        vtt_header = MetadataFormatter.format_vtt_metadata(metadata)
        # Remove existing WEBVTT header if present and use our metadata-enhanced one
        if transcript.startswith("WEBVTT"):
            final_transcript = transcript[6:].lstrip("\n")
        final_transcript = vtt_header + final_transcript

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_transcript)

    # Calculate and display file size
    file_size = os.path.getsize(output_path) / 1024  # KB
    console.print(f"[green]✓[/green] Saved transcript to: {output_path} ({file_size:.1f} KB)")
