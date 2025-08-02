"""Command-line interface for whisper-transcriber.

Copyright (c) 2025 Whisper Transcriber Contributors
Licensed under the MIT License - see LICENSE file for details.
"""

import os
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .downloader import AudioDownloader
from .model_manager import get_model_manager
from .playlist_processor import PlaylistProcessor
from .transcriber import WhisperTranscriber
from .transcript_io import save_transcript_file
from .utils import (
    DownloadError,
    GPUError,
    PlaylistSizeLimitError,
    TranscriptionError,
    check_cuda_availability,
    get_language_filename,
    validate_url,
)

console = Console()


def validate_playlist_args(output: str, start_index: int, max_videos: int | None) -> Path:
    """Validate and prepare playlist mode arguments.

    Args:
        output: Output directory path
        start_index: Start video index
        max_videos: Maximum number of videos

    Returns:
        Path: Validated output directory path

    Raises:
        SystemExit: If validation fails
    """
    output_path = Path(output)
    if output_path.suffix:  # Has file extension
        console.print(
            "[red]Error:[/red] For playlist mode, output must be a directory path "
            "(no file extension)"
        )
        sys.exit(1)

    # Create output directory if it doesn't exist
    if not output_path.exists():
        console.print(f"[yellow]Creating directory:[/yellow] {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
    elif not output_path.is_dir():
        console.print(f"[red]Error:[/red] Output path exists but is not a directory: {output_path}")
        sys.exit(1)

    # Validate start_index
    if start_index < 1:
        console.print("[red]Error:[/red] --start-index must be >= 1")
        sys.exit(1)

    # Validate max_videos
    if max_videos is not None and max_videos < 1:
        console.print("[red]Error:[/red] --max-videos must be >= 1")
        sys.exit(1)

    return output_path


def validate_single_video_args(output: str) -> Path:
    """Validate and prepare single video mode arguments.

    Args:
        output: Output file path

    Returns:
        Path: Validated output file path
    """
    output_path = Path(output)
    output_dir = output_path.parent
    if output_dir and not output_dir.exists():
        console.print(f"[yellow]Creating directory:[/yellow] {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_path


def show_configuration_table(
    model: str,
    language: str | None,
    format: str,  # noqa: A002
    device: str,
    keep_audio: bool,
    translate: bool,
    playlist: bool,
    max_videos: int | None,
    start_index: int,
) -> None:
    """Display configuration table with current settings."""
    table = Table(title="Configuration", show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    cuda_available, gpu_info = check_cuda_availability()
    device_info = gpu_info if device == "auto" and cuda_available else device

    # Show model type
    model_type = (
        "Hebrew-optimized" if model in WhisperTranscriber.HEBREW_MODELS else "Standard OpenAI"
    )
    table.add_row("Model", f"{model} ({model_type})")
    table.add_row("Language", language or "auto-detect")
    table.add_row("Format", format)
    table.add_row("Device", device_info or device)
    table.add_row("Keep Audio", "Yes" if keep_audio else "No")
    table.add_row("Translate", "Yes (Hebrew → English)" if translate else "No")

    if playlist:
        table.add_row("Mode", "Playlist")
        table.add_row("Max Videos", str(max_videos) if max_videos else "All")
        table.add_row("Start Index", str(start_index))
    else:
        table.add_row("Mode", "Single Video")

    console.print(table)
    console.print()


def process_playlist(
    url: str,
    output_path: Path,
    model: str,
    language: str | None,
    format: str,  # noqa: A002
    device: str,
    gpu_device: int,
    keep_audio: bool,
    translate: bool,
    max_videos: int | None,
    start_index: int,
    verbose: bool,
    translation_context: bool = True,
    context_lines: int = 2,
    transcriber: WhisperTranscriber | None = None,
) -> None:
    """Process a YouTube playlist."""
    # Use provided transcriber or get from model manager
    if transcriber is None:
        device_setting = None if device == "auto" else device
        model_manager = get_model_manager()
        transcriber = model_manager.get_transcriber(
            model_size=model, device=device_setting, gpu_device=gpu_device
        )

    processor = PlaylistProcessor(
        output_dir=output_path,
        transcriber=transcriber,
        max_videos=max_videos,
        start_index=start_index,
        keep_audio=keep_audio,
        translate=translate,
        verbose=verbose,
        translation_context=translation_context,
        context_lines=context_lines,
    )

    try:
        processor.process_playlist(url, language, format)
        console.print("[bold green]All done! 🎉[/bold green]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
    except (DownloadError, TranscriptionError, GPUError, PlaylistSizeLimitError) as e:
        console.print(f"[red]✗ Playlist processing failed:[/red] {e}")
        sys.exit(1)
    finally:
        processor.cleanup()


def handle_transcription_result(result: str | tuple[str, str]) -> tuple[str, str | None, bool]:
    """Handle transcription result, returning tuple of (original, translated, is_translated).

    Args:
        result: Either transcript string or tuple of (original, translated)

    Returns:
        tuple: (transcript_original, transcript_translated, is_translated)
    """
    if isinstance(result, tuple):
        transcript_original, transcript_translated = result
        is_translated = True
    else:
        transcript_original = result
        transcript_translated = None
        is_translated = False

    return transcript_original, transcript_translated, is_translated


def _download_audio(url: str, keep_audio: bool) -> tuple[str, dict[str, Any], AudioDownloader]:
    """Download audio from YouTube URL.

    Args:
        url: YouTube video URL
        keep_audio: Whether to keep the downloaded audio file

    Returns:
        tuple: (audio_path, metadata, downloader)
    """
    console.rule("[bold]Step 1/3: Downloading Audio[/bold]")
    downloader = AudioDownloader()

    try:
        audio_path, metadata = downloader.download(url, keep_audio=keep_audio)
        audio_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
        console.print(f"[green]✓[/green] Downloaded audio: {audio_size:.1f} MB")
        return audio_path, metadata, downloader
    except DownloadError as e:
        console.print(f"[red]✗ Download failed:[/red] {e}")
        sys.exit(1)


def _transcribe_audio_with_retry(
    audio_path: str,
    model: str,
    device_setting: str | None,
    gpu_device: int,
    language: str | None,
    format: str,  # noqa: A002
    metadata: dict[str, Any],
    translate: bool,
    verbose: bool,
    translation_context: bool = True,
    context_lines: int = 2,
) -> tuple[str, str | None, bool]:
    """Transcribe audio with GPU fallback to CPU.

    Args:
        audio_path: Path to audio file
        model: Model name
        device_setting: Device setting (None for auto)
        gpu_device: GPU device ID
        language: Language code
        format: Output format
        metadata: Video metadata
        translate: Whether to translate
        verbose: Whether to show verbose output
        translation_context: Whether to use translation context
        context_lines: Number of context lines to use

    Returns:
        tuple: (transcript_original, transcript_translated, is_translated)
    """
    console.rule("[bold]Step 2/3: Transcribing Audio[/bold]")

    # Get model manager and transcriber
    model_manager = get_model_manager()
    transcriber = model_manager.get_transcriber(
        model_size=model, device=device_setting, gpu_device=gpu_device
    )

    try:
        result = transcriber.transcribe(
            audio_path,
            language=language,
            output_format=format,
            metadata=metadata,
            translate_to_english=translate,
            use_translation_context=translation_context,
            context_lines=context_lines,
        )

        # Handle translation results
        transcript_original, transcript_translated, is_translated = handle_transcription_result(
            result
        )

        # Show preview of transcript
        if verbose and format == "text":
            preview = (
                transcript_original[:500] + "..."
                if len(transcript_original) > 500
                else transcript_original
            )
            console.print(Panel(preview, title="Transcript Preview", border_style="green"))

        return transcript_original, transcript_translated, is_translated

    except GPUError as e:
        console.print(f"[red]✗ GPU Error:[/red] {e}")
        console.print("[yellow]Retrying with CPU...[/yellow]")

        # Retry with CPU using model manager
        transcriber = model_manager.get_transcriber(
            model_size=model,
            device="cpu",
            gpu_device=gpu_device,
            force_reload=True,  # Force reload to get CPU version
        )
        result = transcriber.transcribe(
            audio_path,
            language=language,
            output_format=format,
            metadata=metadata,
            translate_to_english=translate,
            use_translation_context=translation_context,
            context_lines=context_lines,
        )

        # Handle translation results
        return handle_transcription_result(result)


def _validate_inputs(url: str, context_lines: int) -> None:
    """Validate input parameters."""
    if not validate_url(url):
        console.print("[red]Error:[/red] Invalid YouTube URL")
        sys.exit(1)

    if context_lines < 0 or context_lines > 5:
        console.print("[red]Error:[/red] Context lines must be between 0 and 5")
        sys.exit(1)


def _handle_playlist_processing(
    url: str,
    output_path: Path,
    model: str,
    language: str | None,
    output_format: str,
    device: str,
    gpu_device: int,
    keep_audio: bool,
    translate: bool,
    max_videos: int | None,
    start_index: int,
    verbose: bool,
    translation_context: bool,
    context_lines: int,
    model_manager: Any,
    device_setting: str | None,
) -> None:
    """Handle playlist processing workflow."""
    # Pre-load transcriber for playlist processing
    transcriber = model_manager.get_transcriber(
        model_size=model, device=device_setting, gpu_device=gpu_device
    )

    process_playlist(
        url,
        output_path,
        model,
        language,
        output_format,
        device,
        gpu_device,
        keep_audio,
        translate,
        max_videos,
        start_index,
        verbose,
        translation_context,
        context_lines,
        transcriber=transcriber,
    )


def _handle_single_video_processing(
    url: str,
    output_path: Path,
    model: str,
    device_setting: str | None,
    gpu_device: int,
    language: str | None,
    output_format: str,
    translate: bool,
    verbose: bool,
    translation_context: bool,
    context_lines: int,
    keep_audio: bool,
) -> None:
    """Handle single video processing workflow."""
    # Download audio
    audio_path, metadata, downloader = _download_audio(url, keep_audio)

    # Transcribe audio with retry
    transcript_original, transcript_translated, is_translated = _transcribe_audio_with_retry(
        audio_path,
        model,
        device_setting,
        gpu_device,
        language,
        output_format,
        metadata,
        translate,
        verbose,
        translation_context,
        context_lines,
    )

    # Step 3: Save transcript(s)
    console.rule("[bold]Step 3/3: Saving Transcript(s)[/bold]")

    if is_translated:
        # Save both Hebrew and English versions
        output_he = get_language_filename(str(output_path), "he")
        output_en = get_language_filename(str(output_path), "en")

        # Save Hebrew version
        save_transcript_file(transcript_original, output_he, output_format, metadata)

        # Save English version
        assert transcript_translated is not None  # Type checker
        save_transcript_file(transcript_translated, output_en, output_format, metadata)
    else:
        # Save single transcript
        save_transcript_file(transcript_original, str(output_path), output_format, metadata)

    # Cleanup
    if not keep_audio:
        downloader.cleanup()


@click.command()
@click.option(
    "--translation-context/--no-translation-context",
    default=True,
    help="Include previous translations as context for consistency (default: enabled)",
)
@click.option(
    "--context-lines",
    type=int,
    default=2,
    help="Number of previous translated lines to include as context (0-5, default: 2)",
)
@click.argument("url")
@click.argument("output", type=click.Path())
@click.option(
    "-m",
    "--model",
    type=click.Choice(WhisperTranscriber.AVAILABLE_MODELS),
    default="ivrit-turbo",
    help="Whisper model size. Hebrew models: ivrit-turbo (recommended), ivrit-large, ivrit-small",
)
@click.option(
    "-l",
    "--language",
    default=None,
    help="Source language (e.g., he for Hebrew). Auto-detect if not specified.",
)
@click.option(
    "-f",
    "--format",
    type=click.Choice(["text", "srt", "vtt", "json"]),
    default="text",
    help="Output format (text includes metadata header, json includes full metadata)",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "auto"]),
    default="auto",
    help="Device to use for transcription",
)
@click.option(
    "--gpu-device",
    type=int,
    default=0,
    help="GPU device ID to use for translation (default: 0)",
)
@click.option("--keep-audio", is_flag=True, help="Keep the downloaded audio file")
@click.option(
    "--translate",
    is_flag=True,
    help="Translate Hebrew transcripts to English (creates both _he and _en files)",
)
@click.option(
    "--playlist",
    is_flag=True,
    help="Process entire YouTube playlist (output must be a directory)",
)
@click.option(
    "--max-videos",
    type=int,
    default=None,
    help="Maximum number of videos to process from playlist",
)
@click.option(
    "--start-index",
    type=int,
    default=1,
    help="Start processing from this video index (1-based)",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def main(
    url: str,
    output: str,
    model: str,
    language: str | None,
    format: str,  # noqa: A002
    device: str,
    gpu_device: int,
    keep_audio: bool,
    translate: bool,
    playlist: bool,
    max_videos: int | None,
    start_index: int,
    verbose: bool,
    translation_context: bool,
    context_lines: int,
) -> None:
    """Download YouTube audio and transcribe it using Whisper (with Hebrew optimization).

    URL: YouTube video URL or playlist URL (use --playlist flag for playlists)
    OUTPUT: Output file path for single video, or directory path for playlist

    Models:
    - ivrit-turbo: Hebrew-optimized, fast (recommended for Hebrew content)
    - ivrit-large: Hebrew-optimized, highest quality
    - ivrit-small: Hebrew-optimized, smallest size
    - Standard OpenAI models: tiny, base, small, medium, large, large-v2, large-v3
    """
    model_manager = None
    try:
        # Print header
        console.print(
            Panel.fit(
                "[bold cyan]Whisper Transcriber[/bold cyan]\n"
                "YouTube Audio Transcription Tool\n"
                "[dim]With Hebrew-optimized models from ivrit-ai[/dim]",
                border_style="cyan",
            )
        )

        # Validate inputs
        _validate_inputs(url, context_lines)

        # Validate playlist vs single video mode
        if playlist:
            output_path = validate_playlist_args(output, start_index, max_videos)
        else:
            output_path = validate_single_video_args(output)

        # Show configuration
        if verbose:
            show_configuration_table(
                model,
                language,
                format,
                device,
                keep_audio,
                translate,
                playlist,
                max_videos,
                start_index,
            )

        # Get model manager for persistent models
        model_manager = get_model_manager()
        device_setting = None if device == "auto" else device

        # Handle playlist or single video processing
        if playlist:
            _handle_playlist_processing(
                url,
                output_path,
                model,
                language,
                format,
                device,
                gpu_device,
                keep_audio,
                translate,
                max_videos,
                start_index,
                verbose,
                translation_context,
                context_lines,
                model_manager,
                device_setting,
            )
        else:
            _handle_single_video_processing(
                url,
                output_path,
                model,
                device_setting,
                gpu_device,
                language,
                format,
                translate,
                verbose,
                translation_context,
                context_lines,
                keep_audio,
            )

        console.print("\n[bold green]✓ Transcription completed successfully![/bold green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except TranscriptionError as e:
        console.print(f"\n[red]Error:[/red] {e}")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        console.print(f"\n[red]Unexpected error:[/red] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)
    finally:
        # Show memory stats in verbose mode
        if verbose and model_manager:
            memory_stats = model_manager.get_memory_stats()
            if memory_stats["loaded_transcribers"] > 0 or memory_stats["loaded_translators"] > 0:
                console.print("\n[dim]Memory Usage:[/dim]")
                console.print(
                    f"[dim]  Loaded models: {memory_stats['loaded_transcribers']} transcriber(s), "
                    f"{memory_stats['loaded_translators']} translator(s)[/dim]"
                )
                if "gpu_memory_allocated_mb" in memory_stats:
                    console.print(
                        f"[dim]  GPU memory: "
                        f"{memory_stats['gpu_memory_allocated_mb']:.1f} MB allocated[/dim]"
                    )

        # Clean up model manager resources to free GPU memory
        if model_manager:
            model_manager.cleanup_all()


if __name__ == "__main__":
    main()
