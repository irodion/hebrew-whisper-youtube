"""Command-line interface for whisper-transcriber.

Copyright (c) 2025 Whisper Transcriber Contributors
Licensed under the MIT License - see LICENSE file for details.
"""

import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .downloader import AudioDownloader
from .metadata_formatter import MetadataFormatter
from .transcriber import WhisperTranscriber
from .utils import (
    DownloadError,
    GPUError,
    TranscriptionError,
    check_cuda_availability,
    validate_url,
)

console = Console()


@click.command()
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
@click.option("--keep-audio", is_flag=True, help="Keep the downloaded audio file")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def main(
    url: str,
    output: str,
    model: str,
    language: str | None,
    format: str,  # noqa: A002
    device: str,
    keep_audio: bool,
    verbose: bool,
) -> None:
    """Download YouTube audio and transcribe it using Whisper (with Hebrew optimization).

    URL: YouTube video URL
    OUTPUT: Output file path for transcription

    Models:
    - ivrit-turbo: Hebrew-optimized, fast (recommended for Hebrew content)
    - ivrit-large: Hebrew-optimized, highest quality
    - ivrit-small: Hebrew-optimized, smallest size
    - Standard OpenAI models: tiny, base, small, medium, large, large-v2, large-v3
    """
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

        # Validate URL
        if not validate_url(url):
            console.print("[red]Error:[/red] Invalid YouTube URL")
            sys.exit(1)

        # Check output directory exists
        output_path = Path(output)
        output_dir = output_path.parent
        if output_dir and not output_dir.exists():
            console.print(f"[yellow]Creating directory:[/yellow] {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)

        # Show configuration
        if verbose:
            table = Table(title="Configuration", show_header=False)
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="white")

            cuda_available, gpu_info = check_cuda_availability()
            device_info = gpu_info if device == "auto" and cuda_available else device

            # Show model type
            model_type = (
                "Hebrew-optimized"
                if model in WhisperTranscriber.get_hebrew_models()
                else "Standard OpenAI"
            )
            table.add_row("Model", f"{model} ({model_type})")
            table.add_row("Language", language or "auto-detect")
            table.add_row("Format", format)
            table.add_row("Device", device_info or device)
            table.add_row("Keep Audio", "Yes" if keep_audio else "No")

            console.print(table)
            console.print()

        # Step 1: Download audio
        console.rule("[bold]Step 1/3: Downloading Audio[/bold]")
        downloader = AudioDownloader()

        try:
            audio_path, metadata = downloader.download(url, keep_audio=keep_audio)
            audio_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
            console.print(f"[green]✓[/green] Downloaded audio: {audio_size:.1f} MB")
        except DownloadError as e:
            console.print(f"[red]✗ Download failed:[/red] {e}")
            sys.exit(1)

        # Step 2: Transcribe audio
        console.rule("[bold]Step 2/3: Transcribing Audio[/bold]")

        device_setting = None if device == "auto" else device
        transcriber = WhisperTranscriber(model_size=model, device=device_setting)

        try:
            transcript = transcriber.transcribe(
                audio_path, language=language, output_format=format, metadata=metadata
            )

            # Show preview of transcript
            if verbose and format == "text":
                preview = transcript[:500] + "..." if len(transcript) > 500 else transcript
                console.print(Panel(preview, title="Transcript Preview", border_style="green"))

        except GPUError as e:
            console.print(f"[red]✗ GPU Error:[/red] {e}")
            console.print("[yellow]Retrying with CPU...[/yellow]")

            # Retry with CPU
            transcriber = WhisperTranscriber(model_size=model, device="cpu")
            transcript = transcriber.transcribe(
                audio_path, language=language, output_format=format, metadata=metadata
            )

        # Step 3: Save transcript
        console.rule("[bold]Step 3/3: Saving Transcript[/bold]")

        # Add metadata header for text format
        if format == "text":
            header = MetadataFormatter.format_text_header(metadata)
            transcript = header + transcript
        elif format == "vtt":
            # For VTT, prepend metadata as comments
            vtt_header = MetadataFormatter.format_vtt_metadata(metadata)
            # Remove existing WEBVTT header if present and use our metadata-enhanced one
            if transcript.startswith("WEBVTT"):
                transcript = transcript[6:].lstrip("\n")
            transcript = vtt_header + transcript

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        file_size = os.path.getsize(output_path) / 1024  # KB
        console.print(f"[green]✓[/green] Saved transcript to: {output_path} ({file_size:.1f} KB)")

        # Cleanup
        if not keep_audio:
            downloader.cleanup()

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


if __name__ == "__main__":
    main()
