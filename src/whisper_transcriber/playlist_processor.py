"""YouTube playlist processing module.

Copyright (c) 2025 Whisper Transcriber Contributors
Licensed under the MIT License - see LICENSE file for details.
"""

import re
from pathlib import Path
from typing import Any

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from yt_dlp import YoutubeDL

from .downloader import AudioDownloader
from .transcriber import WhisperTranscriber
from .transcript_io import save_transcript_file
from .utils import DownloadError, GPUError, TranscriptionError, console


class PlaylistProcessor:
    """Handle batch processing of YouTube playlists.

    Constants:
        MAX_FILENAME_LENGTH: Maximum length for generated filenames (200 chars)
        MAX_PLAYLIST_SIZE: Maximum number of videos that can be processed (9999)
        INDEX_PADDING: Number of digits used for video index padding (4)
    """

    MAX_FILENAME_LENGTH = 200
    MAX_PLAYLIST_SIZE = 9999  # Based on 4-digit padding (0001-9999)
    INDEX_PADDING = 4

    def __init__(
        self,
        output_dir: Path,
        transcriber: WhisperTranscriber,
        max_videos: int | None = None,
        start_index: int = 1,
        keep_audio: bool = False,
        translate: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize the playlist processor.

        Args:
            output_dir: Directory to save transcripts
            transcriber: Configured WhisperTranscriber instance
            max_videos: Maximum number of videos to process (None for all)
            start_index: Start processing from this video index (1-based)
            keep_audio: Whether to keep downloaded audio files
            translate: Whether to translate transcripts
            verbose: Whether to show verbose output
        """
        self.output_dir = output_dir
        self.transcriber = transcriber
        self.max_videos = max_videos
        self.start_index = start_index
        self.keep_audio = keep_audio
        self.translate = translate
        self.verbose = verbose
        self.downloader = AudioDownloader()

    def process_playlist(
        self,
        playlist_url: str,
        language: str | None,
        output_format: str,
    ) -> None:
        """Process all videos in a playlist.

        Args:
            playlist_url: YouTube playlist URL
            language: Source language for transcription
            output_format: Output format (text, srt, vtt, json)
        """
        # Extract playlist information
        console.print("[bold]Extracting playlist information...[/bold]")

        try:
            playlist_info = self._extract_playlist_info(playlist_url)
            videos = playlist_info["entries"]
            playlist_title = playlist_info.get("title", "Unknown Playlist")

            console.print(f"[green]✓[/green] Found playlist: {playlist_title}")
            console.print(f"[green]✓[/green] Total videos: {len(videos)}")

            # Validate playlist size
            if len(videos) > self.MAX_PLAYLIST_SIZE:
                console.print(
                    f"[red]✗ Playlist too large:[/red] {len(videos)} videos found, "
                    f"maximum supported is {self.MAX_PLAYLIST_SIZE}"
                )
                return

        except (KeyError, ValueError) as e:
            console.print(f"[red]✗ Failed to extract playlist:[/red] {e}")
            return

        # Apply filtering
        videos = self._filter_videos(videos)

        if not videos:
            console.print("[yellow]No videos to process after filtering[/yellow]")
            return

        console.print(f"[cyan]Processing {len(videos)} videos...[/cyan]")

        # Process videos
        success_count = 0
        failed_videos = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            playlist_task = progress.add_task("Processing playlist...", total=len(videos))

            for i, video_info in enumerate(videos, 1):
                video_title = video_info.get("title", f"Video {i}")
                video_url = video_info.get("webpage_url") or video_info.get("url")

                if not video_url:
                    console.print(f"[yellow]⚠ Skipping video {i}: No URL found[/yellow]")
                    failed_videos.append((i, video_title, "No URL found"))
                    progress.update(playlist_task, advance=1)
                    continue

                console.print(f"\n[bold cyan]Video {i}/{len(videos)}:[/bold cyan] {video_title}")

                try:
                    # Generate safe filename
                    safe_filename = self._generate_safe_filename(i, video_title, output_format)
                    output_path = self.output_dir / safe_filename

                    # Process single video
                    self._process_single_video(
                        video_url, output_path, language, output_format, video_info
                    )

                    success_count += 1
                    console.print(f"[green]✓ Completed video {i}[/green]")

                except (DownloadError, TranscriptionError, GPUError) as e:
                    console.print(f"[red]✗ Failed to process video {i}:[/red] {e}")
                    failed_videos.append((i, video_title, str(e)))

                progress.update(playlist_task, advance=1)

        # Print summary
        self._print_summary(success_count, failed_videos, playlist_title)

    def _extract_playlist_info(self, playlist_url: str) -> dict[str, Any]:
        """Extract playlist information using yt-dlp."""
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,  # Don't download, just get info
        }

        with YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(playlist_url, download=False)  # type: ignore[no-any-return]

    def _filter_videos(self, videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply filtering based on start_index and max_videos."""
        # Filter by start index (convert from 1-based to 0-based)
        start_idx = self.start_index - 1
        if start_idx > 0:
            videos = videos[start_idx:]
            console.print(f"[cyan]Starting from video {self.start_index}[/cyan]")

        # Limit number of videos
        if self.max_videos is not None:
            videos = videos[: self.max_videos]
            console.print(f"[cyan]Limited to {self.max_videos} videos[/cyan]")

        return videos

    def _generate_safe_filename(self, index: int, title: str, output_format: str) -> str:
        """Generate safe filename for video transcript.

        Args:
            index: Video index in playlist
            title: Video title
            output_format: Output format extension

        Returns:
            Safe filename with format: 0001_video_title.ext
        """
        # Clean title for use in filename
        safe_title = re.sub(r'[<>:"/\\|?*]', "_", title)  # Replace invalid chars
        safe_title = re.sub(r"_+", "_", safe_title)  # Collapse multiple underscores
        safe_title = safe_title.strip("_. ")  # Remove leading/trailing chars

        # Calculate space for title (accounting for index padding, underscore, dot, and extension)
        index_str = str(index).zfill(self.INDEX_PADDING)
        reserved_chars = len(index_str) + 1 + 1 + len(output_format)  # index + _ + . + ext
        max_title_length = self.MAX_FILENAME_LENGTH - reserved_chars

        if len(safe_title) > max_title_length:
            safe_title = safe_title[:max_title_length].rstrip("_. ")

        # Format with configurable padding: 0001_video_title.ext
        return f"{index_str}_{safe_title}.{output_format}"

    def _process_single_video(
        self,
        video_url: str,
        output_path: Path,
        language: str | None,
        output_format: str,
        video_info: dict[str, Any],
    ) -> None:
        """Process a single video from the playlist."""
        # Download audio
        try:
            audio_path, metadata = self.downloader.download(video_url, keep_audio=self.keep_audio)
        except DownloadError as e:
            msg = f"Download failed: {e}"
            raise TranscriptionError(msg) from e

        # Merge video info into metadata
        metadata.update(video_info)

        # Transcribe
        try:
            result = self.transcriber.transcribe(
                audio_path,
                language=language,
                output_format=output_format,
                metadata=metadata,
                translate_to_english=self.translate,
            )

            # Handle translation results
            if isinstance(result, tuple):
                transcript_original, transcript_translated = result
                is_translated = True
            else:
                transcript_original = result
                transcript_translated = None
                is_translated = False

            # Save transcript(s)
            if is_translated:
                # Save both Hebrew and English versions
                output_he = output_path.with_name(output_path.stem + "_he" + output_path.suffix)
                output_en = output_path.with_name(output_path.stem + "_en" + output_path.suffix)

                self._save_transcript(transcript_original, output_he, output_format, metadata)
                if transcript_translated is not None:
                    self._save_transcript(transcript_translated, output_en, output_format, metadata)
            else:
                self._save_transcript(transcript_original, output_path, output_format, metadata)

        except (GPUError, TranscriptionError) as e:
            msg = f"Transcription failed: {e}"
            raise TranscriptionError(msg) from e

    def _save_transcript(
        self,
        transcript: str,
        output_path: Path,
        output_format: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Save transcript to file."""
        save_transcript_file(str(transcript), str(output_path), output_format, metadata)

    def _print_summary(
        self,
        success_count: int,
        failed_videos: list[tuple[int, str, str]],
        playlist_title: str,
    ) -> None:
        """Print processing summary."""
        console.print("\n[bold green]Playlist Processing Complete![/bold green]")
        console.print(f"Playlist: {playlist_title}")

        # Summary table
        table = Table(title="Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="white")

        table.add_row("Videos Processed", str(success_count + len(failed_videos)))
        table.add_row("Successful", str(success_count))
        table.add_row("Failed", str(len(failed_videos)))

        console.print(table)

        # Failed videos details
        if failed_videos:
            console.print("\n[bold red]Failed Videos:[/bold red]")
            for index, title, error in failed_videos:
                console.print(f"  {index:03d}. {title}")
                console.print(f"       [red]Error:[/red] {error}")

    def cleanup(self) -> None:
        """Clean up resources used by the playlist processor."""
        if hasattr(self, "downloader"):
            self.downloader.cleanup()
