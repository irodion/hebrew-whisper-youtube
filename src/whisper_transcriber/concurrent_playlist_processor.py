"""Concurrent playlist processor with download queue and producer-consumer pattern.

This module implements an optimized playlist processor that downloads videos
concurrently while maintaining sequential transcription for GPU efficiency.
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError as YtDlpDownloadError
from yt_dlp.utils import ExtractorError

from .downloader import AudioDownloader
from .model_manager import get_model_manager
from .transcriber import WhisperTranscriber
from .transcript_io import save_transcript_file
from .utils import (
    DownloadError,
    GPUError,
    PlaylistSizeLimitError,
    TranscriptionError,
    TranslationError,
)

console = Console()


@dataclass
class VideoTask:
    """Container for video processing task information."""

    index: int
    title: str
    url: str
    output_path: Path
    audio_path: str | None = None
    metadata: dict[str, Any] | None = None
    error: Exception | None = None
    completed: bool = False


class ConcurrentPlaylistProcessor:
    """Handle batch processing of YouTube playlists with concurrent downloads.

    This processor implements a producer-consumer pattern where multiple
    downloads happen concurrently (producer) while transcription happens
    sequentially (consumer) to optimize resource utilization.
    """

    MAX_FILENAME_LENGTH = 200
    MAX_PLAYLIST_SIZE = 9999
    INDEX_PADDING = 4
    DEFAULT_DOWNLOAD_WORKERS = 3  # Optimal for most internet connections
    DOWNLOAD_QUEUE_SIZE = 5  # Max downloads to queue ahead

    def __init__(
        self,
        output_dir: Path,
        transcriber: WhisperTranscriber,
        max_videos: int | None = None,
        start_index: int = 1,
        keep_audio: bool = False,
        translate: bool = False,
        verbose: bool = False,
        download_workers: int | None = None,
    ) -> None:
        """Initialize the concurrent playlist processor.

        Args:
            output_dir: Directory to save transcripts
            transcriber: Configured WhisperTranscriber instance
            max_videos: Maximum number of videos to process
            start_index: Start processing from this video index (1-based)
            keep_audio: Whether to keep downloaded audio files
            translate: Whether to translate transcripts
            verbose: Whether to show verbose output
            download_workers: Number of concurrent download workers
        """
        self.output_dir = output_dir
        self.transcriber = transcriber
        self.max_videos = max_videos
        self.start_index = start_index
        self.keep_audio = keep_audio
        self.translate = translate
        self.verbose = verbose
        self.download_workers = download_workers or self.DEFAULT_DOWNLOAD_WORKERS

        # Threading components
        self.download_queue: queue.Queue[VideoTask] = queue.Queue(maxsize=self.DOWNLOAD_QUEUE_SIZE)
        self.transcribe_queue: queue.Queue[VideoTask | None] = queue.Queue()
        self.download_lock = threading.Lock()
        self.stats_lock = threading.Lock()

        # Progress tracking
        self.downloads_completed = 0
        self.transcriptions_completed = 0
        self.failed_videos: list[tuple[int, str, str]] = []

        # Model manager for translator
        self.model_manager = get_model_manager() if translate else None

        # Temporary directory management
        self.temp_dirs: list[str] = []

    def process_playlist(
        self,
        playlist_url: str,
        language: str | None,
        output_format: str,
    ) -> None:
        """Process all videos in a playlist with concurrent downloads.

        Args:
            playlist_url: YouTube playlist URL
            language: Source language for transcription
            output_format: Output format (text, srt, vtt, json)
        """
        start_time = time.time()

        # Extract playlist information
        console.print("[bold]Extracting playlist information...[/bold]")
        videos = self._extract_and_filter_playlist(playlist_url)

        if not videos:
            console.print("[yellow]No videos to process after filtering[/yellow]")
            return

        # Create video tasks
        tasks = self._create_video_tasks(videos, output_format)

        # Process videos concurrently
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # Create progress tasks
            download_task = progress.add_task("Downloading", total=len(tasks))
            transcribe_task = progress.add_task("Transcribing", total=len(tasks))

            # Start processing threads
            with ThreadPoolExecutor(max_workers=self.download_workers + 1) as executor:
                # Submit download workers
                download_futures = []
                for _ in range(self.download_workers):
                    future = executor.submit(self._download_worker, tasks, progress, download_task)
                    download_futures.append(future)

                # Submit transcription worker
                transcribe_future = executor.submit(
                    self._transcribe_worker,
                    language,
                    output_format,
                    progress,
                    transcribe_task,
                )

                # Wait for all workers to complete
                for future in as_completed([*download_futures, transcribe_future]):
                    if exception := future.exception():
                        console.print(f"[red]Worker error:[/red] {exception}")

        # Clean up and print summary
        self._cleanup_temp_dirs()
        elapsed_time = time.time() - start_time
        self._print_summary(len(tasks), elapsed_time)

    def _extract_and_filter_playlist(self, playlist_url: str) -> list[dict[str, Any]]:
        """Extract playlist information and apply filtering."""
        try:
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": True,
            }

            with YoutubeDL(ydl_opts) as ydl:
                playlist_info = ydl.extract_info(playlist_url, download=False)

            videos = playlist_info["entries"]
            playlist_title = playlist_info.get("title", "Unknown Playlist")

            console.print(f"[green]✓[/green] Found playlist: {playlist_title}")
            console.print(f"[green]✓[/green] Total videos: {len(videos)}")

            # Validate playlist size
            if len(videos) > self.MAX_PLAYLIST_SIZE:
                error_msg = (
                    f"Playlist contains {len(videos)} videos, which exceeds the maximum "
                    f"allowed size of {self.MAX_PLAYLIST_SIZE} videos"
                )
                raise PlaylistSizeLimitError(error_msg)

            # Apply filtering
            return self._filter_videos(videos)

        except (
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            YtDlpDownloadError,
            ExtractorError,
        ) as e:
            console.print(f"[red]✗ Failed to extract playlist:[/red] {e}")
            raise

    def _filter_videos(self, videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply start_index and max_videos filtering."""
        # Filter by start index
        start_idx = self.start_index - 1
        if start_idx > 0:
            videos = videos[start_idx:]
            console.print(f"[cyan]Starting from video {self.start_index}[/cyan]")

        # Limit number of videos
        if self.max_videos is not None:
            videos = videos[: self.max_videos]
            console.print(f"[cyan]Limited to {self.max_videos} videos[/cyan]")

        return videos

    def _create_video_tasks(
        self, videos: list[dict[str, Any]], output_format: str
    ) -> list[VideoTask]:
        """Create VideoTask objects for each video."""
        tasks = []
        for i, video_info in enumerate(videos, 1):
            video_title = video_info.get("title", f"Video {i}")
            video_url = video_info.get("webpage_url") or video_info.get("url")

            if not video_url:
                console.print(f"[yellow]⚠ Skipping video {i}: No URL found[/yellow]")
                continue

            # Generate safe filename
            safe_filename = self._generate_safe_filename(i, video_title, output_format)
            output_path = self.output_dir / safe_filename

            task = VideoTask(
                index=i,
                title=video_title,
                url=video_url,
                output_path=output_path,
            )
            tasks.append(task)

        return tasks

    def _generate_safe_filename(self, index: int, title: str, output_format: str) -> str:
        """Generate safe filename for video transcript."""
        # Clean title
        safe_title = re.sub(r'[<>:"/\\|?*]', "_", title)
        safe_title = re.sub(r"_+", "_", safe_title)
        safe_title = safe_title.strip("_. ")

        # Calculate space for title
        index_str = str(index).zfill(self.INDEX_PADDING)
        reserved_chars = len(index_str) + 1 + 1 + len(output_format)
        max_title_length = self.MAX_FILENAME_LENGTH - reserved_chars

        if len(safe_title) > max_title_length:
            safe_title = safe_title[:max_title_length].rstrip("_. ")

        return f"{index_str}_{safe_title}.{output_format}"

    def _download_worker(
        self,
        tasks: list[VideoTask],
        progress: Progress,
        download_task: TaskID,
    ) -> None:
        """Worker thread for downloading videos."""
        for task in tasks:
            if task.completed:
                continue

            try:
                # Create temporary directory for this download
                temp_dir = tempfile.mkdtemp(prefix=f"whisper_video_{task.index}_")
                # Track directory immediately to ensure cleanup on failure
                with self.download_lock:
                    self.temp_dirs.append(temp_dir)

                downloader = AudioDownloader(output_dir=temp_dir)

                if self.verbose:
                    console.print(f"[cyan]Downloading:[/cyan] {task.title}")

                # Download audio
                audio_path, metadata = downloader.download(task.url, keep_audio=self.keep_audio)
                task.audio_path = audio_path
                task.metadata = metadata

                # Put in transcription queue
                self.transcribe_queue.put(task)

                # Update progress
                with self.stats_lock:
                    self.downloads_completed += 1
                    progress.update(download_task, completed=self.downloads_completed)

            except DownloadError as e:
                task.error = e
                self.transcribe_queue.put(task)  # Still queue for proper handling
                console.print(f"[red]✗ Download failed for video {task.index}:[/red] {e}")

            except (OSError, RuntimeError, ValueError) as e:
                task.error = e
                self.transcribe_queue.put(task)
                console.print(f"[red]✗ Unexpected error for video {task.index}:[/red] {e}")

        # Signal end of downloads
        self.transcribe_queue.put(None)

    def _transcribe_worker(
        self,
        language: str | None,
        output_format: str,
        progress: Progress,
        transcribe_task: TaskID,
    ) -> None:
        """Worker thread for transcribing videos."""
        if self.translate and self.model_manager:
            try:
                # Get translator from model manager for efficiency
                _ = self.model_manager.get_translator(
                    device=self.transcriber.device,
                    gpu_device=self.transcriber.gpu_device,
                )
            except (
                GPUError,
                TranslationError,
                RuntimeError,
                OSError,
                ValueError,
                ImportError,
            ) as e:
                console.print(f"[yellow]⚠ Failed to load translator:[/yellow] {e}")
                self.translate = False

        while True:
            # Get next task from queue
            task = self.transcribe_queue.get()
            if task is None:  # End signal
                break

            # Skip if download failed
            if task.error or not task.audio_path:
                with self.stats_lock:
                    self.failed_videos.append((task.index, task.title, str(task.error)))
                    self.transcriptions_completed += 1
                    progress.update(transcribe_task, completed=self.transcriptions_completed)
                continue

            try:
                if self.verbose:
                    console.print(f"[cyan]Transcribing:[/cyan] {task.title}")

                # Transcribe audio
                result = self.transcriber.transcribe(
                    task.audio_path,
                    language=language,
                    output_format=output_format,
                    metadata=task.metadata,
                    translate_to_english=self.translate,
                )

                # Handle results
                if isinstance(result, tuple):
                    transcript_original, transcript_translated = result
                    # Save both versions
                    output_he = task.output_path.with_name(
                        task.output_path.stem + "_he" + task.output_path.suffix
                    )
                    output_en = task.output_path.with_name(
                        task.output_path.stem + "_en" + task.output_path.suffix
                    )
                    save_transcript_file(
                        transcript_original, str(output_he), output_format, task.metadata
                    )
                    save_transcript_file(
                        transcript_translated, str(output_en), output_format, task.metadata
                    )
                else:
                    # Save single transcript
                    save_transcript_file(
                        result, str(task.output_path), output_format, task.metadata
                    )

                task.completed = True

                # Clean up audio file if not keeping
                if not self.keep_audio and task.audio_path and os.path.exists(task.audio_path):
                    os.remove(task.audio_path)

                # Clear GPU cache after each video
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except (TranscriptionError, GPUError, TranslationError) as e:
                with self.stats_lock:
                    self.failed_videos.append((task.index, task.title, str(e)))
                console.print(f"[red]✗ Transcription failed for video {task.index}:[/red] {e}")

            except (OSError, RuntimeError, ValueError) as e:
                with self.stats_lock:
                    self.failed_videos.append((task.index, task.title, str(e)))
                console.print(f"[red]✗ Unexpected error for video {task.index}:[/red] {e}")

            finally:
                # Update progress
                with self.stats_lock:
                    self.transcriptions_completed += 1
                    progress.update(transcribe_task, completed=self.transcriptions_completed)

    def _cleanup_temp_dirs(self) -> None:
        """Clean up all temporary directories."""
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _print_summary(self, total_videos: int, elapsed_time: float) -> None:
        """Print processing summary with performance metrics."""
        console.print("\n[bold green]Playlist Processing Complete![/bold green]")

        # Create summary table
        table = Table(title="Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        success_count = total_videos - len(self.failed_videos)
        table.add_row("Videos Processed", str(total_videos))
        table.add_row("Successful", str(success_count))
        table.add_row("Failed", str(len(self.failed_videos)))
        table.add_row("Total Time", f"{elapsed_time:.1f} seconds")
        table.add_row("Average Time/Video", f"{elapsed_time / total_videos:.1f} seconds")

        # Performance metrics
        if self.verbose:
            table.add_row("Download Workers", str(self.download_workers))
            if torch.cuda.is_available():
                table.add_row(
                    "GPU Memory (MB)",
                    f"{torch.cuda.memory_allocated() / 1024 / 1024:.1f}",
                )

        console.print(table)

        # Failed videos details
        if self.failed_videos:
            console.print("\n[bold red]Failed Videos:[/bold red]")
            for index, title, error in self.failed_videos:
                console.print(f"  {index:03d}. {title}")
                console.print(f"       [red]Error:[/red] {error}")

    def cleanup(self) -> None:
        """Clean up resources."""
        self._cleanup_temp_dirs()
