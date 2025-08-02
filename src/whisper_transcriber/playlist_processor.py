"""YouTube playlist processing module.

Copyright (c) 2025 Whisper Transcriber Contributors
Licensed under the MIT License - see LICENSE file for details.
"""

import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from yt_dlp import YoutubeDL

from .downloader import AudioDownloader
from .transcriber import WhisperTranscriber
from .transcript_io import save_transcript_file
from .utils import DownloadError, GPUError, PlaylistSizeLimitError, TranscriptionError, console


class PlaylistProcessor:
    """Handle batch processing of YouTube playlists.

    This enhanced version includes concurrent download capabilities for improved
    performance while maintaining sequential transcription for GPU efficiency.

    Constants:
        MAX_FILENAME_LENGTH: Maximum length for generated filenames (200 chars)
        MAX_PLAYLIST_SIZE: Maximum number of videos that can be processed (9999)
        INDEX_PADDING: Number of digits used for video index padding (4)
        DEFAULT_DOWNLOAD_WORKERS: Default number of concurrent download workers (3)
        DOWNLOAD_QUEUE_SIZE: Maximum downloads to queue ahead (5)
    """

    MAX_FILENAME_LENGTH = 200
    MAX_PLAYLIST_SIZE = 9999  # Based on 4-digit padding (0001-9999)
    INDEX_PADDING = 4
    DEFAULT_DOWNLOAD_WORKERS = 3  # Optimal for most connections
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
        translation_context: bool = True,
        context_lines: int = 2,
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
            download_workers: Number of concurrent download workers (default: 3)
            translation_context: Whether to use translation context
            context_lines: Number of context lines to use
        """
        if not 0 <= context_lines <= 5:
            msg = "context_lines must be between 0 and 5"
            raise ValueError(msg)
        self.output_dir = output_dir
        self.transcriber = transcriber
        self.max_videos = max_videos
        self.start_index = start_index
        self.keep_audio = keep_audio
        self.translate = translate
        self.verbose = verbose
        self.download_workers = download_workers or self.DEFAULT_DOWNLOAD_WORKERS
        self.translation_context = translation_context
        self.context_lines = context_lines

        # Concurrent processing components
        self.download_queue: queue.Queue[Any] = queue.Queue(maxsize=self.DOWNLOAD_QUEUE_SIZE)
        self.download_complete_event = threading.Event()
        self.download_lock = threading.Lock()

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
        # Extract playlist information
        console.print("[bold]Extracting playlist information...[/bold]")

        try:
            playlist_info = self._extract_playlist_info(playlist_url)
            videos = playlist_info["entries"]
            playlist_title = playlist_info.get("title")

            console.print(
                f"[green]✓[/green] Found playlist: {playlist_title or 'Unknown Playlist'}"
            )
            console.print(f"[green]✓[/green] Total videos: {len(videos)}")

            # Validate playlist size
            if len(videos) > self.MAX_PLAYLIST_SIZE:
                error_msg = (
                    f"Playlist too large: {len(videos)} videos found, "
                    f"maximum supported is {self.MAX_PLAYLIST_SIZE}"
                )
                console.print(f"[red]✗ {error_msg}[/red]")
                raise PlaylistSizeLimitError(error_msg)

        except Exception as e:
            console.print(f"[red]✗ Failed to extract playlist:[/red] {e}")
            error_msg = f"Failed to extract playlist: {e}"
            raise DownloadError(error_msg) from e

        # Apply filtering
        videos = self._filter_videos(videos)

        if not videos:
            console.print("[yellow]No videos to process after filtering[/yellow]")
            return

        console.print(
            f"[cyan]Processing {len(videos)} videos with "
            f"{self.download_workers} download workers...[/cyan]"
        )

        # Process videos with concurrent downloads
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
            download_task = progress.add_task("Downloading", total=len(videos))
            transcribe_task = progress.add_task("Transcribing", total=len(videos))

            # Prepare video tasks
            video_tasks = []
            for i, video_info in enumerate(videos, 1):
                video_title = video_info.get("title", f"Video {i}")
                video_url = video_info.get("webpage_url") or video_info.get("url")

                if not video_url:
                    console.print(f"[yellow]⚠ Skipping video {i}: No URL found[/yellow]")
                    failed_videos.append((i, video_title, "No URL found"))
                    progress.update(download_task, advance=1)
                    progress.update(transcribe_task, advance=1)
                    continue

                # Generate safe filename
                safe_filename = self._generate_safe_filename(i, video_title, output_format)
                output_path = self.output_dir / safe_filename

                video_tasks.append(
                    {
                        "index": i,
                        "title": video_title,
                        "url": video_url,
                        "output_path": output_path,
                        "video_info": video_info,
                        "playlist_title": playlist_title,
                    }
                )

            # Process with concurrent downloads and sequential transcription
            results = self._process_videos_concurrently(
                video_tasks, language, output_format, progress, download_task, transcribe_task
            )

            # Count results
            for result in results:
                if result["success"]:
                    success_count += 1
                else:
                    failed_videos.append((result["index"], result["title"], result["error"]))

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
            Safe filename with format: NNNN_video_title.ext
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

    def _process_videos_concurrently(
        self,
        video_tasks: list[dict[str, Any]],
        language: str | None,
        output_format: str,
        progress: Progress,
        download_task: Any,
        transcribe_task: Any,
    ) -> list[dict[str, Any]]:
        """Process videos with concurrent downloads and sequential transcription.

        Args:
            video_tasks: List of video task dictionaries
            language: Source language for transcription
            output_format: Output format
            progress: Progress object for tracking
            download_task: Download progress task ID
            transcribe_task: Transcription progress task ID

        Returns:
            List of result dictionaries with success status
        """
        results = []
        download_futures = []
        transcribe_index = 0
        downloads_completed = 0

        # Thread synchronization for shared state
        state_lock = threading.Lock()
        progress_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=self.download_workers) as download_executor:
            # Start initial downloads
            for i in range(min(self.download_workers, len(video_tasks))):
                task = video_tasks[i]
                future = download_executor.submit(
                    self._download_video_concurrent,
                    task["url"],
                    task["index"],
                    task["title"],
                )
                download_futures.append((i, future))

            # Process downloads and transcriptions
            while transcribe_index < len(video_tasks) or download_futures:
                # Check completed downloads
                completed_downloads = []
                for i, (task_idx, future) in enumerate(download_futures):
                    if future.done():
                        completed_downloads.append((i, task_idx, future))

                # Process completed downloads
                for _, task_idx, future in sorted(completed_downloads, key=lambda x: x[1]):
                    download_futures.remove((task_idx, future))

                    try:
                        audio_path, metadata = future.result()
                        with state_lock:
                            video_tasks[task_idx]["audio_path"] = audio_path
                            video_tasks[task_idx]["metadata"] = metadata
                            downloads_completed += 1
                            current_downloads = downloads_completed

                        with progress_lock:
                            progress.update(download_task, completed=current_downloads)

                        # Start next download if available
                        with state_lock:
                            next_idx = downloads_completed + len(download_futures)

                        if next_idx < len(video_tasks):
                            next_task = video_tasks[next_idx]
                            next_future = download_executor.submit(
                                self._download_video_concurrent,
                                next_task["url"],
                                next_task["index"],
                                next_task["title"],
                            )
                            download_futures.append((next_idx, next_future))

                    except (DownloadError, OSError, RuntimeError) as e:
                        with state_lock:
                            video_tasks[task_idx]["error"] = str(e)
                            downloads_completed += 1
                            current_downloads = downloads_completed

                        with progress_lock:
                            progress.update(download_task, completed=current_downloads)

                # Process ready transcriptions in order
                while transcribe_index < len(video_tasks):
                    with state_lock:
                        task = video_tasks[transcribe_index]

                        # Check if download is complete
                        if "audio_path" not in task and "error" not in task:
                            break  # Wait for download to complete

                        # Copy task data for thread safety
                        task_copy = task.copy()

                    # Process transcription outside of lock
                    if "error" in task_copy:
                        results.append(
                            {
                                "index": task_copy["index"],
                                "title": task_copy["title"],
                                "success": False,
                                "error": task_copy["error"],
                            }
                        )
                    else:
                        result = self._transcribe_video_sequential(
                            task_copy["audio_path"],
                            task_copy["output_path"],
                            language,
                            output_format,
                            task_copy["metadata"],
                            task_copy["video_info"],
                            task_copy["index"],
                            task_copy["title"],
                            task_copy.get("playlist_title"),
                        )
                        results.append(result)

                    transcribe_index += 1
                    with progress_lock:
                        progress.update(transcribe_task, completed=transcribe_index)

                    # Clear GPU cache after each transcription
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Small sleep to prevent busy waiting
                if download_futures and transcribe_index < len(video_tasks):
                    time.sleep(0.1)

        return results

    def _download_video_concurrent(
        self,
        video_url: str,
        index: int,
        title: str,
    ) -> tuple[str, dict[str, Any]]:
        """Download a single video (thread-safe for concurrent execution).

        Args:
            video_url: URL of the video
            index: Video index in playlist
            title: Video title

        Returns:
            Tuple of (audio_path, metadata)
        """
        if self.verbose:
            console.print(f"[cyan]Downloading video {index}:[/cyan] {title}")

        try:
            # Create a new downloader instance for thread safety
            downloader = AudioDownloader()
            audio_path, metadata = downloader.download(video_url, keep_audio=self.keep_audio)
            return audio_path, metadata
        except (DownloadError, OSError, RuntimeError, ValueError) as e:
            error_msg = f"Download failed: {e}"
            raise DownloadError(error_msg) from e

    def _transcribe_video_sequential(
        self,
        audio_path: str,
        output_path: Path,
        language: str | None,
        output_format: str,
        metadata: dict[str, Any],
        video_info: dict[str, Any],
        index: int,
        title: str,
        playlist_title: str | None = None,
    ) -> dict[str, Any]:
        """Transcribe a single video (sequential execution for GPU efficiency).

        Args:
            audio_path: Path to downloaded audio
            output_path: Output file path
            language: Source language
            output_format: Output format
            metadata: Download metadata
            video_info: Video information
            index: Video index
            title: Video title
            playlist_title: Optional playlist title for context

        Returns:
            Result dictionary with success status
        """
        if self.verbose:
            console.print(f"[cyan]Transcribing video {index}:[/cyan] {title}")

        try:
            # Merge video info into metadata
            metadata.update(video_info)

            # Add playlist title to metadata for translation context
            if playlist_title:
                metadata["playlist_title"] = playlist_title

            # Transcribe
            result = self.transcriber.transcribe(
                audio_path,
                language=language,
                output_format=output_format,
                metadata=metadata,
                translate_to_english=self.translate,
                use_translation_context=self.translation_context,
                context_lines=self.context_lines,
            )

            # Handle translation results
            if isinstance(result, tuple):
                transcript_original, transcript_translated = result
                # Save both versions
                output_he = output_path.with_name(output_path.stem + "_he" + output_path.suffix)
                output_en = output_path.with_name(output_path.stem + "_en" + output_path.suffix)
                self._save_transcript(transcript_original, output_he, output_format, metadata)
                if transcript_translated is not None:
                    self._save_transcript(transcript_translated, output_en, output_format, metadata)
            else:
                self._save_transcript(result, output_path, output_format, metadata)

            console.print(f"[green]✓ Completed video {index}[/green]")

            return {
                "index": index,
                "title": title,
                "success": True,
                "error": None,
            }

        except (TranscriptionError, GPUError, DownloadError, OSError, RuntimeError) as e:
            console.print(f"[red]✗ Failed to process video {index}:[/red] {e}")
            return {
                "index": index,
                "title": title,
                "success": False,
                "error": str(e),
            }

    def _process_single_video(
        self,
        video_url: str,
        output_path: Path,
        language: str | None,
        output_format: str,
        video_info: dict[str, Any],
        playlist_title: str | None = None,
    ) -> None:
        """Process a single video from the playlist."""
        # Download audio
        try:
            downloader = AudioDownloader()
            audio_path, metadata = downloader.download(video_url, keep_audio=self.keep_audio)
        except DownloadError as e:
            msg = f"Download failed: {e}"
            raise TranscriptionError(msg) from e

        # Merge video info into metadata
        metadata.update(video_info)

        # Add playlist title to metadata for translation context
        if playlist_title:
            metadata["playlist_title"] = playlist_title

        # Transcribe
        try:
            result = self.transcriber.transcribe(
                audio_path,
                language=language,
                output_format=output_format,
                metadata=metadata,
                translate_to_english=self.translate,
                use_translation_context=self.translation_context,
                context_lines=self.context_lines,
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
        playlist_title: str | None,
    ) -> None:
        """Print processing summary with error categorization."""
        console.print("\n[bold green]Playlist Processing Complete![/bold green]")
        console.print(f"Playlist: {playlist_title or 'Unknown Playlist'}")

        # Summary table
        table = Table(title="Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="white")

        table.add_row("Videos Processed", str(success_count + len(failed_videos)))
        table.add_row("Successful", str(success_count))
        table.add_row("Failed", str(len(failed_videos)))

        console.print(table)

        # Categorize and print errors
        if failed_videos:
            error_categories = self._categorize_errors(failed_videos)
            self._print_categorized_errors(error_categories)

    def _categorize_errors(
        self, failed_videos: list[tuple[int, str, str]]
    ) -> dict[str, list[tuple[int, str, str]]]:
        """Categorize errors by type.

        Args:
            failed_videos: List of (index, title, error) tuples

        Returns:
            Dict mapping error types to lists of failed videos
        """
        error_categories: dict[str, list[tuple[int, str, str]]] = {
            "download": [],
            "transcription": [],
            "gpu": [],
            "translation": [],
            "other": [],
        }

        for index, title, error in failed_videos:
            error_lower = error.lower()
            if "download" in error_lower or "url" in error_lower:
                error_categories["download"].append((index, title, error))
            elif "transcription" in error_lower or "whisper" in error_lower:
                error_categories["transcription"].append((index, title, error))
            elif "gpu" in error_lower or "cuda" in error_lower or "memory" in error_lower:
                error_categories["gpu"].append((index, title, error))
            elif "translation" in error_lower or "dictalm" in error_lower:
                error_categories["translation"].append((index, title, error))
            else:
                error_categories["other"].append((index, title, error))

        return error_categories

    def _print_categorized_errors(
        self, error_categories: dict[str, list[tuple[int, str, str]]]
    ) -> None:
        """Print categorized errors.

        Args:
            error_categories: Dict mapping error types to lists of failed videos
        """
        console.print("\n[bold red]Failed Videos by Category:[/bold red]")

        if error_categories["download"]:
            console.print("\n[yellow]Download Errors:[/yellow]")
            for index, title, error in error_categories["download"]:
                console.print(f"  {index:03d}. {title}")
                console.print(f"       [red]Error:[/red] {error}")

        if error_categories["transcription"]:
            console.print("\n[yellow]Transcription Errors:[/yellow]")
            for index, title, error in error_categories["transcription"]:
                console.print(f"  {index:03d}. {title}")
                console.print(f"       [red]Error:[/red] {error}")

        if error_categories["gpu"]:
            console.print("\n[yellow]GPU/Memory Errors:[/yellow]")
            for index, title, error in error_categories["gpu"]:
                console.print(f"  {index:03d}. {title}")
                console.print(f"       [red]Error:[/red] {error}")
            console.print("[dim]Tip: Try using --device cpu or reducing batch size[/dim]")

        if error_categories["translation"]:
            console.print("\n[yellow]Translation Errors:[/yellow]")
            for index, title, error in error_categories["translation"]:
                console.print(f"  {index:03d}. {title}")
                console.print(f"       [red]Error:[/red] {error}")

        if error_categories["other"]:
            console.print("\n[yellow]Other Errors:[/yellow]")
            for index, title, error in error_categories["other"]:
                console.print(f"  {index:03d}. {title}")
                console.print(f"       [red]Error:[/red] {error}")

    def cleanup(self) -> None:
        """Clean up resources used by the playlist processor."""
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clear any remaining queue items
        if hasattr(self, "download_queue"):
            while not self.download_queue.empty():
                try:
                    self.download_queue.get_nowait()
                except queue.Empty:
                    break
