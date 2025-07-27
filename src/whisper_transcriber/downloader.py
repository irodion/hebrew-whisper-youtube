"""YouTube audio downloader module.

Copyright (c) 2025 Whisper Transcriber Contributors
Licensed under the MIT License - see LICENSE file for details.
"""
import os
import shutil
import tempfile
from typing import Optional, Dict, Any, Tuple
from yt_dlp import YoutubeDL
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn
from .utils import DownloadError, console, check_disk_space

class AudioDownloader:
    """Handle YouTube audio downloading with progress tracking."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="whisper_transcriber_")
        self._progress = None
        self._task_id = None
        
    def _progress_hook(self, d: Dict[str, Any]) -> None:
        """Hook for yt-dlp progress updates."""
        if d['status'] == 'downloading' and self._progress and self._task_id is not None:
            total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
            downloaded = d.get('downloaded_bytes', 0)
            
            if total > 0:
                self._progress.update(self._task_id, total=total, completed=downloaded)
                
    def download(self, url: str, keep_audio: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        Download audio from YouTube URL and extract metadata.
        
        Args:
            url: YouTube video URL
            keep_audio: Whether to keep the audio file after transcription
            
        Returns:
            Tuple of (path to the downloaded WAV file, video metadata dict)
            
        Raises:
            DownloadError: If download fails
        """
        if not check_disk_space(self.output_dir):
            raise DownloadError("Insufficient disk space (need at least 2GB free)")
            
        audio_path = os.path.join(self.output_dir, "audio.%(ext)s")
        
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'outtmpl': audio_path,
            'quiet': True,
            'no_warnings': True,
            'progress_hooks': [self._progress_hook],
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'postprocessor_args': [
                '-ar', '16000',  # Whisper prefers 16kHz
            ],
        }
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                console=console,
            ) as progress:
                self._progress = progress
                self._task_id = progress.add_task("Downloading audio...", total=None)
                
                with YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    
                    # Extract metadata
                    metadata = {
                        'title': info.get('title', 'Unknown'),
                        'duration': info.get('duration', 0),
                        'uploader': info.get('uploader', 'Unknown'),
                        'upload_date': info.get('upload_date', ''),
                        'description': info.get('description', ''),
                        'view_count': info.get('view_count', 0),
                        'like_count': info.get('like_count', 0),
                        'channel': info.get('channel', ''),
                        'channel_id': info.get('channel_id', ''),
                        'webpage_url': info.get('webpage_url', url)
                    }
                    
                    console.print(f"[bold]Title:[/bold] {metadata['title']}")
                    if metadata['duration']:
                        from .utils import format_time
                        console.print(f"[bold]Duration:[/bold] {format_time(metadata['duration'])}")
                    
                    ydl.download([url])
                    
        except Exception as e:
            raise DownloadError(f"Failed to download audio: {str(e)}") from e
        
        wav_path = os.path.join(self.output_dir, "audio.wav")
        if not os.path.exists(wav_path):
            raise DownloadError("Audio conversion to WAV failed")
            
        return wav_path, metadata
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        # Get the system temp directory in a cross-platform way
        temp_dir = tempfile.gettempdir()
        
        # Check if output_dir exists and is within the temp directory
        if os.path.exists(self.output_dir):
            # Resolve paths to handle symlinks and relative paths
            output_path = os.path.realpath(self.output_dir)
            temp_path = os.path.realpath(temp_dir)
            
            # Only delete if the directory is within the temp directory
            if output_path.startswith(temp_path + os.sep) or output_path == temp_path:
                shutil.rmtree(self.output_dir, ignore_errors=True)