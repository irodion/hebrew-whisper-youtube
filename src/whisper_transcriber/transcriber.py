"""Whisper transcription module with GPU support.

Copyright (c) 2025 Whisper Transcriber Contributors
Licensed under the MIT License - see LICENSE file for details.
"""
import os
from typing import Optional, List, Any
from faster_whisper import WhisperModel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from .utils import GPUError, console, check_cuda_availability

class WhisperTranscriber:
    """Handle audio transcription using Faster Whisper with GPU acceleration."""
    
    AVAILABLE_MODELS = [
        # Standard OpenAI models
        "tiny", "base", "small", "medium", "large", "large-v2", "large-v3",
        # Hebrew-optimized ivrit-ai models
        "ivrit-small", "ivrit-large", "ivrit-turbo"
    ]
    
    # Mapping of friendly names to HuggingFace model IDs
    HEBREW_MODELS = {
        "ivrit-small": "ivrit-ai/whisper-large-v3-turbo-ct2",  # Turbo model for speed
        "ivrit-large": "ivrit-ai/whisper-large-v3-ct2",        # Full large model
        "ivrit-turbo": "ivrit-ai/whisper-large-v3-turbo-ct2",  # Same as small, best balance
    }
    
    def __init__(self, model_size: str = "ivrit-turbo", device: Optional[str] = None):
        """
        Initialize the transcriber.
        
        Args:
            model_size: Whisper model size (including Hebrew models like 'ivrit-turbo')
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_size = model_size
        self.device = device or self._auto_select_device()
        self.compute_type = self._select_compute_type()
        self.model = None
        self.is_hebrew_model = model_size in self.HEBREW_MODELS
        
    def _auto_select_device(self) -> str:
        """Auto-select best available device with fallback protection."""
        cuda_available, gpu_info = check_cuda_availability()
        
        if cuda_available:
            console.print(f"[green]✓[/green] GPU detected: {gpu_info}")
            console.print("[dim]Note: Will fallback to CPU if GPU initialization fails[/dim]")
            return "cuda"
        else:
            console.print("[yellow]⚠[/yellow] No GPU detected, using CPU (slower)")
            return "cpu"
            
    def _select_compute_type(self) -> str:
        """Select optimal compute type based on device."""
        if self.device == "cuda":
            # float16 is fastest on GPU with minimal quality loss
            return "float16"
        else:
            # int8 is efficient on CPU
            return "int8"
            
    def _load_model(self) -> None:
        """Load the Whisper model."""
        if self.model is not None:
            return
        
        # Determine model path/ID
        if self.is_hebrew_model:
            model_id = self.HEBREW_MODELS[self.model_size]
            console.print(f"Loading Hebrew-optimized Whisper model ({self.model_size}) on {self.device.upper()}...")
            console.print(f"[dim]Model: {model_id}[/dim]")
        else:
            model_id = self.model_size
            console.print(f"Loading Whisper {self.model_size} model on {self.device.upper()}...")
        
        try:
            self.model = WhisperModel(
                model_id,
                device=self.device,
                compute_type=self.compute_type,
                download_root=os.path.expanduser("~/.cache/whisper")
            )
            
            if self.is_hebrew_model:
                console.print("[green]✓[/green] Hebrew-optimized model loaded successfully")
                
        except Exception as e:
            error_msg = str(e).lower()
            # Check for GPU/CUDA related errors
            if any(keyword in error_msg for keyword in ["cuda", "gpu", "cudnn", "cublas", "curand", "cusparse"]):
                if self.device == "cuda":
                    console.print(f"[yellow]⚠ GPU initialization failed:[/yellow] {str(e)}")
                    console.print("[yellow]Falling back to CPU mode...[/yellow]")
                    
                    # Retry with CPU
                    self.device = "cpu"
                    self.compute_type = "int8"  # CPU-optimized
                    
                    try:
                        self.model = WhisperModel(
                            model_id,
                            device=self.device,
                            compute_type=self.compute_type,
                            download_root=os.path.expanduser("~/.cache/whisper")
                        )
                        console.print("[green]✓[/green] Successfully loaded model on CPU")
                        return
                    except Exception as cpu_error:
                        raise GPUError(f"Both GPU and CPU initialization failed. GPU: {str(e)}, CPU: {str(cpu_error)}") from cpu_error
                else:
                    raise GPUError(f"GPU initialization failed: {str(e)}") from e
            else:
                raise
            
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        output_format: str = "text"
    ) -> str:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detect)
            task: 'transcribe' or 'translate'
            output_format: 'text', 'srt', or 'vtt'
            
        Returns:
            Transcribed text in requested format
        """
        self._load_model()
        
        # For Hebrew models, warn about translation and force Hebrew language
        if self.is_hebrew_model:
            if task == "translate":
                console.print("[yellow]⚠ Warning:[/yellow] Translation not supported with Hebrew models, using transcription instead")
                task = "transcribe"
            
            # Force Hebrew language for Hebrew models (language detection is degraded)
            if not language:
                language = "he"
                console.print("[dim]Forcing language to Hebrew for optimized model[/dim]")
        
        # Transcription options
        options = {
            "beam_size": 5,
            "best_of": 5,
            "patience": 1,
            "length_penalty": 1,
            "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": True,
            "initial_prompt": None,
            "word_timestamps": output_format != "text",
            "vad_filter": True,  # Voice activity detection
            "vad_parameters": {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "max_speech_duration_s": float("inf"),
                "min_silence_duration_ms": 2000,
                "speech_pad_ms": 400,
            },
        }
        
        if language:
            options["language"] = language
            
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            progress.add_task("Transcribing audio...", total=None)
            
            segments, info = self.model.transcribe(audio_path, **options)
            
            # Convert generator to list to process
            segments_list = list(segments)
            
            # Show detected language if auto-detected
            if not language and info.language:
                console.print(f"[bold]Detected language:[/bold] {info.language}")
                
        # Format output based on requested format
        if output_format == "text":
            return self._format_as_text(segments_list)
        elif output_format == "srt":
            return self._format_as_srt(segments_list)
        elif output_format == "vtt":
            return self._format_as_vtt(segments_list)
        else:
            raise ValueError(f"Unknown output format: {output_format}")
            
    def _format_as_text(self, segments: List[Any]) -> str:
        """Format segments as plain text."""
        return "\n".join(segment.text.strip() for segment in segments)
        
    def _format_as_srt(self, segments: List[Any]) -> str:
        """Format segments as SRT subtitles."""
        srt_output = []
        for i, segment in enumerate(segments, 1):
            start = self._seconds_to_srt_time(segment.start)
            end = self._seconds_to_srt_time(segment.end)
            text = segment.text.strip()
            srt_output.append(f"{i}\n{start} --> {end}\n{text}\n")
        return "\n".join(srt_output)
        
    def _format_as_vtt(self, segments: List[Any]) -> str:
        """Format segments as WebVTT subtitles."""
        vtt_output = ["WEBVTT\n"]
        for segment in segments:
            start = self._seconds_to_vtt_time(segment.start)
            end = self._seconds_to_vtt_time(segment.end)
            text = segment.text.strip()
            vtt_output.append(f"{start} --> {end}\n{text}\n")
        return "\n".join(vtt_output)
        
    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        """Convert seconds to SRT timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")
        
    @staticmethod
    def _seconds_to_vtt_time(seconds: float) -> str:
        """Convert seconds to VTT timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"