"""Whisper transcription module with GPU support.

Copyright (c) 2025 Whisper Transcriber Contributors
Licensed under the MIT License - see LICENSE file for details.
"""

import json
import os
from typing import Any, ClassVar

import torch
from faster_whisper import WhisperModel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .resource_manager import managed_gpu_memory
from .utils import GPUError, TranslationError, check_cuda_availability, console


class WhisperTranscriber:
    """Handle audio transcription using Faster Whisper with GPU acceleration."""

    AVAILABLE_MODELS: ClassVar[list[str]] = [
        # Standard OpenAI models
        "tiny",
        "base",
        "small",
        "medium",
        "large",
        "large-v2",
        "large-v3",
        # Hebrew-optimized ivrit-ai models
        "ivrit-small",
        "ivrit-large",
        "ivrit-turbo",
    ]

    # Mapping of friendly names to HuggingFace model IDs
    HEBREW_MODELS: ClassVar[dict[str, str]] = {
        "ivrit-small": "ivrit-ai/whisper-large-v3-turbo-ct2",  # Turbo model for speed
        "ivrit-large": "ivrit-ai/whisper-large-v3-ct2",  # Full large model
        "ivrit-turbo": "ivrit-ai/whisper-large-v3-turbo-ct2",  # Same as small, best balance
    }

    def __init__(
        self, model_size: str = "ivrit-turbo", device: str | None = None, gpu_device: int = 0
    ) -> None:
        """Initialize the transcriber.

        Args:
            model_size: Whisper model size (including Hebrew models like 'ivrit-turbo')
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            gpu_device: GPU device ID to use for translation (default: 0)
        """
        self.model_size = model_size
        self.device = device or self._auto_select_device()
        self.gpu_device = gpu_device
        self.compute_type = self._select_compute_type()
        self.model: WhisperModel | None = None
        self.is_hebrew_model = model_size in self.HEBREW_MODELS

    def _auto_select_device(self) -> str:
        """Auto-select best available device with fallback protection."""
        cuda_available, gpu_info = check_cuda_availability()

        if cuda_available:
            console.print(f"[green]✓[/green] GPU detected: {gpu_info}")
            console.print("[dim]Note: Will fallback to CPU if GPU initialization fails[/dim]")
            return "cuda"
        console.print("[yellow]⚠[/yellow] No GPU detected, using CPU (slower)")
        return "cpu"

    def _select_compute_type(self) -> str:
        """Select optimal compute type based on device."""
        if self.device == "cuda":
            # float16 is fastest on GPU with minimal quality loss
            return "float16"
        # int8 is efficient on CPU
        return "int8"

    @classmethod
    def get_hebrew_models(cls) -> dict[str, str]:
        """Get Hebrew models mapping."""
        return cls.HEBREW_MODELS

    def _load_model(self) -> None:
        """Load the Whisper model."""
        if self.model is not None:
            return

        # Determine model path/ID
        if self.is_hebrew_model:
            model_id = self.HEBREW_MODELS[self.model_size]
            console.print(
                f"Loading Hebrew-optimized Whisper model ({self.model_size}) "
                f"on {self.device.upper()}..."
            )
            console.print(f"[dim]Model: {model_id}[/dim]")
        else:
            model_id = self.model_size
            console.print(f"Loading Whisper {self.model_size} model on {self.device.upper()}...")

        try:
            self.model = WhisperModel(
                model_id,
                device=self.device,
                compute_type=self.compute_type,
                download_root=os.path.expanduser("~/.cache/whisper"),
            )

            if self.is_hebrew_model:
                console.print("[green]✓[/green] Hebrew-optimized model loaded successfully")

        except (RuntimeError, torch.cuda.OutOfMemoryError, ValueError, OSError) as e:
            error_msg = str(e).lower()
            # Check for GPU/CUDA related errors
            if any(
                keyword in error_msg
                for keyword in ["cuda", "gpu", "cudnn", "cublas", "curand", "cusparse"]
            ):
                if self.device == "cuda":
                    console.print(f"[yellow]⚠ GPU initialization failed:[/yellow] {e!s}")
                    console.print("[yellow]Falling back to CPU mode...[/yellow]")

                    # Retry with CPU
                    self.device = "cpu"
                    self.compute_type = "int8"  # CPU-optimized

                    try:
                        self.model = WhisperModel(
                            model_id,
                            device=self.device,
                            compute_type=self.compute_type,
                            download_root=os.path.expanduser("~/.cache/whisper"),
                        )
                        console.print("[green]✓[/green] Successfully loaded model on CPU")
                        return
                    except (RuntimeError, OSError, ValueError) as cpu_error:
                        gpu_cpu_error_msg = (
                            f"Both GPU and CPU initialization failed. GPU: {e!s}, "
                            f"CPU: {cpu_error!s}"
                        )
                        raise GPUError(gpu_cpu_error_msg) from cpu_error
                else:
                    gpu_error_msg = f"GPU initialization failed: {e!s}"
                    raise GPUError(gpu_error_msg) from e
            else:
                raise

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        task: str = "transcribe",
        output_format: str = "text",
        metadata: dict[str, Any] | None = None,
        translate_to_english: bool = False,
    ) -> str | tuple[str, str]:
        """Transcribe audio file.

        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detect)
            task: 'transcribe' or 'translate'
            output_format: 'text', 'srt', 'vtt', or 'json'
            metadata: Optional video metadata dict for enhanced output
            translate_to_english: If True and Hebrew detected, also translate to English

        Returns:
            Transcribed text in requested format, or tuple of (original, translated) if translating
        """
        self._load_model()

        # Handle Hebrew model specifics
        language = self._handle_hebrew_model_setup(language, task)

        # Perform transcription
        segments_list, info = self._perform_transcription(audio_path, language, output_format)

        # Format original output
        original = self._format_output(segments_list, output_format, metadata, info)

        # Handle translation if requested
        if self._should_translate(translate_to_english, language, info):
            translated = self._perform_translation(segments_list, output_format, metadata, info)
            if translated:
                return original, translated

        return original

    def _handle_hebrew_model_setup(self, language: str | None, task: str) -> str | None:
        """Handle Hebrew model specific setup and warnings."""
        if not self.is_hebrew_model:
            return language

        if task == "translate":
            console.print(
                "[yellow]⚠ Warning:[/yellow] Translation not supported "
                "with Hebrew models, using transcription instead"
            )

        # Force Hebrew language for Hebrew models (language detection is degraded)
        if not language:
            language = "he"
            console.print("[dim]Forcing language to Hebrew for optimized model[/dim]")

        return language

    def _get_transcription_options(
        self, language: str | None, output_format: str
    ) -> dict[str, Any]:
        """Get transcription options for the Whisper model."""
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

        return options

    def _perform_transcription(
        self, audio_path: str, language: str | None, output_format: str
    ) -> tuple[list[Any], Any]:
        """Perform the actual transcription and return segments and info."""
        options = self._get_transcription_options(language, output_format)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            progress.add_task("Transcribing audio...", total=None)

            assert self.model is not None  # Should be loaded by _load_model()
            segments, info = self.model.transcribe(audio_path, **options)

            # Convert generator to list to process
            segments_list = list(segments)

            # Show detected language if auto-detected
            if not language and info.language:
                console.print(f"[bold]Detected language:[/bold] {info.language}")

        return segments_list, info

    def _format_output(
        self,
        segments_list: list[Any],
        output_format: str,
        metadata: dict[str, Any] | None,
        info: Any,
    ) -> str:
        """Format segments into the requested output format."""
        if output_format == "text":
            return self._format_as_text(segments_list)
        if output_format == "srt":
            return self._format_as_srt(segments_list)
        if output_format == "vtt":
            return self._format_as_vtt(segments_list)
        if output_format == "json":
            return self._format_as_json(segments_list, metadata or {}, info)
        error_msg = f"Unknown output format: {output_format}"
        raise ValueError(error_msg)

    def _should_translate(
        self, translate_to_english: bool, language: str | None, info: Any
    ) -> bool:
        """Check if translation should be performed."""
        return translate_to_english and (
            language == "he" or (info.language and info.language == "he")
        )

    def _perform_translation(
        self,
        segments_list: list[Any],
        output_format: str,
        metadata: dict[str, Any] | None,
        info: Any,
    ) -> str | None:
        """Perform translation and return formatted translated output."""
        console.rule("[bold]Step 2.5/3: Translating to English[/bold]")

        # Use GPU memory management context
        with managed_gpu_memory():
            # Get translator from model manager for efficiency
            from .model_manager import get_model_manager

            model_manager = get_model_manager()

            try:
                translator = model_manager.get_translator(
                    device=self.device, gpu_device=self.gpu_device
                )

                # Translate segments
                translated_segments = translator.translate_segments(segments_list)

                # Format translated output
                if output_format == "text":
                    translated = self._format_as_text(translated_segments)
                elif output_format == "srt":
                    translated = self._format_as_srt(translated_segments)
                elif output_format == "vtt":
                    translated = self._format_as_vtt(translated_segments)
                elif output_format == "json":
                    translated = self._format_as_json(translated_segments, metadata or {}, info)
                else:
                    return None  # Should not happen due to earlier validation

                return translated

            except (GPUError, TranslationError) as e:
                console.print(f"[yellow]⚠ Translation failed:[/yellow] {e!s}")
                console.print("[yellow]Continuing with Hebrew transcript only[/yellow]")
                return None

    def _normalize_segment(self, segment: Any) -> dict[str, Any]:
        """Convert segment object or dictionary into consistent dictionary format.

        Args:
            segment: Either a segment object with .start, .end, .text attributes
                    or a dictionary with "start", "end", "text" keys

        Returns:
            dict: Dictionary with "start", "end", "text" keys
        """
        if isinstance(segment, dict):
            return {"start": segment["start"], "end": segment["end"], "text": segment["text"]}
        return {"start": segment.start, "end": segment.end, "text": segment.text}

    def _format_as_text(self, segments: list[Any]) -> str:
        """Format segments as plain text."""
        normalized_segments = [self._normalize_segment(segment) for segment in segments]
        return "\n".join(segment["text"].strip() for segment in normalized_segments)

    def _format_as_srt(self, segments: list[Any]) -> str:
        """Format segments as SRT subtitles."""
        normalized_segments = [self._normalize_segment(segment) for segment in segments]
        srt_output = []
        for i, segment in enumerate(normalized_segments, 1):
            start = self._seconds_to_srt_time(segment["start"])
            end = self._seconds_to_srt_time(segment["end"])
            text = segment["text"].strip()
            srt_output.append(f"{i}\n{start} --> {end}\n{text}\n")
        return "\n".join(srt_output)

    def _format_as_vtt(self, segments: list[Any]) -> str:
        """Format segments as WebVTT subtitles."""
        normalized_segments = [self._normalize_segment(segment) for segment in segments]
        vtt_output = ["WEBVTT\n"]
        for segment in normalized_segments:
            start = self._seconds_to_vtt_time(segment["start"])
            end = self._seconds_to_vtt_time(segment["end"])
            text = segment["text"].strip()
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

    def _format_as_json(self, segments: list[Any], metadata: dict[str, Any], info: Any) -> str:
        """Format segments and metadata as JSON."""
        from .metadata_formatter import MetadataFormatter

        # Format segments with timestamps
        normalized_segments = [self._normalize_segment(segment) for segment in segments]
        transcript_segments = []
        for segment in normalized_segments:
            transcript_segments.append(
                {
                    "start": round(segment["start"], 3),
                    "end": round(segment["end"], 3),
                    "text": segment["text"].strip(),
                }
            )

        # Clean metadata for JSON
        clean_metadata = MetadataFormatter.format_json_metadata(metadata.copy())

        # Add transcription info
        clean_metadata["detected_language"] = getattr(info, "language", None)
        clean_metadata["transcription_model"] = self.model_size

        output = {"metadata": clean_metadata, "transcript": transcript_segments}

        return json.dumps(output, ensure_ascii=False, indent=2)
