"""DictaLM 2.0 translation module for Hebrew to English.

Copyright (c) 2025 Whisper Transcriber Contributors
Licensed under the MIT License - see LICENSE file for details.
"""

from typing import Any, ClassVar

import torch
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import GPUError, TranslationError, check_cuda_availability, console


class DictaLMTranslator:
    """Handle Hebrew to English translation using DictaLM 2.0-Instruct model."""

    MODEL_NAME: ClassVar[str] = "dicta-il/dictalm2.0-instruct"

    def __init__(self, device: str | None = None, gpu_device: int = 0) -> None:
        """Initialize the translator.

        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            gpu_device: GPU device ID to use for translation (default: 0)
        """
        self.device = device or self._auto_select_device()
        self.gpu_device = gpu_device
        self.model: AutoModelForCausalLM | None = None
        self.tokenizer: AutoTokenizer | None = None

    def _auto_select_device(self) -> str:
        """Auto-select best available device with fallback protection."""
        cuda_available, gpu_info = check_cuda_availability()

        if cuda_available:
            console.print(f"[green]✓[/green] GPU detected for translation: {gpu_info}")
            console.print("[dim]Note: Will fallback to CPU if GPU initialization fails[/dim]")
            return "cuda"
        console.print("[yellow]⚠[/yellow] No GPU detected for translation, using CPU (slower)")
        return "cpu"

    def _load_model(self) -> None:
        """Load the DictaLM model and tokenizer."""
        if self.model is not None and self.tokenizer is not None:
            return

        console.print(f"Loading DictaLM 2.0-Instruct model on {self.device.upper()}...")
        console.print(f"[dim]Model: {self.MODEL_NAME}[/dim]")

        try:
            # Load tokenizer (always on CPU)
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

            # Load model with appropriate precision
            if self.device == "cuda":
                # Use bfloat16 on GPU to save memory
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.MODEL_NAME,
                    torch_dtype=torch.bfloat16,
                    device_map={"": self.gpu_device},  # Load all on specified GPU
                )
            else:
                # Use full precision on CPU
                self.model = AutoModelForCausalLM.from_pretrained(self.MODEL_NAME)
                self.model = self.model.to(self.device)

            assert self.model is not None
            self.model.eval()  # Set to evaluation mode
            console.print("[green]✓[/green] Translation model loaded successfully")

        except (RuntimeError, torch.cuda.OutOfMemoryError, ValueError, OSError, ImportError) as e:
            error_msg = str(e).lower()
            # Check for GPU/CUDA related errors
            if any(
                keyword in error_msg
                for keyword in ["cuda", "gpu", "cudnn", "cublas", "out of memory"]
            ):
                if self.device == "cuda":
                    console.print(f"[yellow]⚠ GPU initialization failed:[/yellow] {e!s}")
                    console.print("[yellow]Falling back to CPU mode for translation...[/yellow]")

                    # Clean up any partially loaded GPU resources before CPU fallback
                    if hasattr(self, "model") and self.model is not None:
                        del self.model
                        self.model = None
                    if hasattr(self, "tokenizer") and self.tokenizer is not None:
                        del self.tokenizer
                        self.tokenizer = None

                    # Clear GPU cache to free memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Retry with CPU
                    self.device = "cpu"
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
                        self.model = AutoModelForCausalLM.from_pretrained(self.MODEL_NAME)
                        self.model = self.model.to(self.device)
                        self.model.eval()
                        console.print(
                            "[green]✓[/green] Successfully loaded translation model on CPU"
                        )
                        return
                    except (RuntimeError, OSError, ValueError, ImportError) as cpu_error:
                        gpu_cpu_error_msg = (
                            f"Both GPU and CPU initialization failed. GPU: {e!s}, "
                            f"CPU: {cpu_error!s}"
                        )
                        raise GPUError(gpu_cpu_error_msg) from cpu_error
                else:
                    gpu_error_msg = f"GPU initialization failed: {e!s}"
                    raise GPUError(gpu_error_msg) from e
            else:
                error_msg = f"Failed to load translation model: {e!s}"
                raise TranslationError(error_msg) from e

    def translate_text(self, text: str, max_retries: int = 2) -> str:
        """Translate a single text from Hebrew to English with retry logic.

        Args:
            text: Hebrew text to translate
            max_retries: Maximum number of retry attempts (default: 2)

        Returns:
            English translation
        """
        if not text.strip():
            return ""

        self._load_model()
        assert self.model is not None
        assert self.tokenizer is not None

        # Construct instruction prompt for translation
        prompt = (
            f"<s>[INST] Translate the following sentence to English and ONLY give the "
            f"translation:\n{text.strip()} [/INST]"
        )

        for attempt in range(max_retries + 1):
            try:
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt")
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate translation
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,  # Sufficient for subtitle lines
                        do_sample=False,  # Deterministic output
                        temperature=1.0,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                # Decode output
                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extract just the translation (remove prompt echo if present)
                if "[/INST]" in decoded:
                    translation = decoded.split("[/INST]", 1)[1].strip()
                else:
                    translation = decoded.strip()

                return str(translation)

            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if attempt < max_retries:
                    # Clear GPU cache and retry
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    console.print(
                        f"[yellow]Translation attempt {attempt + 1} failed, retrying...[/yellow]"
                    )
                    # Small delay before retry
                    import time

                    time.sleep(0.5)
                else:
                    # Final attempt failed
                    console.print(
                        f"[red]⚠ Translation failed after {max_retries + 1} attempts "
                        f"for text: '{text[:50]}...'[/red]"
                    )
                    console.print(f"[red]Error: {e!s}[/red]")
                    console.print(
                        "[yellow]⚠ WARNING: Returning original Hebrew text "
                        "instead of English translation[/yellow]"
                    )
                    # Return original text as fallback with clear indication this is not translated
                    return f"[UNTRANSLATED] {text}"

        # Should never reach here, but add return for type checker
        return f"[UNTRANSLATED] {text}"

    def translate_segments(self, segments: list[Any]) -> list[dict[str, Any]]:
        """Translate a list of transcription segments from Hebrew to English.

        Args:
            segments: List of segments with 'start', 'end', and 'text' attributes

        Returns:
            List of translated segments as dictionaries
        """
        self._load_model()

        translated_segments = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Translating to English...", total=len(segments))

            for segment in segments:
                # Translate the text
                translated_text = self.translate_text(segment.text)

                # Create translated segment
                translated_segment = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": translated_text,
                }
                translated_segments.append(translated_segment)

                progress.update(task, advance=1)

        return translated_segments

    def cleanup(self) -> None:
        """Clean up model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
