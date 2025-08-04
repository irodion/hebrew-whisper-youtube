"""Model manager for persistent model instances.

This module implements a singleton pattern to manage Whisper and DictaLM models,
ensuring they persist across multiple transcription tasks to avoid repeated loading overhead.
"""

from __future__ import annotations

import gc
import threading
from typing import Any

import torch
from rich.console import Console

from .qwen_translator import QwenMTTranslator
from .transcriber import WhisperTranscriber
from .translator import DictaLMTranslator
from .utils import GPUError, TranslationError

console = Console()


class ModelManager:
    """Singleton manager for ML model instances.

    Ensures models are loaded once and reused across multiple operations,
    significantly reducing initialization overhead for batch processing.
    """

    _instance: ModelManager | None = None
    _lock = threading.Lock()

    def __new__(cls) -> ModelManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the model manager."""
        # Only initialize once
        if hasattr(self, "_initialized"):
            return

        with self._lock:
            # Double-check after acquiring lock
            if hasattr(self, "_initialized"):
                return

            self._transcribers: dict[str, WhisperTranscriber] = {}
            self._translators: dict[str, DictaLMTranslator] = {}
            self._default_device: str | None = None
            self._initialized = True
            console.print("[green]✓[/green] Model manager initialized")

    def get_transcriber(
        self,
        model_size: str = "ivrit-turbo",
        device: str | None = None,
        gpu_device: int = 0,
        force_reload: bool = False,
    ) -> WhisperTranscriber:
        """Get or create a WhisperTranscriber instance.

        Args:
            model_size: Whisper model size
            device: Device to use ('cuda', 'cpu', or None for auto)
            gpu_device: GPU device ID
            force_reload: Force reload of the model

        Returns:
            WhisperTranscriber instance with loaded model
        """
        # Create cache key with more specificity to prevent collisions
        # Include compute type for differentiation between different device configs
        compute_type = (
            "float16"
            if (device == "cuda" or (device is None and torch.cuda.is_available()))
            else "int8"
        )
        cache_key = f"{model_size}_{device or 'auto'}_{gpu_device}_{compute_type}"

        # Check if we need to create new instance
        if force_reload and cache_key in self._transcribers:
            old_transcriber = self._transcribers.pop(cache_key)
            # Clean up old model properly
            old_transcriber.cleanup()
            del old_transcriber

            # Force garbage collection to ensure memory is freed
            gc.collect()

            if device == "cuda" or (device is None and torch.cuda.is_available()):
                torch.cuda.empty_cache()

        # Return existing instance if available
        if cache_key in self._transcribers:
            console.print(f"[dim]Reusing loaded Whisper model: {model_size}[/dim]")
            return self._transcribers[cache_key]

        # Create new instance
        console.print(f"[cyan]Loading new Whisper model: {model_size}[/cyan]")
        try:
            transcriber = WhisperTranscriber(
                model_size=model_size, device=device, gpu_device=gpu_device
            )
            # Pre-load the model
            transcriber._load_model()
            self._transcribers[cache_key] = transcriber
            return transcriber
        except (GPUError, RuntimeError, OSError, ValueError, ImportError) as e:
            console.print(f"[red]✗ Failed to load Whisper model:[/red] {e}")
            raise

    def get_translator(
        self,
        device: str | None = None,
        gpu_device: int = 0,
        force_reload: bool = False,
        translator_type: str = "local",
        translation_batch_size: int | None = None,
    ) -> DictaLMTranslator | QwenMTTranslator:
        """Get or create a translator instance based on type.

        Args:
            device: Device to use ('cuda', 'cpu', or None for auto)
            gpu_device: GPU device ID
            force_reload: Force reload of the model
            translator_type: Type of translator ('local' for DictaLM, 'qwen' for Qwen-MT)
            translation_batch_size: Optional batch size for Qwen translator

        Returns:
            Translator instance (DictaLMTranslator or QwenMTTranslator)
        """
        # Handle Qwen translator (online, no caching needed)
        if translator_type == "qwen":
            console.print("[cyan]Creating Qwen-MT translator[/cyan]")
            try:
                return QwenMTTranslator(batch_size=translation_batch_size)
            except TranslationError as e:
                console.print(f"[red]✗ Failed to create Qwen-MT translator:[/red] {e}")
                raise

        # Handle local DictaLM translator (cache for efficiency)
        torch_dtype = (
            "bfloat16"
            if (device == "cuda" or (device is None and torch.cuda.is_available()))
            else "float32"
        )
        cache_key = f"dictalm_{device or 'auto'}_{gpu_device}_{torch_dtype}"

        # Check if we need to create new instance
        if force_reload and cache_key in self._translators:
            old_translator = self._translators.pop(cache_key)
            old_translator.cleanup()
            del old_translator

        # Return existing instance if available
        if cache_key in self._translators:
            console.print("[dim]Reusing loaded DictaLM model[/dim]")
            return self._translators[cache_key]

        # Create new instance
        console.print("[cyan]Loading new DictaLM model[/cyan]")
        try:
            translator = DictaLMTranslator(device=device, gpu_device=gpu_device)
            # Pre-load the model
            translator._load_model()
            self._translators[cache_key] = translator
            return translator
        except (GPUError, TranslationError, RuntimeError, OSError, ValueError, ImportError) as e:
            console.print(f"[red]✗ Failed to load DictaLM model:[/red] {e}")
            raise

    def cleanup_transcriber(
        self, model_size: str, device: str | None = None, gpu_device: int = 0
    ) -> None:
        """Clean up a specific transcriber instance.

        Args:
            model_size: Model size to clean up
            device: Device specification
            gpu_device: GPU device ID
        """
        # Reconstruct cache key with same logic as get_transcriber
        compute_type = (
            "float16"
            if (device == "cuda" or (device is None and torch.cuda.is_available()))
            else "int8"
        )
        cache_key = f"{model_size}_{device or 'auto'}_{gpu_device}_{compute_type}"
        if cache_key in self._transcribers:
            transcriber = self._transcribers.pop(cache_key)
            transcriber.cleanup()
            del transcriber
            console.print(f"[dim]Cleaned up Whisper model: {model_size}[/dim]")

    def cleanup_translator(self, device: str | None = None, gpu_device: int = 0) -> None:
        """Clean up a specific translator instance.

        Args:
            device: Device specification
            gpu_device: GPU device ID
        """
        # Reconstruct cache key with same logic as get_translator
        torch_dtype = (
            "bfloat16"
            if (device == "cuda" or (device is None and torch.cuda.is_available()))
            else "float32"
        )
        cache_key = f"dictalm_{device or 'auto'}_{gpu_device}_{torch_dtype}"
        if cache_key in self._translators:
            translator = self._translators.pop(cache_key)
            translator.cleanup()
            del translator
            console.print("[dim]Cleaned up DictaLM model[/dim]")

    def cleanup_all(self) -> None:
        """Clean up all loaded models and free memory."""
        console.print("[yellow]Cleaning up all models...[/yellow]")

        # Clean up transcribers
        for transcriber in self._transcribers.values():
            transcriber.cleanup()
        self._transcribers.clear()

        # Clean up translators
        for translator in self._translators.values():
            translator.cleanup()
        self._translators.clear()

        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        console.print("[green]✓[/green] All models cleaned up")

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory usage statistics for loaded models.

        Returns:
            Dictionary with memory usage information
        """
        stats = {
            "loaded_transcribers": len(self._transcribers),
            "loaded_translators": len(self._translators),
            "transcriber_models": list(self._transcribers.keys()),
            "translator_models": list(self._translators.keys()),
        }

        if torch.cuda.is_available():
            stats["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            stats["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024

        return stats

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        if cls._instance is not None:
            cls._instance.cleanup_all()
        cls._instance = None


# Global instance getter
def get_model_manager() -> ModelManager:
    """Get the global ModelManager instance.

    Returns:
        ModelManager singleton instance
    """
    return ModelManager()
