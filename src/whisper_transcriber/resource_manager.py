"""Resource management utilities for proper cleanup and memory tracking.

This module provides context managers and utilities for managing resources
like GPU memory, temporary files, and model instances.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import time
from collections.abc import Generator
from typing import Any

import torch
from rich.console import Console

console = Console()


@contextlib.contextmanager
def managed_temp_directory(prefix: str = "whisper_") -> Generator[str, None, None]:
    """Context manager for temporary directory creation and cleanup.

    Args:
        prefix: Prefix for the temporary directory name

    Yields:
        Path to the temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield temp_dir
    finally:
        # Ensure cleanup even if errors occur
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@contextlib.contextmanager
def managed_gpu_memory() -> Generator[None, None, None]:
    """Context manager for GPU memory management.

    Clears GPU cache before and after operations to minimize fragmentation.
    """
    initial_memory = 0

    if torch.cuda.is_available():
        # Clear cache before operation
        torch.cuda.empty_cache()
        # Record initial memory
        initial_memory = torch.cuda.memory_allocated()

    try:
        yield
    finally:
        if torch.cuda.is_available():
            # Clear cache after operation
            torch.cuda.empty_cache()

            # Report memory usage if significant
            final_memory = torch.cuda.memory_allocated()
            memory_diff = final_memory - initial_memory
            if memory_diff > 100 * 1024 * 1024:  # More than 100MB
                console.print(
                    f"[dim]GPU memory increased by {memory_diff / 1024 / 1024:.1f} MB[/dim]"
                )


@contextlib.contextmanager
def timed_operation(
    operation_name: str, verbose: bool = False
) -> Generator[dict[str, Any], None, None]:
    """Context manager for timing operations.

    Args:
        operation_name: Name of the operation being timed
        verbose: Whether to print timing information

    Yields:
        Dictionary to store operation metadata
    """
    start_time = time.time()
    metadata = {"name": operation_name, "start_time": start_time}

    try:
        yield metadata
    finally:
        elapsed_time = time.time() - start_time
        metadata["elapsed_time"] = elapsed_time

        if verbose:
            console.print(f"[dim]{operation_name} completed in {elapsed_time:.1f} seconds[/dim]")


class ResourceTracker:
    """Track and manage system resources during processing.

    Provides utilities for monitoring memory usage, cleaning up resources,
    and ensuring proper resource management throughout the application.
    """

    def __init__(self, verbose: bool = False) -> None:
        """Initialize the resource tracker.

        Args:
            verbose: Whether to print resource usage information
        """
        self.verbose = verbose
        self.temp_dirs: list[str] = []
        self.start_time = time.time()
        self.operations: list[dict[str, Any]] = []

    def add_temp_dir(self, temp_dir: str) -> None:
        """Register a temporary directory for cleanup.

        Args:
            temp_dir: Path to temporary directory
        """
        self.temp_dirs.append(temp_dir)

    @contextlib.contextmanager
    def track_operation(self, operation_name: str) -> Generator[None, None, None]:
        """Track a named operation with timing and resource usage.

        Args:
            operation_name: Name of the operation

        Yields:
            None
        """
        with timed_operation(operation_name, self.verbose) as metadata:
            self.operations.append(metadata)

            if torch.cuda.is_available():
                initial_gpu_memory = torch.cuda.memory_allocated()

            try:
                yield
            finally:
                if torch.cuda.is_available():
                    final_gpu_memory = torch.cuda.memory_allocated()
                    metadata["gpu_memory_change"] = final_gpu_memory - initial_gpu_memory

    def cleanup(self) -> None:
        """Clean up all tracked resources."""
        # Clean up temporary directories
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    if self.verbose:
                        console.print(f"[dim]Cleaned up temp directory: {temp_dir}[/dim]")
                except (OSError, PermissionError) as e:
                    console.print(f"[yellow]Warning: Failed to clean up {temp_dir}: {e}[/yellow]")

        self.temp_dirs.clear()

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_summary(self) -> dict[str, Any]:
        """Get summary of resource usage.

        Returns:
            Dictionary with resource usage statistics
        """
        total_time = time.time() - self.start_time

        summary = {
            "total_time": total_time,
            "operations_count": len(self.operations),
            "temp_dirs_created": len(self.temp_dirs),
        }

        if torch.cuda.is_available():
            summary["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            summary["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024

        # Calculate operation statistics
        if self.operations:
            operation_times = [op.get("elapsed_time", 0) for op in self.operations]
            summary["avg_operation_time"] = sum(operation_times) / len(operation_times)
            summary["max_operation_time"] = max(operation_times)

            # GPU memory stats if available
            gpu_changes = [
                op.get("gpu_memory_change", 0)
                for op in self.operations
                if "gpu_memory_change" in op
            ]
            if gpu_changes:
                summary["max_gpu_memory_increase_mb"] = max(gpu_changes) / 1024 / 1024

        return summary

    def print_summary(self) -> None:
        """Print a formatted summary of resource usage."""
        summary = self.get_summary()

        console.print("\n[bold]Resource Usage Summary[/bold]")
        console.print(f"Total time: {summary['total_time']:.1f} seconds")
        console.print(f"Operations completed: {summary['operations_count']}")

        if "avg_operation_time" in summary:
            console.print(f"Average operation time: {summary['avg_operation_time']:.1f} seconds")
            console.print(f"Max operation time: {summary['max_operation_time']:.1f} seconds")

        if torch.cuda.is_available():
            console.print(f"GPU memory allocated: {summary['gpu_memory_allocated_mb']:.1f} MB")
            console.print(f"GPU memory reserved: {summary['gpu_memory_reserved_mb']:.1f} MB")

            if "max_gpu_memory_increase_mb" in summary:
                console.print(
                    f"Max GPU memory increase: {summary['max_gpu_memory_increase_mb']:.1f} MB"
                )
