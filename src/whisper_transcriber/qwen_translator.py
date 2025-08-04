"""Qwen-MT online translation module for Hebrew to English translation.

This module provides integration with Alibaba's Qwen-MT-turbo model
through the Dashscope API for high-quality translation with context.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI, OpenAIError, RateLimitError
from rich.console import Console

from .utils import TranslationError

console = Console()

# Load environment variables from .env file
load_dotenv()


class QwenMTTranslator:
    """Translator using Qwen-MT-turbo online model via Dashscope API."""

    def __init__(self, batch_size: int | None = None) -> None:
        """Initialize the Qwen-MT translator with API credentials.

        Args:
            batch_size: Optional batch size override (1-50). If None, uses env var or default.
        """
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            msg = (
                "DASHSCOPE_API_KEY not found in environment variables.\n"
                "Please set it in .env file or export it as environment variable.\n"
                "Get your API key from: https://dashscope.console.aliyun.com/"
            )
            raise TranslationError(msg)

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        self.model = "qwen-mt-turbo"

        # Configurable batch size for translation
        if batch_size is not None:
            # Use provided batch size (already validated by CLI)
            self.batch_size = max(1, min(50, batch_size))
        else:
            # Fall back to environment variable or default
            batch_size_str = os.getenv("QWEN_TRANSLATION_BATCH_SIZE", "10")
            try:
                self.batch_size = max(1, min(50, int(batch_size_str)))
            except ValueError:
                self.batch_size = 10
                console.print(
                    f"[yellow]⚠ Invalid QWEN_TRANSLATION_BATCH_SIZE '{batch_size_str}', "
                    f"using default: 10[/yellow]"
                )

    def translate_full_transcript(
        self,
        transcript_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Translate full transcript from Hebrew to English with context.

        Args:
            transcript_text: Full Hebrew transcript text (may include <new_line> tags)
            metadata: Optional metadata containing title, playlist, description

        Returns:
            Translated English text or None on failure
        """
        if not transcript_text.strip():
            return ""

        try:
            # Build context from metadata
            context_parts = []

            if metadata:
                # Add video title/filename
                if title := metadata.get("title"):
                    context_parts.append(f"Video Title: {title}")

                # Add playlist name if available
                if (playlist := metadata.get("playlist_title")) or (
                    playlist := metadata.get("playlist")
                ):
                    context_parts.append(f"Playlist: {playlist}")

                # Add description excerpt if available
                if description := metadata.get("description"):
                    # Take first 200 characters of description
                    desc_excerpt = description[:200].strip()
                    if desc_excerpt:
                        context_parts.append(f"Description excerpt: {desc_excerpt}")

            # Build the full prompt - simple for Qwen-MT
            prompt_parts = ["Translate the following Hebrew transcript to English."]

            # Add instructions for preserving tags
            if "<new_line>" in transcript_text:
                prompt_parts.append(
                    "\nIMPORTANT: Preserve the <new_line> markers exactly as they appear "
                    "in the original text. Do not add, remove, or reorder any <new_line> markers. "
                    "Treat each line or block delimited by these markers as independent text and "
                    "maintain their structure and line boundaries in the output."
                )

            prompt_parts.append(f"\n{transcript_text}")
            prompt = "\n".join(prompt_parts)

            # Prepare translation options with context in domains
            domain_parts = ["video_content"]
            if context_parts:
                context_summary = " | ".join(context_parts)
                domain_parts.append(f"Context: {context_summary[:200]}")

            domain_context = " | ".join(domain_parts)

            translation_options = {
                "source_lang": "Hebrew",
                "target_lang": "English",
                "domains": domain_context,
            }

            # Make API call with retry
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    console.print("[cyan]Sending transcript to Qwen-MT for translation...[/cyan]")

                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        extra_body={"translation_options": translation_options},
                        timeout=60,  # Longer timeout for batch
                    )

                    translation: str | None = completion.choices[0].message.content

                    if translation:
                        console.print("[green]✓ Translation completed successfully[/green]")
                        return translation
                    console.print("[yellow]⚠ Empty translation received[/yellow]")
                    return None

                except (APIConnectionError, RateLimitError, APIStatusError, OpenAIError) as e:
                    if attempt < max_retries:
                        console.print(
                            f"[yellow]Translation attempt {attempt + 1} failed, "
                            f"retrying...[/yellow]"
                        )
                        time.sleep(2**attempt)  # Exponential backoff
                    else:
                        console.print(f"[red]✗ Translation failed: {e!s}[/red]")
                        return None

        except (APIConnectionError, RateLimitError, APIStatusError, OpenAIError) as e:
            console.print(f"[red]✗ Translation error: {e!s}[/red]")
            return None

        return None

    def translate_transcript_json(self, transcript_obj: dict[str, Any]) -> dict[str, Any] | None:
        """Translate a transcript JSON object using batch processing.

        Args:
            transcript_obj: Transcript object with segments

        Returns:
            Updated transcript object with translations or None on failure
        """
        segments = transcript_obj.get("segments", [])
        if not segments:
            return None

        console.print(
            f"[cyan]Translating {len(segments)} segments in batches of {self.batch_size}...[/cyan]"
        )

        # Create a deep copy to avoid mutating the original
        import copy

        result = transcript_obj.copy()
        # Deep copy the segments list to avoid mutating original segment dictionaries
        segments_copy = copy.deepcopy(segments)
        result["segments"] = segments_copy
        all_translated_texts = []

        # Extract metadata for context
        metadata = transcript_obj.get("metadata", {})

        # Process segments in batches
        for batch_start in range(0, len(segments_copy), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(segments_copy))
            batch_segments = segments_copy[batch_start:batch_end]
            batch_num = (batch_start // self.batch_size) + 1
            total_batches = (len(segments_copy) + self.batch_size - 1) // self.batch_size

            console.print(
                f"[cyan]Processing batch {batch_num}/{total_batches} "
                f"(segments {batch_start + 1}-{batch_end})...[/cyan]"
            )

            # Collect texts from batch
            batch_texts = []
            for segment in batch_segments:
                text = segment.get("text", "").strip()
                # Keep empty segments to maintain alignment
                batch_texts.append(text if text else "")

            # Skip batch if all segments are empty
            if not any(batch_texts):
                for segment in batch_segments:
                    segment["translated_text"] = ""
                    all_translated_texts.append("")
                continue

            # Create batch translation prompt with context
            translated_batch = self._translate_batch(
                batch_texts,
                metadata,
                batch_num,
                total_batches,
                # Pass context from previous batch for continuity
                previous_context=all_translated_texts[-1] if all_translated_texts else None,
            )

            if translated_batch is None:
                # Fallback: mark batch as untranslated
                for segment in batch_segments:
                    segment["translated_text"] = "[UNTRANSLATED]"
                    all_translated_texts.append("[UNTRANSLATED]")
            else:
                # Validate alignment before mapping
                if len(translated_batch) != len(batch_segments):
                    console.print(
                        f"[yellow]⚠ Alignment mismatch in batch {batch_num}: "
                        f"expected {len(batch_segments)}, got {len(translated_batch)}[/yellow]"
                    )
                    # Align the translation batch to match segment count
                    if len(translated_batch) < len(batch_segments):
                        # Pad with placeholders if we have fewer translations
                        translated_batch.extend(
                            ["[UNTRANSLATED]"] * (len(batch_segments) - len(translated_batch))
                        )
                    else:
                        # Truncate if we have more translations than segments
                        translated_batch = translated_batch[: len(batch_segments)]

                # Map translations back to segments (now guaranteed to be same length)
                for segment, translated_text in zip(batch_segments, translated_batch, strict=True):
                    segment["translated_text"] = translated_text
                    all_translated_texts.append(translated_text)

        # Add translated full text (excluding error markers)
        result["translated_full_text"] = "\n".join(
            line
            for line in all_translated_texts
            if line and line not in ["[UNTRANSLATED]", "[TRANSLATION_ERROR]"]
        )

        console.print(f"[green]✓ Completed translation of {len(segments_copy)} segments[/green]")
        return result

    def _translate_batch(
        self,
        batch_texts: list[str],
        metadata: dict[str, Any],
        batch_num: int,
        total_batches: int,
        previous_context: str | None = None,
        is_sub_batch: bool = False,
    ) -> list[str] | None:
        """Translate a batch of texts with context.

        Args:
            batch_texts: List of texts to translate (may include empty strings)
            metadata: Video metadata for context
            batch_num: Current batch number
            total_batches: Total number of batches
            previous_context: Last line from previous batch for continuity
            is_sub_batch: Whether this is a sub-batch from recursive splitting

        Returns:
            List of translated texts maintaining 1:1 correspondence, or None on failure
        """
        # Prepare the batch content
        batch_content = self._prepare_batch_content(batch_texts)

        # Build the translation prompt
        prompt = self._build_translation_prompt(batch_texts, batch_content)

        # Prepare translation options
        translation_options = self._build_translation_options(metadata, batch_num, total_batches)

        # Attempt translation with retry logic
        return self._attempt_translation_with_retry(
            prompt,
            translation_options,
            batch_texts,
            batch_num,
            metadata,
            previous_context,
            is_sub_batch,
        )

    def _prepare_batch_content(self, batch_texts: list[str]) -> str:
        """Prepare batch content with numbered lines format."""
        batch_lines = []
        for i, text in enumerate(batch_texts, 1):
            batch_lines.append(f"{i}. {text}")
        return "\n".join(batch_lines)

    def _build_translation_prompt(self, batch_texts: list[str], batch_content: str) -> str:
        """Build the translation prompt for batch processing."""
        prompt_parts = [
            f"Translate these {len(batch_texts)} Hebrew subtitle lines to English.",
            "Each line is numbered. Translate the text after each number.",
            f"Output EXACTLY {len(batch_texts)} lines in the same format: "
            f"'1. [translation]', '2. [translation]', etc.",
            "Preserve empty lines as empty (e.g., '3. ') and maintain the exact line count "
            "and numbering.",
            "",
            f"{batch_content}",
        ]
        return "\n".join(prompt_parts)

    def _build_translation_options(
        self, metadata: dict[str, Any], batch_num: int, total_batches: int
    ) -> dict[str, Any]:
        """Build translation options with context information."""
        # Build context information
        context_parts = []

        # Add video metadata context
        if title := metadata.get("title"):
            context_parts.append(f"Video Title: {title}")

        if playlist := (metadata.get("playlist_title") or metadata.get("playlist")):
            context_parts.append(f"Playlist: {playlist}")

        if description := metadata.get("description"):
            desc_excerpt = description[:200].strip()
            if desc_excerpt:
                context_parts.append(f"Description excerpt: {desc_excerpt}")

        # Prepare translation options with rich domain context
        domain_parts = ["video_subtitles"]

        if context_parts:
            # Add context information to domains
            context_summary = " | ".join(context_parts)
            domain_parts.append(f"Context: {context_summary[:150]}")  # Limit context length

        # Add batch info to domains
        if isinstance(total_batches, int) and total_batches > 0:
            domain_parts.append(
                f"These lines are transcripts from batch {batch_num} of {total_batches}"
            )
        else:
            domain_parts.append(
                f"These lines are transcripts from a sub-batch of batch {batch_num}"
            )
        domain_parts.append(
            "Translate each numbered line and maintain the exact same numbering format"
        )

        domain_context = " | ".join(domain_parts)

        return {
            "source_lang": "Hebrew",
            "target_lang": "English",
            "domains": domain_context,
        }

    def _attempt_translation_with_retry(
        self,
        prompt: str,
        translation_options: dict[str, Any],
        batch_texts: list[str],
        batch_num: int,
        metadata: dict[str, Any],
        previous_context: str | None,
        is_sub_batch: bool = False,
    ) -> list[str] | None:
        """Attempt translation with retry logic and error handling."""
        max_retries = 2
        base_delay = 1

        for attempt in range(max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={"translation_options": translation_options},
                    timeout=60,  # Longer timeout for batch
                )

                response = completion.choices[0].message.content
                if not response:
                    batch_label = f"{batch_num} (sub-batch)" if is_sub_batch else str(batch_num)
                    console.print(f"[yellow]⚠ Empty response for batch {batch_label}[/yellow]")
                    return None

                # Parse and validate the response
                batch_label = f"{batch_num} (sub-batch)" if is_sub_batch else str(batch_num)
                return self._parse_and_validate_response(response, batch_texts, batch_label)

            except (APIConnectionError, RateLimitError, APIStatusError, OpenAIError) as e:
                batch_label = f"{batch_num} (sub-batch)" if is_sub_batch else str(batch_num)
                if self._handle_retry_error(e, attempt, max_retries, base_delay, batch_label):
                    continue

                # Try fallback for rate limits (only for non-sub-batches)
                if not is_sub_batch and isinstance(e, RateLimitError):
                    return self._try_smaller_batches(
                        batch_texts, metadata, batch_num, previous_context
                    )

                console.print(f"[red]✗ Translation failed for batch {batch_label}: {e!s}[/red]")
                return None

        return None

    def _parse_and_validate_response(
        self, response: str, batch_texts: list[str], batch_num: int | str
    ) -> list[str]:
        """Parse the response and validate line count."""
        import re

        response_lines = response.strip().split("\n")
        translated_lines = []

        # Extract translations from numbered format (supports "1.", "1)", "1-" styles)
        line_pattern = re.compile(r"^\s*(\d+)[\.\)\-]\s*(.*?)$")

        for response_line in response_lines:
            match = line_pattern.match(response_line.strip())
            if match:
                translation = match.group(2).strip()
                translated_lines.append(translation)
            elif response_line.strip():  # Non-empty line that doesn't match pattern
                # Try to extract text without number prefix
                cleaned = response_line.strip()
                # Remove common prefixes
                prefixes_to_remove = ["Translation:", "English:", "In English:"]
                for prefix in prefixes_to_remove:
                    if cleaned.lower().startswith(prefix.lower()):
                        cleaned = cleaned[len(prefix) :].strip()
                        break
                translated_lines.append(cleaned)

        # Validate and align line count
        if len(translated_lines) != len(batch_texts):
            console.print(
                f"[yellow]⚠ Batch {batch_num}: Expected {len(batch_texts)} lines, "
                f"got {len(translated_lines)}. Attempting to align...[/yellow]"
            )
            translated_lines = self._align_translation_lines(translated_lines, batch_texts)

        # Final cleanup
        cleaned_lines = [line.strip() if line else "" for line in translated_lines]

        console.print(f"[green]✓ Batch {batch_num} translated successfully[/green]")
        return cleaned_lines

    def _handle_retry_error(
        self,
        error: Exception,
        attempt: int,
        max_retries: int,
        base_delay: int,
        batch_num: int | str,
    ) -> bool:
        """Handle retry logic for translation errors. Returns True if should retry."""
        # Check for rate limit
        if isinstance(error, RateLimitError):
            if attempt < max_retries:
                delay = base_delay * (2**attempt) + (attempt * 0.5)
                console.print(
                    f"[yellow]⚠ Rate limit for batch {batch_num}, "
                    f"waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}..."
                    f"[/yellow]"
                )
                time.sleep(delay)
                return True
            console.print(f"[red]✗ Rate limit exceeded for batch {batch_num}[/red]")

        return False

    def _align_translation_lines(
        self, translated_lines: list[str], batch_texts: list[str]
    ) -> list[str]:
        """Align translation lines to match expected batch size."""
        if len(translated_lines) < len(batch_texts):
            # Pad with empty strings
            translated_lines.extend([""] * (len(batch_texts) - len(translated_lines)))
        else:
            # Truncate excess
            translated_lines = translated_lines[: len(batch_texts)]
        return translated_lines

    def _try_smaller_batches(
        self,
        batch_texts: list[str],
        metadata: dict[str, Any],
        batch_num: int,
        previous_context: str | None,
    ) -> list[str] | None:
        """Try with smaller batch size as fallback for rate limits."""
        if len(batch_texts) <= 2:
            return None

        console.print("[yellow]Retrying with smaller sub-batches...[/yellow]")
        mid = len(batch_texts) // 2

        first_half = self._translate_batch(
            batch_texts[:mid],
            metadata,
            batch_num,
            -1,  # Use -1 to indicate sub-batch
            previous_context,
            is_sub_batch=True,
        )

        if not first_half:
            return None

        second_half = self._translate_batch(
            batch_texts[mid:],
            metadata,
            batch_num,
            -1,  # Use -1 to indicate sub-batch
            first_half[-1],
            is_sub_batch=True,
        )

        if second_half:
            return first_half + second_half
        # Keep first_half and pad the missing lines for best-effort output
        console.print(
            "[yellow]⚠ Second sub-batch failed, using placeholders for remaining segments[/yellow]"
        )
        return first_half + ["[UNTRANSLATED]"] * (len(batch_texts) - len(first_half))

    def translate_segments(self, segments: list[Any]) -> list[dict[str, Any]]:
        """Translate segments using batch approach for guaranteed alignment.

        This method maintains interface compatibility with DictaLMTranslator.

        Args:
            segments: List of segments with 'start', 'end', and 'text' attributes

        Returns:
            List of translated segments with guaranteed 1:1 mapping
        """
        if not segments:
            return []

        # Extract texts and translate using batch path to preserve alignment
        batch_texts = [getattr(seg, "text", "") or "" for seg in segments]
        translated_batch = self._translate_batch(
            batch_texts, metadata={}, batch_num=1, total_batches=1, previous_context=None
        )

        if not translated_batch:
            # Return empty segments instead of None to match interface
            return [
                {"start": segment.start, "end": segment.end, "text": "[UNTRANSLATED]"}
                for segment in segments
            ]

        translated_segments = []
        for i, segment in enumerate(segments):
            translated_text_segment = (
                translated_batch[i] if i < len(translated_batch) else "[UNTRANSLATED]"
            )
            translated_segments.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": translated_text_segment,
                }
            )

        return translated_segments

    def cleanup(self) -> None:
        """Clean up resources used by the online translator."""
        # Close HTTP connections if available
        if hasattr(self, "client") and self.client is not None:
            if hasattr(self.client, "close"):
                # Cleanup errors are expected and can be safely ignored
                # The connection may already be closed or in an invalid state
                with contextlib.suppress(OSError, RuntimeError, OpenAIError):
                    self.client.close()

            # Clear API client reference
            self.client = None
