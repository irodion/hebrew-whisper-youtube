"""Video metadata formatting utilities.

Copyright (c) 2025 Whisper Transcriber Contributors
Licensed under the MIT License - see LICENSE file for details.
"""

from datetime import UTC, datetime
from typing import Any

from .utils import format_time


class MetadataFormatter:
    """Format video metadata for different output types."""

    @staticmethod
    def format_text_header(metadata: dict[str, Any]) -> str:
        """Format metadata as a text header.

        Args:
            metadata: Video metadata dictionary

        Returns:
            Formatted text header string
        """
        lines = [
            "=" * 80,
            "VIDEO INFORMATION",
            "=" * 80,
            f"Title: {metadata.get('title', 'Unknown')}",
            f"Channel: {metadata.get('channel', metadata.get('uploader', 'Unknown'))}",
        ]

        # Add duration if available
        duration = metadata.get("duration", 0)
        if duration:
            lines.append(f"Duration: {format_time(duration)}")

        # Add upload date if available
        upload_date = metadata.get("upload_date", "")
        if upload_date:
            try:
                date_obj = datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=UTC)
                formatted_date = date_obj.strftime("%B %d, %Y")
                lines.append(f"Upload Date: {formatted_date}")
            except ValueError:
                lines.append(f"Upload Date: {upload_date}")

        # Add view count if available
        view_count = metadata.get("view_count", 0)
        if view_count:
            lines.append(f"Views: {view_count:,}")

        # Add URL
        if metadata.get("webpage_url"):
            lines.append(f"URL: {metadata['webpage_url']}")

        # Add description preview if available
        description = metadata.get("description", "").strip()
        if description:
            lines.extend(["", "Description:", "-" * 80])
            # Limit description to first 3 lines or 500 chars
            desc_lines = description.split("\n")[:3]
            desc_preview = "\n".join(desc_lines)
            if len(desc_preview) > 500:
                desc_preview = desc_preview[:497] + "..."
            lines.append(desc_preview)

        lines.extend(["=" * 80, "", "TRANSCRIPT", "-" * 80, ""])

        return "\n".join(lines)

    @staticmethod
    def format_vtt_metadata(metadata: dict[str, Any]) -> str:
        """Format metadata as VTT NOTE comments.

        Args:
            metadata: Video metadata dictionary

        Returns:
            VTT-formatted metadata comments
        """
        lines = ["WEBVTT", ""]

        # Add metadata as NOTE comments
        lines.append("NOTE")
        lines.append(f"Title: {metadata.get('title', 'Unknown')}")
        lines.append(f"Channel: {metadata.get('channel', metadata.get('uploader', 'Unknown'))}")

        duration = metadata.get("duration", 0)
        if duration:
            lines.append(f"Duration: {format_time(duration)}")

        upload_date = metadata.get("upload_date", "")
        if upload_date:
            try:
                date_obj = datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=UTC)
                formatted_date = date_obj.strftime("%B %d, %Y")
                lines.append(f"Upload Date: {formatted_date}")
            except ValueError:
                lines.append(f"Upload Date: {upload_date}")

        if metadata.get("webpage_url"):
            lines.append(f"URL: {metadata['webpage_url']}")

        lines.append("")  # Empty line after metadata

        return "\n".join(lines)

    @staticmethod
    def format_json_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Format metadata for JSON output.

        Args:
            metadata: Video metadata dictionary

        Returns:
            Cleaned metadata dictionary for JSON
        """
        # Create a copy to avoid mutating the input
        metadata_copy = metadata.copy()

        # Format upload date if present
        upload_date = metadata_copy.get("upload_date", "")
        if upload_date:
            try:
                date_obj = datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=UTC)
                metadata_copy["upload_date_formatted"] = date_obj.strftime("%Y-%m-%d")
            except ValueError:
                pass

        # Format duration if present
        duration = metadata_copy.get("duration", 0)
        if duration:
            metadata_copy["duration_formatted"] = format_time(duration)

        return metadata_copy
