#!/usr/bin/env bash
set -euo pipefail

# Test script for whisper-transcriber

# Test with a short public domain video (Big Buck Bunny trailer)
# This is a 30-second trailer that should work well for testing
VIDEO_URL="https://www.youtube.com/watch?v=YE7VzlLtp-4"
OUTPUT_FILE="test_output.txt"

echo "Testing whisper-transcriber with Hebrew-optimized model..."
echo "URL: $VIDEO_URL"
echo "Output: $OUTPUT_FILE"
echo ""

# Run the transcriber with Hebrew model, auto device detection, and verbose output
python -m whisper_transcriber -m ivrit-turbo --device auto -v "$VIDEO_URL" "$OUTPUT_FILE"

# Check if output file was created
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "Success! Transcript saved to $OUTPUT_FILE"
    echo "First 500 characters of transcript:"
    echo "---"
    head -c 500 "$OUTPUT_FILE"
    echo ""
    echo "---"
    exit 0
else
    echo "Error: Output file not created"
    exit 1
fi
