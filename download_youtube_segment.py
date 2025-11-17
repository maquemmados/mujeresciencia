#!/usr/bin/env python3
"""
Simple YouTube Video Segment Downloader

Downloads a YouTube video and extracts a specific time segment as a WAV audio file.

Example usage:
    python download_youtube_segment.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --start 0:16 --end 0:32
    python download_youtube_segment.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --start 16 --end 32 --output my_segment.wav
"""

import argparse
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

import yt_dlp


def check_ffmpeg():
    """
    Check if ffmpeg is installed and accessible.

    Returns:
        bool: True if ffmpeg is available, False otherwise
    """
    if shutil.which('ffmpeg') is None:
        return False
    if shutil.which('ffprobe') is None:
        return False
    return True


def parse_time(time_str):
    """
    Parse time string to seconds.

    Supports formats:
    - Seconds: "16" or "16.5"
    - MM:SS: "1:30" or "01:30"
    - HH:MM:SS: "1:05:30"

    Args:
        time_str: Time string to parse

    Returns:
        Time in seconds as float
    """
    if ':' not in time_str:
        # Simple seconds format
        return float(time_str)

    parts = time_str.split(':')
    if len(parts) == 2:
        # MM:SS format
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        # HH:MM:SS format
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        raise ValueError(f"Invalid time format: {time_str}")


def download_youtube_video(url: str, output_path: str) -> str:
    """
    Download YouTube video as audio in native format (no conversion).

    Args:
        url: YouTube video URL
        output_path: Path to save the audio file (without extension)

    Returns:
        Path to the downloaded audio file
    """
    print(f"Downloading YouTube video: {url}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path + '.%(ext)s',  # Keep original extension
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            # Get the actual filename with extension
            if 'requested_downloads' in info and info['requested_downloads']:
                audio_path = info['requested_downloads'][0]['filepath']
            else:
                # Fallback: try common audio extensions
                ext = info.get('ext', 'webm')
                audio_path = f"{output_path}.{ext}"

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Downloaded file not found: {audio_path}")

        print(f"✓ Downloaded: {audio_path}")
        return audio_path
    except Exception as e:
        print(f"Error downloading video: {e}", file=sys.stderr)
        raise


def extract_segment(input_path: str, output_path: str, start_time: float, end_time: float):
    """
    Extract a segment from audio file and convert to WAV using ffmpeg.

    Args:
        input_path: Path to input audio file (any format)
        output_path: Path to output segment file (WAV)
        start_time: Start time in seconds
        end_time: End time in seconds
    """
    duration = end_time - start_time

    if duration <= 0:
        raise ValueError(f"Invalid time range: start={start_time}, end={end_time}")

    print(f"\nExtracting segment from {start_time}s to {end_time}s (duration: {duration}s)")

    # Use ffmpeg to extract the segment and convert to WAV
    # -ss before -i for faster seeking
    cmd = [
        'ffmpeg',
        '-ss', str(start_time),  # Seek to start position (before -i for speed)
        '-i', input_path,
        '-t', str(duration),  # Duration to extract from seek position
        '-acodec', 'pcm_s16le',  # WAV format (16-bit PCM)
        '-ar', '44100',  # Sample rate: 44.1kHz
        '-ac', '2',  # Stereo (or use 1 for mono)
        '-y',  # Overwrite output file
        output_path
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        print(f"✓ Segment saved: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"\nError extracting segment with ffmpeg:", file=sys.stderr)
        print(f"Command: {' '.join(cmd)}", file=sys.stderr)
        print(f"Error output:\n{e.stderr}", file=sys.stderr)
        raise
    except FileNotFoundError:
        print(f"\nError: ffmpeg command not found!", file=sys.stderr)
        print(f"Please make sure ffmpeg is installed and in your PATH", file=sys.stderr)
        raise


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:05.2f}"
    else:
        return f"{minutes}:{secs:05.2f}"


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube video and extract a specific time segment as WAV audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract segment from 0:16 to 0:32
  %(prog)s https://www.youtube.com/watch?v=dQw4w9WgXcQ --start 0:16 --end 0:32

  # Same thing, using seconds
  %(prog)s https://www.youtube.com/watch?v=dQw4w9WgXcQ --start 16 --end 32

  # Specify custom output filename
  %(prog)s https://www.youtube.com/watch?v=dQw4w9WgXcQ --start 16 --end 32 --output my_segment.wav

  # Extract from 1:30 to 2:15
  %(prog)s https://www.youtube.com/watch?v=dQw4w9WgXcQ --start 1:30 --end 2:15
        """
    )

    parser.add_argument(
        'url',
        help='YouTube video URL'
    )
    parser.add_argument(
        '--start',
        required=True,
        help='Start time (formats: seconds "16", MM:SS "0:16", or HH:MM:SS "1:05:30")'
    )
    parser.add_argument(
        '--end',
        required=True,
        help='End time (formats: seconds "32", MM:SS "0:32", or HH:MM:SS "1:05:45")'
    )
    parser.add_argument(
        '--output', '-o',
        default='segment.wav',
        help='Output filename (default: segment.wav)'
    )
    parser.add_argument(
        '--keep-full',
        action='store_true',
        help='Keep the full downloaded audio file'
    )

    args = parser.parse_args()

    try:
        # Check if ffmpeg is available
        if not check_ffmpeg():
            print("\n❌ Error: ffmpeg and ffprobe are required but not found!", file=sys.stderr)
            print("\nPlease install ffmpeg:", file=sys.stderr)
            print("  - Windows: Download from https://ffmpeg.org/download.html", file=sys.stderr)
            print("            Or use: winget install ffmpeg", file=sys.stderr)
            print("  - macOS:   brew install ffmpeg", file=sys.stderr)
            print("  - Linux:   sudo apt-get install ffmpeg (Ubuntu/Debian)", file=sys.stderr)
            print("            Or: sudo yum install ffmpeg (RedHat/CentOS)", file=sys.stderr)
            print("\nMake sure ffmpeg is in your system PATH after installation.", file=sys.stderr)
            sys.exit(1)

        # Parse times
        start_time = parse_time(args.start)
        end_time = parse_time(args.end)

        print(f"Segment: {format_time(start_time)} to {format_time(end_time)}")
        print(f"Duration: {end_time - start_time:.2f} seconds")
        print()

        # Create temporary file for full download
        with tempfile.NamedTemporaryFile(delete=False, suffix='') as tmp_file:
            temp_path = tmp_file.name

        full_audio_path = None  # Initialize to avoid reference errors
        try:
            # Step 1: Download YouTube video
            full_audio_path = download_youtube_video(args.url, temp_path)

            # Step 2: Extract segment
            extract_segment(full_audio_path, args.output, start_time, end_time)

            print(f"\n{'='*60}")
            print(f"✓ Success!")
            print(f"  Output: {args.output}")
            print(f"  Segment: {format_time(start_time)} - {format_time(end_time)}")
            print(f"  Duration: {end_time - start_time:.2f}s")
            print(f"{'='*60}")

        finally:
            # Cleanup temporary file
            if full_audio_path and not args.keep_full:
                try:
                    if os.path.exists(full_audio_path):
                        os.remove(full_audio_path)
                        print(f"\n✓ Cleaned up temporary file")
                except Exception as e:
                    print(f"Warning: Could not remove temporary file: {e}", file=sys.stderr)
            elif full_audio_path and args.keep_full:
                print(f"\nFull audio saved: {full_audio_path}")

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
