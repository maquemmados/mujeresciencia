# YouTube Video Segment Downloader - Usage Examples

This repository contains two scripts for downloading and processing YouTube videos:

## 1. Simple Segment Extraction (`download_youtube_segment.py`)

A simple script to download a YouTube video and extract a specific time segment as a WAV audio file.

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Make sure ffmpeg is installed on your system
# Ubuntu/Debian: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
# Windows: download from https://ffmpeg.org/
```

### Usage Examples

```bash
# Extract segment from 0:16 to 0:32 (MM:SS format)
./download_youtube_segment.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --start 0:16 --end 0:32

# Same thing, using seconds
./download_youtube_segment.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --start 16 --end 32

# Specify custom output filename
./download_youtube_segment.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --start 16 --end 32 --output my_segment.wav

# Extract from 1:30 to 2:15
./download_youtube_segment.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --start 1:30 --end 2:15

# Keep the full downloaded audio file
./download_youtube_segment.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --start 16 --end 32 --keep-full
```

### Time Format Support

The script supports multiple time formats:
- **Seconds**: `16` or `16.5`
- **MM:SS**: `1:30` or `01:30`
- **HH:MM:SS**: `1:05:30`

---

## 2. Automatic Segmentation (`segment_youtube_video.py`)

A more advanced script that uses WhisperX AI to automatically detect speech segments and export them with transcriptions.

### Usage Examples

```bash
# Basic usage with automatic segmentation
./segment_youtube_video.py https://www.youtube.com/watch?v=dQw4w9WgXcQ

# Specify minimum segment duration (10 seconds)
./segment_youtube_video.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --min-duration 10

# Use with audio normalization for perception studies
./segment_youtube_video.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --normalize lufs

# Custom normalization level
./segment_youtube_video.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --normalize rms --target-level -20.0
```

---

## Quick Start Guide

For most users who just want to extract a specific segment, use the **simple script**:

```bash
./download_youtube_segment.py <YouTube_URL> --start <start_time> --end <end_time>
```

For automatic speech-based segmentation and transcription, use the **advanced script**:

```bash
./segment_youtube_video.py <YouTube_URL>
```
