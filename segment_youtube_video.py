#!/usr/bin/env python3
"""
YouTube Video Segmentation using WhisperX

This script downloads a YouTube video and segments it into audio files
based on utterances detected by WhisperX, with a minimum duration of 5 seconds.
"""

import argparse
import os
import sys
from pathlib import Path
import json

import yt_dlp
import whisperx
import torch
import soundfile as sf
import librosa
import numpy as np
import pyloudnorm as pyln
from tqdm import tqdm


def download_youtube_video(url: str, output_dir: str = "downloads", cookies_file: str = None) -> str:
    """
    Download YouTube video and extract audio.

    Args:
        url: YouTube video URL
        output_dir: Directory to save the downloaded audio
        cookies_file: Path to cookies file (for bypassing restrictions)

    Returns:
        Path to the downloaded audio file
    """
    print(f"Downloading YouTube video: {url}")

    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'quiet': False,
        # Options to help avoid 403 errors
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web'],
            }
        },
    }

    # Add cookies if provided
    if cookies_file:
        if not os.path.exists(cookies_file):
            print(f"Warning: Cookies file not found: {cookies_file}")
        else:
            ydl_opts['cookiefile'] = cookies_file
            print(f"Using cookies from: {cookies_file}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info['id']
            audio_path = os.path.join(output_dir, f"{video_id}.wav")

        print(f"✓ Audio downloaded: {audio_path}")
        return audio_path

    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        if "403" in error_msg or "Forbidden" in error_msg:
            print("\n" + "="*70)
            print("ERROR: YouTube blocked the download (HTTP 403 Forbidden)")
            print("="*70)
            print("\nThis usually happens due to YouTube's anti-bot measures.")
            print("\nTry these solutions:\n")
            print("1. Update yt-dlp to the latest version:")
            print("   pip install --upgrade yt-dlp")
            print("\n2. Use browser cookies (RECOMMENDED):")
            print("   - Install browser extension: 'Get cookies.txt'")
            print("     Chrome: https://chrome.google.com/webstore/detail/get-cookiestxt/bgaddhkoddajcdgocldbbfleckgcbcid")
            print("     Firefox: https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/")
            print("   - Go to youtube.com and make sure you're logged in")
            print("   - Click the extension and export cookies")
            print("   - Save as 'cookies.txt'")
            print("   - Run: python segment_youtube_video.py URL --cookies cookies.txt")
            print("\n3. Try a different video (some videos have stricter restrictions)")
            print("\n4. Wait a few minutes and try again (rate limiting)")
            print("="*70)
        raise


def transcribe_with_whisperx(audio_path: str, device: str = None,
                             model_name: str = "large-v3") -> dict:
    """
    Transcribe audio using WhisperX with the best model.

    Args:
        audio_path: Path to audio file
        device: Device to use ('cuda' or 'cpu'). Auto-detect if None.
        model_name: WhisperX model name (default: large-v3 - the best model)

    Returns:
        Dictionary containing transcription results with segments
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading WhisperX model: {model_name} on {device}")

    # Load audio
    audio = whisperx.load_audio(audio_path)

    # Load model
    model = whisperx.load_model(model_name, device, compute_type="float16" if device == "cuda" else "float32")

    # Transcribe
    print("Transcribing audio...")
    result = model.transcribe(audio, batch_size=16)

    # Align whisper output
    print("Aligning transcription...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # Diarization (speaker detection) - optional but improves segmentation
    try:
        print("Performing speaker diarization...")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=None, device=device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
    except Exception as e:
        print(f"Warning: Diarization failed ({e}). Continuing without speaker info.")

    print(f"✓ Transcription complete. Language: {result.get('language', 'unknown')}")

    return result


def merge_short_segments(segments: list, min_duration: float = 5.0) -> list:
    """
    Merge segments to ensure minimum duration.

    Args:
        segments: List of segment dictionaries with 'start', 'end', and 'text' keys
        min_duration: Minimum duration in seconds for each segment

    Returns:
        List of merged segments
    """
    if not segments:
        return []

    merged = []
    current_segment = {
        'start': segments[0]['start'],
        'end': segments[0]['end'],
        'text': segments[0]['text'],
        'speaker': segments[0].get('speaker', None)
    }

    for segment in segments[1:]:
        duration = current_segment['end'] - current_segment['start']

        # If current segment is too short, merge with next
        if duration < min_duration:
            current_segment['end'] = segment['end']
            current_segment['text'] += ' ' + segment['text']
            # Keep speaker info if consistent
            if current_segment['speaker'] != segment.get('speaker'):
                current_segment['speaker'] = None
        else:
            # Current segment is long enough, save it and start new one
            merged.append(current_segment)
            current_segment = {
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
                'speaker': segment.get('speaker', None)
            }

    # Add the last segment
    merged.append(current_segment)

    return merged


def normalize_audio(audio: np.ndarray, sample_rate: int, method: str = "lufs",
                   target_level: float = -23.0) -> np.ndarray:
    """
    Normalize audio for perception studies.

    Args:
        audio: Audio signal as numpy array
        sample_rate: Sample rate of the audio
        method: Normalization method ('lufs', 'rms', 'peak', or 'none')
        target_level: Target level for normalization
            - For LUFS: target loudness in LUFS (default: -23.0 LUFS, EBU R128 standard)
            - For RMS: target RMS level in dB (default: -23.0 dB)
            - For peak: target peak level in dB (default: -1.0 dB)

    Returns:
        Normalized audio signal
    """
    if method == "none":
        return audio

    # Ensure audio is float
    audio = audio.astype(np.float32)

    if method == "lufs":
        # Loudness normalization using ITU-R BS.1770-4 / EBU R128
        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(audio)

        # Prevent errors with silent audio
        if np.isinf(loudness) or loudness < -70:
            print(f"  Warning: Audio segment is too quiet (loudness: {loudness:.1f} LUFS), skipping normalization")
            return audio

        # Normalize to target LUFS
        normalized_audio = pyln.normalize.loudness(audio, loudness, target_level)
        return normalized_audio

    elif method == "rms":
        # RMS (Root Mean Square) normalization
        rms = np.sqrt(np.mean(audio ** 2))

        if rms < 1e-8:
            print(f"  Warning: Audio segment is silent (RMS: {rms:.2e}), skipping normalization")
            return audio

        # Convert target dB to linear scale
        target_rms = 10 ** (target_level / 20.0)
        gain = target_rms / rms

        # Apply gain with clipping protection
        normalized_audio = audio * gain
        max_val = np.abs(normalized_audio).max()
        if max_val > 1.0:
            normalized_audio = normalized_audio / max_val * 0.99

        return normalized_audio

    elif method == "peak":
        # Peak normalization
        peak = np.abs(audio).max()

        if peak < 1e-8:
            print(f"  Warning: Audio segment is silent (peak: {peak:.2e}), skipping normalization")
            return audio

        # Convert target dB to linear scale
        target_peak = 10 ** (target_level / 20.0)
        gain = target_peak / peak
        normalized_audio = audio * gain

        return normalized_audio

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def export_audio_segments(audio_path: str, segments: list, output_dir: str = "segments",
                         normalize: str = "none", target_level: float = -23.0):
    """
    Export audio segments to individual files with optional normalization.

    Args:
        audio_path: Path to original audio file
        segments: List of segment dictionaries
        output_dir: Directory to save segment files
        normalize: Normalization method ('lufs', 'rms', 'peak', or 'none')
        target_level: Target level for normalization (in dB or LUFS)
    """
    os.makedirs(output_dir, exist_ok=True)

    if normalize != "none":
        print(f"\nExporting {len(segments)} audio segments with {normalize.upper()} normalization (target: {target_level})...")
    else:
        print(f"\nExporting {len(segments)} audio segments...")

    # Load audio
    audio, sr = librosa.load(audio_path, sr=None)

    # Export metadata
    metadata = []

    for idx, segment in enumerate(tqdm(segments, desc="Exporting segments")):
        start_time = segment['start']
        end_time = segment['end']
        duration = end_time - start_time

        # Convert time to samples
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Extract segment
        segment_audio = audio[start_sample:end_sample]

        # Apply normalization if requested
        if normalize != "none":
            segment_audio = normalize_audio(segment_audio, sr, method=normalize, target_level=target_level)

        # Create filename
        speaker_info = f"_speaker{segment['speaker']}" if segment.get('speaker') else ""
        norm_info = f"_norm{normalize}" if normalize != "none" else ""
        filename = f"segment_{idx+1:03d}{speaker_info}_{duration:.1f}s{norm_info}.wav"
        filepath = os.path.join(output_dir, filename)

        # Save audio
        sf.write(filepath, segment_audio, sr)

        # Save metadata
        metadata.append({
            'segment_id': idx + 1,
            'filename': filename,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'text': segment['text'],
            'speaker': segment.get('speaker'),
            'normalization': normalize if normalize != "none" else None,
            'target_level': target_level if normalize != "none" else None
        })

    # Save metadata as JSON
    metadata_path = os.path.join(output_dir, "segments_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✓ Segments exported to: {output_dir}")
    print(f"✓ Metadata saved to: {metadata_path}")
    if normalize != "none":
        print(f"✓ Audio normalized using {normalize.upper()} method (target: {target_level})")


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube video and segment it using WhisperX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s https://www.youtube.com/watch?v=dQw4w9WgXcQ
  %(prog)s https://www.youtube.com/watch?v=dQw4w9WgXcQ --min-duration 10
  %(prog)s https://www.youtube.com/watch?v=dQw4w9WgXcQ --normalize lufs
  %(prog)s https://www.youtube.com/watch?v=dQw4w9WgXcQ --cookies cookies.txt
  %(prog)s https://www.youtube.com/watch?v=dQw4w9WgXcQ --normalize rms --target-level -20.0
  %(prog)s https://www.youtube.com/watch?v=dQw4w9WgXcQ --model large-v2 --device cpu
        """
    )

    parser.add_argument(
        'url',
        help='YouTube video URL'
    )
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory for segments (default: output)'
    )
    parser.add_argument(
        '--min-duration',
        type=float,
        default=5.0,
        help='Minimum duration for segments in seconds (default: 5.0)'
    )
    parser.add_argument(
        '--model',
        default='large-v3',
        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
        help='WhisperX model to use (default: large-v3 - best quality)'
    )
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        help='Device to use for inference (default: auto-detect)'
    )
    parser.add_argument(
        '--keep-audio',
        action='store_true',
        help='Keep downloaded audio file'
    )
    parser.add_argument(
        '--cookies',
        type=str,
        help='Path to cookies.txt file (use browser cookies to bypass YouTube restrictions)'
    )
    parser.add_argument(
        '--normalize',
        default='none',
        choices=['none', 'lufs', 'rms', 'peak'],
        help='Audio normalization method for perception studies (default: none). '
             'LUFS is recommended for perceptual studies.'
    )
    parser.add_argument(
        '--target-level',
        type=float,
        default=-23.0,
        help='Target normalization level in dB/LUFS (default: -23.0). '
             'For LUFS: -23.0 is EBU R128 standard. '
             'For peak: -1.0 to -3.0 is common. '
             'For RMS: -20.0 to -23.0 is typical.'
    )

    args = parser.parse_args()

    try:
        # Step 1: Download YouTube video
        audio_path = download_youtube_video(args.url, output_dir="temp_downloads", cookies_file=args.cookies)

        # Step 2: Transcribe with WhisperX
        result = transcribe_with_whisperx(audio_path, device=args.device, model_name=args.model)

        # Step 3: Merge short segments
        print(f"\nMerging segments (min duration: {args.min_duration}s)...")
        segments = result.get('segments', [])
        merged_segments = merge_short_segments(segments, min_duration=args.min_duration)
        print(f"✓ Merged from {len(segments)} to {len(merged_segments)} segments")

        # Step 4: Export audio segments
        export_audio_segments(audio_path, merged_segments, output_dir=args.output_dir,
                            normalize=args.normalize, target_level=args.target_level)

        # Cleanup
        if not args.keep_audio:
            os.remove(audio_path)
            print(f"\n✓ Cleaned up temporary audio file")

        print(f"\n{'='*60}")
        print(f"✓ Processing complete!")
        print(f"  - Total segments: {len(merged_segments)}")
        print(f"  - Output directory: {args.output_dir}")
        print(f"  - Metadata: {os.path.join(args.output_dir, 'segments_metadata.json')}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
