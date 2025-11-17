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
from tqdm import tqdm


def download_youtube_video(url: str, output_dir: str = "downloads") -> str:
    """
    Download YouTube video and extract audio.

    Args:
        url: YouTube video URL
        output_dir: Directory to save the downloaded audio

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
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info['id']
        audio_path = os.path.join(output_dir, f"{video_id}.wav")

    print(f"✓ Audio downloaded: {audio_path}")
    return audio_path


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


def export_audio_segments(audio_path: str, segments: list, output_dir: str = "segments"):
    """
    Export audio segments to individual files.

    Args:
        audio_path: Path to original audio file
        segments: List of segment dictionaries
        output_dir: Directory to save segment files
    """
    os.makedirs(output_dir, exist_ok=True)

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

        # Create filename
        speaker_info = f"_speaker{segment['speaker']}" if segment.get('speaker') else ""
        filename = f"segment_{idx+1:03d}{speaker_info}_{duration:.1f}s.wav"
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
            'speaker': segment.get('speaker')
        })

    # Save metadata as JSON
    metadata_path = os.path.join(output_dir, "segments_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✓ Segments exported to: {output_dir}")
    print(f"✓ Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube video and segment it using WhisperX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s https://www.youtube.com/watch?v=dQw4w9WgXcQ
  %(prog)s https://www.youtube.com/watch?v=dQw4w9WgXcQ --min-duration 10
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

    args = parser.parse_args()

    try:
        # Step 1: Download YouTube video
        audio_path = download_youtube_video(args.url, output_dir="temp_downloads")

        # Step 2: Transcribe with WhisperX
        result = transcribe_with_whisperx(audio_path, device=args.device, model_name=args.model)

        # Step 3: Merge short segments
        print(f"\nMerging segments (min duration: {args.min_duration}s)...")
        segments = result.get('segments', [])
        merged_segments = merge_short_segments(segments, min_duration=args.min_duration)
        print(f"✓ Merged from {len(segments)} to {len(merged_segments)} segments")

        # Step 4: Export audio segments
        export_audio_segments(audio_path, merged_segments, output_dir=args.output_dir)

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
