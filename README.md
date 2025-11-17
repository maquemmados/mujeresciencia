# YouTube Video Segmentation with WhisperX

A Python script that downloads YouTube videos and segments them into individual audio files based on utterances detected by WhisperX, with a minimum duration of 5 seconds.

## Features

- Download audio from YouTube videos using yt-dlp
- Transcribe and segment audio using WhisperX with the best available model (large-v3)
- Automatic speaker diarization (speaker detection)
- Merge short segments to meet minimum duration requirements (default: 5 seconds)
- **Audio normalization for perception studies** (LUFS, RMS, or Peak normalization)
- Export segments with metadata (timestamps, text, speaker info)
- Support for GPU acceleration (CUDA) when available

## Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system
- (Optional) CUDA-compatible GPU for faster processing

### Installing FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

## Installation

1. Clone or download this repository:
```bash
git clone <repository-url>
cd mujeresciencia
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Download and segment a YouTube video:

```bash
python segment_youtube_video.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

This will:
- Download the audio from the YouTube video
- Transcribe it using WhisperX large-v3 model
- Segment the audio with a minimum duration of 5 seconds
- Save segments to the `output/` directory

### Advanced Options

```bash
python segment_youtube_video.py [URL] [OPTIONS]
```

**Options:**

- `--output-dir DIR` - Output directory for segments (default: `output`)
- `--min-duration SECONDS` - Minimum duration for segments in seconds (default: 5.0)
- `--model MODEL` - WhisperX model to use (default: `large-v3`)
  - Available models: `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`
  - `large-v3` provides the best quality
- `--device DEVICE` - Device to use: `cuda` or `cpu` (default: auto-detect)
- `--normalize METHOD` - Audio normalization method (default: `none`)
  - **`lufs`**: Loudness normalization (ITU-R BS.1770-4 / EBU R128) - **Recommended for perception studies**
  - `rms`: RMS (Root Mean Square) normalization
  - `peak`: Peak normalization
  - `none`: No normalization
- `--target-level LEVEL` - Target normalization level in dB/LUFS (default: -23.0)
  - For LUFS: -23.0 LUFS is the EBU R128 broadcast standard
  - For RMS: -20.0 to -23.0 dB is typical
  - For Peak: -1.0 to -3.0 dB is common
- `--keep-audio` - Keep the downloaded audio file after processing

### Examples

**Segment with 10-second minimum duration:**
```bash
python segment_youtube_video.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --min-duration 10
```

**With LUFS normalization (recommended for perception studies):**
```bash
python segment_youtube_video.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --normalize lufs
```

**With RMS normalization and custom target level:**
```bash
python segment_youtube_video.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --normalize rms --target-level -20.0
```

**With peak normalization:**
```bash
python segment_youtube_video.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --normalize peak --target-level -3.0
```

**Use a different model for faster processing:**
```bash
python segment_youtube_video.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --model medium
```

**Force CPU usage:**
```bash
python segment_youtube_video.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --device cpu
```

**Keep original audio file:**
```bash
python segment_youtube_video.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --keep-audio
```

**Custom output directory:**
```bash
python segment_youtube_video.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --output-dir my_segments
```

**Complete example for perception study:**
```bash
python segment_youtube_video.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
  --min-duration 5 \
  --normalize lufs \
  --target-level -23.0 \
  --output-dir perception_study_stimuli
```

## Output

The script creates an output directory with:

1. **Audio segment files** - Named as `segment_XXX_speakerY_Zs_normMETHOD.wav`
   - `XXX`: Segment number (001, 002, etc.)
   - `speakerY`: Speaker ID (if detected)
   - `Zs`: Duration in seconds
   - `normMETHOD`: Normalization method (if applied)

2. **Metadata file** - `segments_metadata.json` containing:
   - Segment ID
   - Filename
   - Start and end timestamps
   - Duration
   - Transcribed text
   - Speaker information (if available)
   - Normalization method and target level (if applied)

### Example Output Structure

**Without normalization:**
```
output/
├── segment_001_speaker0_8.2s.wav
├── segment_002_speaker1_6.5s.wav
├── segment_003_speaker0_12.3s.wav
├── ...
└── segments_metadata.json
```

**With LUFS normalization:**
```
output/
├── segment_001_speaker0_8.2s_normlufs.wav
├── segment_002_speaker1_6.5s_normlufs.wav
├── segment_003_speaker0_12.3s_normlufs.wav
├── ...
└── segments_metadata.json
```

### Example Metadata

**Without normalization:**
```json
[
  {
    "segment_id": 1,
    "filename": "segment_001_speaker0_8.2s.wav",
    "start_time": 0.0,
    "end_time": 8.2,
    "duration": 8.2,
    "text": "Welcome to this tutorial on machine learning.",
    "speaker": "SPEAKER_00",
    "normalization": null,
    "target_level": null
  }
]
```

**With LUFS normalization:**
```json
[
  {
    "segment_id": 1,
    "filename": "segment_001_speaker0_8.2s_normlufs.wav",
    "start_time": 0.0,
    "end_time": 8.2,
    "duration": 8.2,
    "text": "Welcome to this tutorial on machine learning.",
    "speaker": "SPEAKER_00",
    "normalization": "lufs",
    "target_level": -23.0
  },
  {
    "segment_id": 2,
    "filename": "segment_002_speaker1_6.5s_normlufs.wav",
    "start_time": 8.2,
    "end_time": 14.7,
    "duration": 6.5,
    "text": "Today we'll discuss neural networks.",
    "speaker": "SPEAKER_01",
    "normalization": "lufs",
    "target_level": -23.0
  }
]
```

## How It Works

1. **Download**: Uses yt-dlp to download the best quality audio from YouTube
2. **Transcription**: Uses WhisperX's large-v3 model for accurate transcription
3. **Alignment**: Aligns transcription with precise timestamps
4. **Diarization**: Detects different speakers in the audio
5. **Segmentation**: Merges short segments to meet minimum duration
6. **Normalization** (optional): Applies audio normalization for consistent loudness across segments
   - **LUFS**: Perceptual loudness normalization (ITU-R BS.1770-4 standard)
   - **RMS**: Energy-based normalization
   - **Peak**: Maximum amplitude normalization
7. **Export**: Saves individual audio files and metadata

## Audio Normalization for Perception Studies

Audio normalization ensures consistent loudness across all segments, which is crucial for perception studies where you want to control for volume differences. The script offers three methods:

### LUFS (Loudness Units relative to Full Scale)
- **Recommended for perception studies**
- Measures perceived loudness using psychoacoustic models
- Follows ITU-R BS.1770-4 / EBU R128 broadcast standards
- Default: -23.0 LUFS (EBU R128 standard)
- Best for: Studies where perceived loudness matters

### RMS (Root Mean Square)
- Measures average signal energy
- Simple and computationally efficient
- Default: -23.0 dB
- Best for: Quick normalization, compatibility with older workflows

### Peak Normalization
- Normalizes based on maximum amplitude
- Preserves relative loudness differences between segments
- Default: -1.0 dB (leaves headroom to prevent clipping)
- Best for: Preventing clipping while maintaining dynamics

**Example for perception studies:**
```bash
# Use LUFS normalization at broadcast standard level
python segment_youtube_video.py "VIDEO_URL" --normalize lufs --target-level -23.0
```

## Performance Notes

- **GPU Recommended**: Processing is significantly faster with a CUDA-compatible GPU
- **Model Size**: `large-v3` provides best quality but requires more resources
  - For faster processing on CPU, use `medium` or `small` models
- **Processing Time**: Expect ~1-2x real-time on GPU, 5-10x on CPU (varies by model)

## Troubleshooting

**Issue: "CUDA out of memory"**
- Use a smaller model: `--model medium`
- Force CPU usage: `--device cpu`

**Issue: "FFmpeg not found"**
- Install FFmpeg (see Prerequisites)
- Ensure FFmpeg is in your system PATH

**Issue: Diarization fails**
- The script will continue without speaker detection
- Segments will still be created without speaker labels

**Issue: Download fails**
- Check that the YouTube URL is valid and accessible
- Some videos may have restrictions

## License

This project uses the following open-source libraries:
- yt-dlp (Unlicense)
- WhisperX (BSD License)
- PyTorch (BSD License)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- [WhisperX](https://github.com/m-bain/whisperX) for accurate transcription and alignment
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube downloading
- [OpenAI Whisper](https://github.com/openai/whisper) for the base speech recognition model
- [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) for ITU-R BS.1770-4 loudness normalization
