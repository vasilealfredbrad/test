# TTS-to-Video Pipeline with AI Video Generation

Complete Python script that generates speech from text using Hugging Face's Bark TTS model, extracts word-level timestamps using OpenAI's Whisper, produces synchronized SRT subtitle files, and creates videos with **two powerful modes**: use existing video or generate AI video from text!

## 🎬 Features

### Core Features
- ✅ **Text-to-Speech**: Natural-sounding speech using `suno/bark-small`
- ✅ **GPU Acceleration**: Automatic CUDA detection for faster processing
- ✅ **Word-Level Timestamps**: Precise timing with Whisper `large-v3`
- ✅ **SRT Subtitle Generation**: Professional subtitle formatting
- ✅ **No FFmpeg Required**: Direct audio/video processing

### 🆕 Video Generation Modes

#### **Option 1: Use Existing Video** 🎥
- Replace video's audio with generated speech
- Burn subtitles directly into the video
- Automatic resolution scaling (720p/1080p)
- Fast and straightforward

#### **Option 2: Generate AI Video from Text** 🤖
- Create video using AI text-to-video models
- **Dual Emotion Detection**:
  - **Text Analysis**: Sentiment analysis (positive/negative/neutral)
  - **Audio Analysis**: Pitch, tempo, energy, voice quality
- Emotion-guided video generation for realistic results
- Cinematic quality output

### Additional Features
- 📊 **Dual Emotion Detection**: Analyze both text sentiment and audio characteristics
- 🎨 **Configurable Resolutions**: 720p or 1080p output
- 🎬 **Professional Frame Rates**: 24-30 FPS
- 💬 **Interactive Menu**: Easy mode selection
- 📝 **Well-Commented Code**: Educational and maintainable

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA (recommended, 8GB+ VRAM for Option 2)
- 15GB+ disk space for models

### Installation

#### 1. Create Virtual Environment

```bash
python -m venv venv
```

#### 2. Activate Virtual Environment

**Windows:**
```powershell
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

#### 3. Install PyTorch with CUDA

Check your CUDA version:
```bash
nvidia-smi
```

Install PyTorch (for CUDA 12.x):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 4. Install Dependencies

```bash
pip install transformers openai-whisper scipy accelerate moviepy diffusers librosa soundfile textblob
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Run the Script

```bash
python generate_subtitles.py
```

### Interactive Menu

You'll be presented with a menu:

```
VIDEO GENERATION MODE SELECTION
======================================================================

Please select how you want to create your video:

  [1] Use Existing Video
      → Replace audio with generated speech
      → Burn subtitles directly into the video
      → Fast and simple

  [2] Generate AI Video from Text
      → Create video using AI text-to-video models
      → Emotion-guided generation (text + audio analysis)
      → More creative but slower

  [0] Skip Video Generation
      → Only generate audio and subtitles
```

### Option 1: Use Existing Video

1. Select option `1`
2. Choose resolution (720p or 1080p)
3. Provide path to your existing video file
4. Script will:
   - Replace audio with generated speech
   - Burn subtitles into the video
   - Scale to selected resolution
   - Output: `out/final_video_720p.mp4` (or 1080p)

### Option 2: Generate AI Video

1. Select option `2`
2. Choose resolution (720p or 1080p)
3. Script will:
   - Analyze text sentiment (positive/negative/neutral)
   - Analyze audio features (pitch, tempo, energy)
   - Combine emotions for video guidance
   - Generate AI video with emotion-appropriate visuals
   - Add generated audio and subtitles
   - Output: `out/ai_generated_video_720p.mp4`

**⚠️ Note:** Option 2 requires significant GPU memory (~8GB+) and takes 10-30 minutes.

### Option 0: Audio & Subtitles Only

1. Select option `0`
2. Script generates:
   - `out/speech.wav` - Generated audio
   - `out/subtitles.srt` - Subtitle file
3. Use these files with your own video editor

## ⚙️ Configuration

### Customize Story Text

Edit the `STORY` variable in `generate_subtitles.py` (line 27):

```python
STORY = """
Your custom story text here.
Can be multiple paragraphs.
The emotion detection will analyze this text.
"""
```

### Configuration Options

```python
# Force CPU usage (disable GPU)
FORCE_CPU = False  # Set to True to use CPU only

# Whisper model size
ASR_MODEL = "large-v3"  # Options: tiny, base, small, medium, large-v3, turbo

# Text-to-video model (for Option 2)
TEXT_TO_VIDEO_MODEL = "damo-vilab/text-to-video-ms-1.7b"

# Subtitle formatting
MAX_WORDS_PER_CAPTION = 10  # Words per subtitle line
MAX_CHARS_PER_CAPTION = 50  # Characters per line

# Video settings
DEFAULT_RESOLUTION = "720p"  # "720p" or "1080p"
DEFAULT_FPS = 30  # 24, 30, or 60
```

## 📊 Emotion Detection

The script uses **dual emotion analysis** for realistic video generation:

### Text Emotion Analysis (TextBlob)
- **Polarity**: -1.0 (negative) to +1.0 (positive)
- **Subjectivity**: 0.0 (objective) to 1.0 (subjective)
- **Detected emotions**: positive/happy, negative/sad, neutral/calm

### Audio Emotion Analysis (Librosa)
- **Pitch**: Fundamental frequency (Hz)
- **Tempo**: Speed in BPM
- **Energy**: Loudness/intensity (RMS)
- **Voice Quality**: Zero-crossing rate
- **Detected emotions**: energetic/excited, calm/relaxed, high-pitched/cheerful

### Combined Profile

Emotions are combined to guide AI video generation:
```
"A video with positive mood and energetic voice presentation"
```

## 🎯 Performance

### GPU (RTX 3060 12GB)

**Option 0 (Audio + Subtitles only):**
- TTS Generation: 30-60 seconds for 14-second audio
- Whisper Transcription: 1-2 minutes
- **Total**: ~2-3 minutes

**Option 1 (Existing Video):**
- TTS + Whisper: 2-3 minutes
- Video Processing: 2-5 minutes
- **Total**: ~5-8 minutes

**Option 2 (AI Video Generation):**
- TTS + Whisper: 2-3 minutes
- AI Video Generation: 10-30 minutes (depends on GPU)
- **Total**: ~15-35 minutes

### CPU Only

- TTS Generation: 5-10 minutes
- Whisper: 3-5 minutes
- Option 1: +5-10 minutes
- Option 2: Not recommended (very slow)

## 📁 Project Structure

```
facelessvideo/
│
├── generate_subtitles.py   # Main enhanced script with video modes
├── requirements.txt         # All dependencies with GPU instructions
├── README.md               # This comprehensive guide
├── .gitignore              # Git ignore patterns
│
├── venv/                   # Virtual environment (not in git)
│
└── out/                    # Generated files
    ├── speech.wav          # Generated audio (24kHz)
    ├── subtitles.srt       # Synchronized subtitles
    ├── final_video_720p.mp4           # Option 1 output
    └── ai_generated_video_720p.mp4    # Option 2 output
```

## 🔧 Troubleshooting

### "CUDA not available"
- Install PyTorch with CUDA (see Installation step 3)
- Verify with `nvidia-smi`

### "Out of memory" (Option 2)
- Use 720p instead of 1080p
- Close other GPU applications
- Reduce text length in `STORY`
- Use Option 1 instead

### MoviePy Errors
```bash
pip install --upgrade moviepy
pip install imageio-ffmpeg
```

### Diffusers Model Download Slow
- First run downloads ~7GB for text-to-video model
- Check internet connection
- Models cache to `~/.cache/huggingface/`

### TextBlob Installation
```bash
pip install textblob
python -m textblob.download_corpora
```

### Librosa Audio Analysis Errors
```bash
pip install librosa soundfile numba
```

## 🎨 Use Cases

### Option 1: Use Existing Video
- **Educational Content**: Replace voiceover with different language
- **Tutorials**: Add narration to screen recordings
- **Product Demos**: Update audio without re-recording
- **Accessibility**: Add subtitles to existing videos

### Option 2: Generate AI Video
- **Faceless YouTube Videos**: Create content without camera
- **Podcast Visualizations**: Convert audio content to video
- **Social Media**: Generate engaging video content
- **Prototyping**: Quick video mockups from text

## 🤖 Models Used

1. **Bark TTS** (`suno/bark-small`)
   - License: MIT
   - Size: ~1.5GB
   - Quality: Natural multilingual TTS

2. **Whisper ASR** (`openai/whisper-large-v3`)
   - License: MIT
   - Size: ~3GB
   - Quality: SOTA speech recognition

3. **Text-to-Video** (`damo-vilab/text-to-video-ms-1.7b`)
   - License: Creative ML Open RAIL-M
   - Size: ~7GB
   - Quality: High-quality video generation

4. **TextBlob** (Sentiment Analysis)
   - License: MIT
   - For text emotion detection

5. **Librosa** (Audio Analysis)
   - License: ISC
   - For audio emotion features

## 📝 Example Workflow

### Complete Workflow (Option 2)

```python
# 1. Edit your story
STORY = """
Welcome to my exciting tutorial!
Today we'll explore amazing AI technology.
Get ready for an incredible journey!
"""

# 2. Run script
python generate_subtitles.py

# 3. Select Option 2 (AI Video Generation)
# 4. Choose 720p resolution
# 5. Wait for processing (~15-20 minutes)

# Output:
# - Text emotion: "positive/happy" (detected from "exciting", "amazing", "incredible")
# - Audio emotion: "energetic/excited" (high tempo, high energy)
# - Combined: "A video with positive mood and energetic voice"
# - AI generates video with uplifting, dynamic visuals
# - Final video with speech audio and subtitles
```

## 🌟 Advanced Tips

### Optimize for Speed
```python
# Use smaller models
ASR_MODEL = "base"  # Instead of "large-v3"

# Skip Option 2 if you have limited GPU
# Use Option 1 with pre-made video instead
```

### Improve Video Quality (Option 2)
```python
# In option2_generate_ai_video function, increase inference steps:
video_frames = pipe(
    prompt,
    num_inference_steps=50,  # Default: 25, Higher = better quality
    num_frames=120  # Default: 60, More frames = smoother
).frames[0]
```

### Customize Emotion Detection
```python
# Adjust emotion thresholds in detect_emotion_from_text:
if sentiment.polarity > 0.5:  # More strict for "positive"
    emotion = "very positive/excited"
elif sentiment.polarity > 0.1:  # Add neutral-positive
    emotion = "slightly positive"
```

## 📄 License

This project is provided as-is for educational and commercial use. The models have their respective licenses:
- Bark TTS: MIT
- Whisper ASR: MIT
- Text-to-Video: Creative ML Open RAIL-M

## 🙏 Credits

- **Bark TTS**: [suno-ai/bark](https://github.com/suno-ai/bark)
- **Whisper ASR**: [OpenAI Whisper](https://github.com/openai/whisper)
- **Hugging Face**: [transformers](https://github.com/huggingface/transformers) & [diffusers](https://github.com/huggingface/diffusers)
- **MoviePy**: [Zulko/moviepy](https://github.com/Zulko/moviepy)
- **Librosa**: Audio analysis library
- **TextBlob**: Sentiment analysis

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements!

---

**Made with ❤️ by AI Assistant**

*Transform your text into professional videos with emotion-aware AI!*
