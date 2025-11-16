"""
Complete TTS-to-Video Pipeline Script with Multi-Mode Video Generation

This script:
1. Converts text to speech using Hugging Face Bark TTS model
2. Uses OpenAI Whisper to extract word-level timestamps from the generated audio
3. Generates an SRT subtitle file with properly timed captions
4. Provides TWO video generation modes:
   - Option 1: Use existing video (replace audio + burn subtitles)
   - Option 2: Generate AI video from text with emotion detection

Author: AI Assistant
Date: 2025
"""

import os
import sys
from pathlib import Path
import time
import torch
import whisper
import numpy as np
from transformers import AutoProcessor, BarkModel
from scipy.io.wavfile import write as write_wav
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Hardcoded story text to convert to speech
STORY = """
Meet Alin, a Romanian guy who accidentally became an AI expert overnight.
It all started when he tried to fix his grandmother's computer.
He clicked on everything, downloaded random files, and somehow installed Python.
Now his friends think he's a genius programmer who can hack anything!
Alin doesn't have the heart to tell them he still Googles how to center a div.
His secret weapon? Copying code from Stack Overflow and hoping for the best.
But hey, fake it till you make it, right? That's the Romanian way!
Now he's creating AI videos to prove he's legit. Wish him luck!
"""

# Output directory for generated files
OUTPUT_DIR = Path("out")

# Model identifiers
TTS_MODEL = "suno/bark-small"
ASR_MODEL = "large-v3"  # Whisper model: tiny, base, small, medium, large-v3, turbo

# Text-to-video models (choose based on quality/speed preference)
# Option 1: Stability AI (Best quality, slower) - Recommended!
TEXT_TO_VIDEO_MODEL = "stabilityai/stable-video-diffusion-img2vid-xt"
USE_IMG2VID = True  # Use image-to-video for much better quality

# Option 2: AnimateDiff (Good quality, faster)
# TEXT_TO_VIDEO_MODEL = "guoyww/animatediff-motion-adapter-v1-5-2"
# USE_IMG2VID = False

# Option 3: Original (Lower quality, fastest)
# TEXT_TO_VIDEO_MODEL = "damo-vilab/text-to-video-ms-1.7b"
# USE_IMG2VID = False

# Device configuration
FORCE_CPU = False  # Change to True if you want to force CPU usage

# Subtitle formatting parameters
MAX_WORDS_PER_CAPTION = 10  # Maximum words per subtitle line
MAX_CHARS_PER_CAPTION = 50  # Maximum characters per subtitle line

# Video configuration
VIDEO_RESOLUTIONS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080)
}
DEFAULT_RESOLUTION = "720p"
DEFAULT_FPS = 30


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_output_directory():
    """Create the output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"✓ Output directory created/verified: {OUTPUT_DIR.absolute()}")


def format_timestamp_srt(seconds):
    """
    Convert seconds (float) to SRT timestamp format: HH:MM:SS,mmm
    
    Args:
        seconds: Time in seconds (float)
    
    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def group_words_into_captions(words):
    """
    Group words into readable caption segments.
    
    Args:
        words: List of word dictionaries with 'word', 'start', 'end' keys
    
    Returns:
        List of caption dictionaries with 'text', 'start', 'end' keys
    """
    captions = []
    current_caption = {
        'words': [],
        'start': None,
        'end': None,
        'text': ''
    }
    
    for word_info in words:
        word = word_info['word'].strip()
        
        # Skip empty words
        if not word:
            continue
        
        # Initialize start time for first word
        if current_caption['start'] is None:
            current_caption['start'] = word_info['start']
        
        # Add word to current caption
        current_caption['words'].append(word)
        current_caption['end'] = word_info['end']
        current_caption['text'] = ' '.join(current_caption['words'])
        
        # Check if we should start a new caption
        should_break = (
            len(current_caption['words']) >= MAX_WORDS_PER_CAPTION or
            len(current_caption['text']) >= MAX_CHARS_PER_CAPTION
        )
        
        if should_break:
            # Save current caption
            captions.append({
                'text': current_caption['text'],
                'start': current_caption['start'],
                'end': current_caption['end']
            })
            
            # Reset for next caption
            current_caption = {
                'words': [],
                'start': None,
                'end': None,
                'text': ''
            }
    
    # Add remaining words as final caption
    if current_caption['words']:
        captions.append({
            'text': current_caption['text'],
            'start': current_caption['start'],
            'end': current_caption['end']
        })
    
    return captions


def generate_srt_file(captions, output_path):
    """
    Generate an SRT subtitle file from caption data.
    
    Args:
        captions: List of caption dictionaries with 'text', 'start', 'end'
        output_path: Path where to save the SRT file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, caption in enumerate(captions, start=1):
            # Caption number
            f.write(f"{idx}\n")
            
            # Timestamp line
            start_time = format_timestamp_srt(caption['start'])
            end_time = format_timestamp_srt(caption['end'])
            f.write(f"{start_time} --> {end_time}\n")
            
            # Caption text
            f.write(f"{caption['text']}\n")
            
            # Blank line separator
            f.write("\n")
    
    print(f"✓ SRT subtitle file saved: {output_path.absolute()}")


def show_main_menu():
    """
    Display the main menu and get user's choice for video mode.
    
    Returns:
        int: User's choice (1 or 2)
    """
    print("\n" + "="*70)
    print("VIDEO GENERATION MODE SELECTION")
    print("="*70)
    print("\nPlease select how you want to create your video:\n")
    print("  [1] Use Existing Video")
    print("      → Replace audio with generated speech")
    print("      → Burn subtitles directly into the video")
    print("      → Fast and simple\n")
    print("  [2] Generate AI Video from Text")
    print("      → Create video using AI text-to-video models")
    print("      → Emotion-guided generation (text + audio analysis)")
    print("      → More creative but slower\n")
    print("  [0] Skip Video Generation")
    print("      → Only generate audio and subtitles\n")
    print("="*70)
    
    while True:
        try:
            choice = input("Enter your choice (0, 1, or 2): ").strip()
            if choice in ['0', '1', '2']:
                return int(choice)
            else:
                print("❌ Invalid choice. Please enter 0, 1, or 2.")
        except KeyboardInterrupt:
            print("\n\n❌ Operation cancelled by user.")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}. Please try again.")


def get_video_settings():
    """
    Get video resolution and FPS settings from user.
    
    Returns:
        tuple: (resolution_key, fps)
    """
    print("\n" + "-"*70)
    print("VIDEO SETTINGS")
    print("-"*70)
    print("\nSelect output resolution:")
    print("  [1] 720p  (1280x720)  - Recommended")
    print("  [2] 1080p (1920x1080) - Higher quality\n")
    
    while True:
        try:
            res_choice = input("Enter resolution choice (1 or 2) [default: 1]: ").strip() or "1"
            if res_choice == "1":
                resolution = "720p"
                break
            elif res_choice == "2":
                resolution = "1080p"
                break
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
        except Exception:
            resolution = "720p"
            break
    
    print(f"\nSelected resolution: {resolution}")
    print(f"Frame rate: {DEFAULT_FPS} fps")
    
    return resolution, DEFAULT_FPS


# ============================================================================
# EMOTION DETECTION MODULE
# ============================================================================

def detect_emotion_from_text(text):
    """
    Detect emotion/sentiment from text using TextBlob.
    
    Args:
        text: Input text string
    
    Returns:
        dict: Emotion analysis results with polarity and subjectivity
    """
    print("\n" + "-"*70)
    print("EMOTION DETECTION: Text Analysis")
    print("-"*70)
    
    try:
        from textblob import TextBlob
        
        blob = TextBlob(text)
        sentiment = blob.sentiment
        
        # Interpret polarity
        if sentiment.polarity > 0.3:
            emotion = "positive/happy"
        elif sentiment.polarity < -0.3:
            emotion = "negative/sad"
        else:
            emotion = "neutral/calm"
        
        print(f"  Sentiment polarity: {sentiment.polarity:.3f} ({emotion})")
        print(f"  Subjectivity: {sentiment.subjectivity:.3f}")
        
        return {
            'polarity': sentiment.polarity,
            'subjectivity': sentiment.subjectivity,
            'emotion': emotion,
            'description': f"{emotion} tone"
        }
    
    except ImportError:
        print("  ⚠ TextBlob not installed. Skipping text emotion detection.")
        return {'emotion': 'neutral', 'description': 'neutral'}
    except Exception as e:
        print(f"  ⚠ Error in text emotion detection: {e}")
        return {'emotion': 'neutral', 'description': 'neutral'}


def detect_emotion_from_audio(audio_path):
    """
    Detect emotion from audio using librosa (pitch, tempo, energy).
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        dict: Audio emotion features
    """
    print("\n" + "-"*70)
    print("EMOTION DETECTION: Audio Analysis")
    print("-"*70)
    
    try:
        import librosa
        
        # Load audio
        y, sr = librosa.load(str(audio_path))
        
        # Extract features
        # 1. Pitch (fundamental frequency)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = np.mean(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 0
        
        # 2. Tempo (speed)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # 3. Energy (loudness/intensity)
        rms = librosa.feature.rms(y=y)
        energy = np.mean(rms)
        
        # 4. Zero crossing rate (voice quality)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        
        # Interpret audio features (convert to Python native types for printing)
        tempo_val = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        
        if tempo_val > 140 and energy > 0.1:
            audio_emotion = "energetic/excited"
        elif tempo_val < 80 and energy < 0.05:
            audio_emotion = "calm/relaxed"
        elif pitch > 200:
            audio_emotion = "high-pitched/cheerful"
        else:
            audio_emotion = "neutral/conversational"
        
        print(f"  Pitch: {float(pitch):.1f} Hz")
        print(f"  Tempo: {tempo_val:.1f} BPM")
        print(f"  Energy: {float(energy):.4f}")
        print(f"  Voice quality (ZCR): {float(zcr_mean):.4f}")
        print(f"  Detected emotion: {audio_emotion}")
        
        # Convert numpy types to Python native types
        tempo_val = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        
        return {
            'pitch': float(pitch),
            'tempo': tempo_val,
            'energy': float(energy),
            'zcr': float(zcr_mean),
            'emotion': audio_emotion,
            'description': f"{audio_emotion} voice"
        }
    
    except ImportError:
        print("  ⚠ Librosa not installed. Skipping audio emotion detection.")
        return {'emotion': 'neutral', 'description': 'neutral voice'}
    except Exception as e:
        print(f"  ⚠ Error in audio emotion detection: {e}")
        return {'emotion': 'neutral', 'description': 'neutral voice'}


def combine_emotions(text_emotion, audio_emotion):
    """
    Combine text and audio emotion analysis into a unified description.
    
    Args:
        text_emotion: Dict from detect_emotion_from_text
        audio_emotion: Dict from detect_emotion_from_audio
    
    Returns:
        str: Combined emotion description for video generation
    """
    text_desc = text_emotion.get('emotion', 'neutral')
    audio_desc = audio_emotion.get('emotion', 'neutral')
    
    combined = f"A video with {text_desc} mood and {audio_desc} presentation"
    
    print("\n" + "-"*70)
    print(f"✓ Combined Emotion Profile: {combined}")
    print("-"*70)
    
    return combined


# ============================================================================
# MAIN PIPELINE FUNCTIONS (TTS + ASR)
# ============================================================================

def step1_text_to_speech(text):
    """
    Step 1: Convert text to speech using Bark TTS model.
    
    Args:
        text: Input text to convert to speech
    
    Returns:
        Path to the generated WAV file
    """
    print("\n" + "="*70)
    print("STEP 1: TEXT-TO-SPEECH GENERATION (Bark)")
    print("="*70)
    
    print(f"Loading TTS model: {TTS_MODEL}")
    
    # Initialize processor and model
    processor = AutoProcessor.from_pretrained(TTS_MODEL)
    
    # Determine device
    device = "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device == "cuda"
    
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"  ✓ GPU detected! Using CUDA for faster generation")
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        if not FORCE_CPU and not torch.cuda.is_available():
            print(f"  ⚠ GPU not available. Using CPU (slower)")
            print(f"  💡 To use GPU: Install PyTorch with CUDA support")
    
    # Load model with safetensors
    model = BarkModel.from_pretrained(
        TTS_MODEL,
        dtype=torch.float16 if use_fp16 else torch.float32,
        use_safetensors=True
    ).to(device)
    
    print(f"✓ Model loaded successfully")
    print(f"\nInput text ({len(text)} chars):")
    print(f"  \"{text[:100]}...\"")
    
    # Process and generate with progress indication
    print("\nGenerating speech...")
    start_time = time.time()
    
    inputs = processor(text, return_tensors="pt").to(device)
    
    print("  [████████████████████] Processing text input...")
    with torch.no_grad():
        audio_array = model.generate(**inputs)
    
    elapsed = time.time() - start_time
    print(f"  ✓ Speech generation completed in {elapsed:.1f} seconds")
    
    # Convert to numpy
    audio_array = audio_array.cpu().numpy().squeeze()
    
    # Convert to float32 if necessary
    if audio_array.dtype == 'float16':
        audio_array = audio_array.astype('float32')
    
    # Get sample rate and save
    sample_rate = model.generation_config.sample_rate
    output_path = OUTPUT_DIR / "speech.wav"
    write_wav(output_path, sample_rate, audio_array)
    
    duration = len(audio_array) / sample_rate
    print(f"✓ Speech generated successfully!")
    print(f"  - Duration: {duration:.2f} seconds")
    print(f"  - Sample rate: {sample_rate} Hz")
    print(f"  - Output file: {output_path.absolute()}")
    
    return output_path


def step2_extract_word_timestamps(audio_path):
    """
    Step 2: Use Whisper ASR to extract word-level timestamps.
    
    Args:
        audio_path: Path to the audio file to transcribe
    
    Returns:
        List of word dictionaries with 'word', 'start', 'end' keys
    """
    print("\n" + "="*70)
    print("STEP 2: SPEECH RECOGNITION WITH WORD TIMESTAMPS (Whisper)")
    print("="*70)
    
    print(f"Loading Whisper model: {ASR_MODEL}")
    model = whisper.load_model(ASR_MODEL)
    
    print(f"✓ Model loaded successfully")
    print(f"\nTranscribing audio file: {audio_path}")
    print("Extracting word-level timestamps...")
    start_time = time.time()
    
    # Load audio directly using scipy
    from scipy.io import wavfile
    sample_rate_wav, audio_data = wavfile.read(audio_path)
    
    # Convert to float32 and normalize
    if audio_data.dtype == 'int16':
        audio_data = audio_data.astype('float32') / 32768.0
    elif audio_data.dtype == 'int32':
        audio_data = audio_data.astype('float32') / 2147483648.0
    elif audio_data.dtype != 'float32':
        audio_data = audio_data.astype('float32')
    
    # Resample to 16kHz if necessary
    if sample_rate_wav != 16000:
        audio_tensor = torch.from_numpy(audio_data).float()
        audio_tensor = torch.nn.functional.interpolate(
            audio_tensor.unsqueeze(0).unsqueeze(0),
            size=int(len(audio_data) * 16000 / sample_rate_wav),
            mode='linear',
            align_corners=False
        ).squeeze()
        audio_data = audio_tensor.numpy()
    
    # Transcribe with progress bar
    result = model.transcribe(
        audio_data,
        word_timestamps=True,
        verbose=False,
        fp16=torch.cuda.is_available()  # Use FP16 on GPU for speed
    )
    
    elapsed = time.time() - start_time
    print(f"  ✓ Transcription completed in {elapsed:.1f} seconds")
    
    # Extract all words
    all_words = []
    for segment in result['segments']:
        if 'words' in segment:
            for word_info in segment['words']:
                all_words.append({
                    'word': word_info['word'],
                    'start': word_info['start'],
                    'end': word_info['end']
                })
    
    print(f"\n✓ Transcription completed!")
    print(f"  - Detected language: {result['language']}")
    print(f"  - Total segments: {len(result['segments'])}")
    print(f"  - Total words: {len(all_words)}")
    print(f"\nFull transcription:")
    print(f"  \"{result['text']}\"")
    
    return all_words


def step3_generate_subtitles(words):
    """
    Step 3: Generate SRT subtitle file from word timestamps.
    
    Args:
        words: List of word dictionaries with timing information
    
    Returns:
        Path to the generated SRT file
    """
    print("\n" + "="*70)
    print("STEP 3: SUBTITLE GENERATION")
    print("="*70)
    
    print(f"Grouping {len(words)} words into readable captions...")
    captions = group_words_into_captions(words)
    
    print(f"✓ Created {len(captions)} caption segments")
    
    output_path = OUTPUT_DIR / "subtitles.srt"
    generate_srt_file(captions, output_path)
    
    return output_path


# ============================================================================
# VIDEO GENERATION - OPTION 1: Use Existing Video
# ============================================================================

def option1_process_existing_video(audio_path, srt_path, resolution, fps):
    """
    Option 1: Replace audio in existing video and burn subtitles.
    
    Args:
        audio_path: Path to generated audio file
        srt_path: Path to SRT subtitle file
        resolution: Target resolution (e.g., "720p")
        fps: Target frame rate
    
    Returns:
        Path to output video file
    """
    print("\n" + "="*70)
    print("OPTION 1: PROCESSING EXISTING VIDEO")
    print("="*70)
    
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
        from moviepy.video.tools.subtitles import SubtitlesClip
        import moviepy  # Verify import works
        
        # Get input video path
        print("\nPlease provide the path to your existing video file:")
        input_video = input("Video path: ").strip().strip('"').strip("'")
        
        if not os.path.exists(input_video):
            print(f"❌ Error: Video file not found: {input_video}")
            return None
        
        print(f"\n✓ Loading video: {input_video}")
        video = VideoFileClip(input_video)
        
        # Load new audio
        print(f"✓ Loading generated audio: {audio_path}")
        new_audio = AudioFileClip(str(audio_path))
        
        # Replace audio
        print("✓ Replacing video audio with generated speech...")
        video_with_new_audio = video.set_audio(new_audio)
        
        # Resize if needed
        target_width, target_height = VIDEO_RESOLUTIONS[resolution]
        if video.size != (target_width, target_height):
            print(f"✓ Resizing video to {resolution} ({target_width}x{target_height})...")
            video_with_new_audio = video_with_new_audio.resize((target_width, target_height))
        
        # Burn subtitles
        print(f"✓ Burning subtitles from: {srt_path}")
        
        def make_text_clip(txt):
            return TextClip(
                txt,
                font='Arial',
                fontsize=36,
                color='white',
                stroke_color='black',
                stroke_width=2,
                method='caption',
                size=(target_width - 100, None)
            ).set_position(('center', 'bottom'))
        
        subtitles = SubtitlesClip(str(srt_path), make_text_clip)
        final_video = CompositeVideoClip([video_with_new_audio, subtitles])
        
        # Write output
        output_path = OUTPUT_DIR / f"final_video_{resolution}.mp4"
        print(f"\n✓ Rendering final video: {output_path}")
        print("  This may take several minutes...")
        
        final_video.write_videofile(
            str(output_path),
            fps=fps,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=str(OUTPUT_DIR / 'temp-audio.m4a'),
            remove_temp=True
        )
        
        # Close clips
        video.close()
        new_audio.close()
        final_video.close()
        
        print(f"\n✓ Video processing complete!")
        print(f"  Output: {output_path.absolute()}")
        
        return output_path
        
    except ImportError as e:
        print(f"❌ Error: MoviePy not installed. Please install it:")
        print(f"   pip install moviepy")
        return None
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# VIDEO GENERATION - OPTION 2: Generate AI Video from Text
# ============================================================================

def option2_generate_ai_video(text, audio_path, srt_path, emotion_profile, resolution, fps):
    """
    Option 2: Generate video from text using AI with emotion guidance.
    
    Args:
        text: Story text
        audio_path: Path to generated audio
        srt_path: Path to SRT subtitle file
        emotion_profile: Combined emotion description
        resolution: Target resolution
        fps: Target frame rate
    
    Returns:
        Path to output video file
    """
    print("\n" + "="*70)
    print("OPTION 2: GENERATING AI VIDEO FROM TEXT")
    print("="*70)
    
    try:
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline, StableVideoDiffusionPipeline
        from diffusers.utils import export_to_video
        from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
        import diffusers  # Verify import works
        
        # Create detailed, story-driven prompt for realistic video generation
        # Extract key story elements and create vivid visual description
        story_scenes = [
            "A young Romanian man in his 20s sitting at a computer desk looking confused",
            "Close-up of hands frantically clicking on a messy Windows computer screen",
            "The same man looking surprised as code appears on screen",
            "Friends gathering around impressed, pointing at the laptop",
            "The man nervously smiling while secretly Googling on his phone",
            "Stack Overflow website visible on screen in background",
            "The man confidently presenting to friends with a fake confident smile",
            "Documentary-style footage of an amateur programmer in a home office"
        ]
        
        # Create a rich, detailed prompt
        prompt = (
            f"Photorealistic documentary footage, cinematic 4K quality. "
            f"Scene: {story_scenes[0]}. "
            f"A real Romanian young man working at his computer, genuine expressions, "
            f"natural home office lighting, authentic setting with typical Romanian apartment decor. "
            f"Real human face with detailed features, natural skin texture, authentic emotions. "
            f"Professional cinematography, shallow depth of field, warm natural lighting. "
            f"Realistic hand movements, natural body language. "
            f"Style: {emotion_profile}. "
            f"Camera: Handheld documentary style, slight camera movement for realism. "
            f"No animations, no cartoons, no artificial elements. Pure photorealistic footage."
        )
        
        print(f"\nGenerating video with prompt:")
        print(f"  \"{prompt[:100]}...\"")
        
        # Step 1: Generate a high-quality initial image for better video quality
        print(f"\n[1/6] Generating high-quality initial image...")
        print("  💡 Using Stable Diffusion for photorealistic base image")
        
        try:
            from diffusers import StableDiffusionPipeline
            
            # Load Stable Diffusion for initial image
            sd_pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16
            )
            sd_pipe.to("cuda")
            
            # Generate high-quality initial frame
            initial_image = sd_pipe(
                prompt,
                num_inference_steps=50,  # High quality image
                guidance_scale=9.0,
                height=720,
                width=1280
            ).images[0]
            
            # Save initial image
            initial_image_path = OUTPUT_DIR / "initial_frame.png"
            initial_image.save(initial_image_path)
            print(f"  ✓ High-quality image generated: {initial_image_path}")
            
            # Clean up SD pipeline to free memory
            del sd_pipe
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ⚠ Could not generate initial image: {e}")
            print(f"  → Continuing with text-to-video only")
            initial_image = None
        
        # Load text-to-video or image-to-video model
        if USE_IMG2VID and initial_image:
            print(f"\n[2/6] Loading image-to-video model (HIGH QUALITY)")
            print("  Model: Stable Video Diffusion")
            print("  (This will download ~10GB on first run)")
        else:
            print(f"\n[2/6] Loading text-to-video model")
            print(f"  Model: {TEXT_TO_VIDEO_MODEL}")
            print("  (This will download ~7GB on first run)")
        
        start_time = time.time()
        
        if USE_IMG2VID and initial_image:
            # Use Stable Video Diffusion for MUCH better quality
            from diffusers import StableVideoDiffusionPipeline
            
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                TEXT_TO_VIDEO_MODEL,
                torch_dtype=torch.float16,
                variant="fp16"
            )
            pipe.to("cuda")
        else:
            # Fallback to text-to-video
            pipe = DiffusionPipeline.from_pretrained(
                TEXT_TO_VIDEO_MODEL,
                torch_dtype=torch.float16,
                variant="fp16"
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.enable_model_cpu_offload()
        
        load_time = time.time() - start_time
        print(f"  ✓ Model loaded in {load_time:.1f}s")
        
        # Generate video frames with ULTRA HIGH quality settings
        print(f"\n[3/6] Generating video frames (HIGH QUALITY MODE)...")
        print(f"  ⏱️  Estimated time: 10-20 minutes (worth it for quality!)")
        print(f"  💡 Using maximum quality settings for photorealistic output")
        
        frame_start = time.time()
        
        if USE_IMG2VID and initial_image:
            # Image-to-video: Much better quality!
            video_frames = pipe(
                initial_image,
                decode_chunk_size=8,
                num_frames=48,
                motion_bucket_id=127,  # More motion
                noise_aug_strength=0.02,  # Less noise
                num_inference_steps=30,  # Higher quality
            ).frames[0]
        else:
            # Text-to-video with enhanced settings
            video_frames = pipe(
                prompt,
                num_inference_steps=30,  # Increased for maximum quality
                num_frames=60,  # More frames for smoother video
                guidance_scale=12.0,  # Strong prompt adherence
                output_type="np"
            ).frames[0]
        
        frame_time = time.time() - start_time
        print(f"  ✓ Video frames generated in {frame_time/60:.1f} minutes")
        
        # Export to temporary video
        temp_video_path = OUTPUT_DIR / "temp_generated_video.mp4"
        print(f"\n[4/6] Exporting high-quality video frames...")
        export_start = time.time()
        export_to_video(video_frames, str(temp_video_path), fps=24)
        export_time = time.time() - export_start
        print(f"  ✓ Export completed in {export_time:.1f}s")
        
        # Load generated video and add audio
        print(f"\n[5/6] Adding audio...")
        video = VideoFileClip(str(temp_video_path))
        audio = AudioFileClip(str(audio_path))
        
        # Loop video if it's shorter than audio
        if video.duration < audio.duration:
            n_loops = int(np.ceil(audio.duration / video.duration))
            video = concatenate_videoclips([video] * n_loops)
            video = video.subclip(0, audio.duration)
        
        video_with_audio = video.set_audio(audio)
        
        # Resize to target resolution
        target_width, target_height = VIDEO_RESOLUTIONS[resolution]
        video_with_audio = video_with_audio.resize((target_width, target_height))
        
        # Note: Subtitles are saved as separate SRT file
        # To burn subtitles into video, use Option 1 with this video or use VLC/video player
        print(f"  ✓ Audio added successfully")
        print(f"  💡 Subtitles saved separately as SRT file (use video player to display)")
        
        # Write final output
        output_path = OUTPUT_DIR / f"ai_generated_video_{resolution}.mp4"
        print(f"\n[6/6] Rendering final HIGH QUALITY video: {output_path}")
        print(f"  ⏱️  Estimated time: 2-4 minutes")
        
        render_start = time.time()
        
        # Render with high bitrate for best quality
        video_with_audio.write_videofile(
            str(output_path),
            fps=fps,
            codec='libx264',
            audio_codec='aac',
            bitrate='8000k',  # High bitrate for quality
            preset='slow',  # Better compression/quality
            verbose=False,
            logger='bar'
        )
        
        render_time = time.time() - render_start
        print(f"  ✓ Rendering completed in {render_time:.1f}s")
        
        # Cleanup
        video.close()
        audio.close()
        video_with_audio.close()
        if temp_video_path.exists():
            temp_video_path.unlink()
        
        total_time = time.time() - start_time
        print(f"\n✓ AI video generation complete!")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Output: {output_path.absolute()}")
        
        return output_path
        
    except ImportError as e:
        print(f"❌ Error: Required libraries not installed: {e}")
        print(f"   pip install diffusers moviepy")
        print(f"\n   Then restart your terminal/script.")
        return None
    except Exception as e:
        print(f"❌ Error generating AI video: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function that runs the complete pipeline."""
    print("\n" + "="*70)
    print("TTS-TO-VIDEO PIPELINE")
    print("Text-to-Speech -> Subtitles -> Video Generation")
    print("="*70)
    
    pipeline_start = time.time()
    
    try:
        # Setup
        create_output_directory()
        
        # Show menu and get user choice
        video_mode = show_main_menu()
        
        # Get video settings if needed
        if video_mode in [1, 2]:
            resolution, fps = get_video_settings()
        else:
            resolution, fps = None, None
        
        # Step 1: Generate speech from text
        audio_path = step1_text_to_speech(STORY)
        
        # Step 2: Extract word-level timestamps
        words = step2_extract_word_timestamps(audio_path)
        
        # Step 3: Generate SRT subtitles
        srt_path = step3_generate_subtitles(words)
        
        # Emotion detection (for Option 2)
        if video_mode == 2:
            text_emotion = detect_emotion_from_text(STORY)
            audio_emotion = detect_emotion_from_audio(audio_path)
            emotion_profile = combine_emotions(text_emotion, audio_emotion)
        else:
            emotion_profile = None
        
        # Video generation based on mode
        if video_mode == 1:
            video_path = option1_process_existing_video(audio_path, srt_path, resolution, fps)
        elif video_mode == 2:
            video_path = option2_generate_ai_video(STORY, audio_path, srt_path, emotion_profile, resolution, fps)
        else:
            video_path = None
            print("\n✓ Skipped video generation (audio and subtitles only)")
        
        # Final summary
        total_pipeline_time = time.time() - pipeline_start
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n⏱️  Total pipeline time: {total_pipeline_time/60:.1f} minutes")
        print("\nGenerated files:")
        print(f"  1. Audio file:    {audio_path.absolute()}")
        print(f"  2. Subtitle file: {srt_path.absolute()}")
        if video_path:
            print(f"  3. Video file:    {video_path.absolute()}")
        print("\n" + "="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
