"""
Quick test script to verify all video generation dependencies are working.
Run this before using generate_subtitles.py to ensure everything is installed correctly.
"""

print("\n" + "="*70)
print("VIDEO GENERATION DEPENDENCIES CHECK")
print("="*70 + "\n")

# Test 1: Core packages
print("Testing core packages...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    print(f"    - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    - GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"  ✗ PyTorch: {e}")

# Test 2: Transformers
print("\nTesting Transformers...")
try:
    from transformers import AutoProcessor, BarkModel
    print("  ✓ Transformers (Bark TTS)")
except ImportError as e:
    print(f"  ✗ Transformers: {e}")

# Test 3: Whisper
print("\nTesting Whisper...")
try:
    import whisper
    print("  ✓ Whisper (ASR)")
except ImportError as e:
    print(f"  ✗ Whisper: {e}")

# Test 4: MoviePy
print("\nTesting MoviePy...")
try:
    from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
    from moviepy.video.tools.subtitles import SubtitlesClip
    import moviepy
    print(f"  ✓ MoviePy {moviepy.__version__}")
    
    # Check Pillow compatibility
    from PIL import Image
    has_antialias = hasattr(Image, 'ANTIALIAS')
    print(f"    - Pillow {Image.__version__}")
    print(f"    - ANTIALIAS available: {has_antialias}")
    if not has_antialias:
        print("    ⚠ WARNING: ANTIALIAS not available - video resize will fail!")
        print("      Fix: pip install pillow==9.5.0")
except ImportError as e:
    print(f"  ✗ MoviePy: {e}")

# Test 5: Diffusers
print("\nTesting Diffusers...")
try:
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    from diffusers.utils import export_to_video
    import diffusers
    print(f"  ✓ Diffusers {diffusers.__version__}")
except ImportError as e:
    print(f"  ✗ Diffusers: {e}")

# Test 6: Audio analysis (Librosa)
print("\nTesting Librosa...")
try:
    import librosa
    import soundfile
    print(f"  ✓ Librosa {librosa.__version__}")
except ImportError as e:
    print(f"  ✗ Librosa: {e}")

# Test 7: Text sentiment (TextBlob)
print("\nTesting TextBlob...")
try:
    from textblob import TextBlob
    # Test sentiment analysis
    blob = TextBlob("This is a great test!")
    sentiment = blob.sentiment.polarity
    print(f"  ✓ TextBlob")
    print(f"    - Sentiment analysis working: {sentiment > 0}")
except ImportError as e:
    print(f"  ✗ TextBlob: {e}")
except Exception as e:
    print(f"  ⚠ TextBlob installed but corpora missing")
    print(f"    Fix: python -m textblob.download_corpora")

# Test 8: Other dependencies
print("\nTesting other dependencies...")
try:
    import scipy
    import numpy as np
    print(f"  ✓ SciPy {scipy.__version__}")
    print(f"  ✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"  ✗ SciPy/NumPy: {e}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\n✅ If all tests passed, you're ready to generate videos!")
print("\nRun: python generate_subtitles.py")
print("\nOptions:")
print("  [1] Use existing video (replace audio + burn subtitles)")
print("  [2] Generate AI video from text (emotion-guided)")
print("  [0] Audio + subtitles only\n")
print("="*70 + "\n")

