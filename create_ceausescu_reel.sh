#!/bin/bash
set -e

echo "🎬 Creating Nicolae Ceaușescu TikTok-Style Reel..."
echo "=================================================="

# Step 1: Wait for all 8 scenes to be generated
echo ""
echo "📹 Step 1: Checking video scenes..."
cd /home/ubuntu/Open-Sora/samples/ion_story/video_256px

# Count generated videos
VIDEO_COUNT=$(ls -1 *.mp4 2>/dev/null | wc -l)
echo "Found $VIDEO_COUNT video scenes"

if [ "$VIDEO_COUNT" -lt 8 ]; then
    echo "⚠️  Warning: Only $VIDEO_COUNT/8 scenes generated. Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# Step 2: Generate audio narration
echo ""
echo "🎤 Step 2: Generating audio narration..."
cd /home/ubuntu/test
source venv/bin/activate

cat > generate_ceausescu_audio.py << 'PYTHON'
import torch
from transformers import AutoProcessor, BarkModel
import scipy.io.wavfile as wavfile
import numpy as np

# Narration script for Nicolae Ceaușescu
text = """Nicolae Ceaușescu. The name that defined an era in Romanian history.

From his desk in the Palace of Parliament, he ruled with absolute power.

His speeches captivated millions. His decisions shaped a nation.

Through the grand halls of power, he walked as an unchallenged leader.

Communist Party meetings. Where his word became law.

But in December 1989, everything changed. The people rose up in Revolution Square.

His final speech from the balcony. A moment that would end his reign forever.

Today, the Palace of Parliament stands as a monument. A reminder of Romania's complex past.

This is the story of Nicolae Ceaușescu. Power, revolution, and the price of absolute control."""

print("🎤 Generating professional narration...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small").to(device)

# Use a more serious, documentary-style voice
inputs = processor(text, voice_preset="v2/en_speaker_9").to(device)

print("Generating audio (this may take 2-3 minutes)...")
with torch.no_grad():
    audio_output = model.generate(**inputs)

audio_array = audio_output.cpu().numpy().squeeze()

# Save audio
wavfile.write('ceausescu_narration.wav', model.generation_config.sample_rate, audio_array)
duration = len(audio_array) / model.generation_config.sample_rate
print(f"✅ Audio generated: {duration:.2f} seconds")
print(f"✅ Saved to: ceausescu_narration.wav")
PYTHON

python generate_ceausescu_audio.py

# Step 3: Create captions file (SRT format)
echo ""
echo "📝 Step 3: Creating captions..."

cat > ceausescu_captions.srt << 'SRT'
1
00:00:00,000 --> 00:00:03,000
Nicolae Ceaușescu
The name that defined an era

2
00:00:03,000 --> 00:00:07,000
From the Palace of Parliament
He ruled with absolute power

3
00:00:07,000 --> 00:00:11,000
His speeches captivated millions
His decisions shaped a nation

4
00:00:11,000 --> 00:00:15,000
Through the grand halls of power
An unchallenged leader

5
00:00:15,000 --> 00:00:19,000
Communist Party meetings
Where his word became law

6
00:00:19,000 --> 00:00:23,000
December 1989
The people rose up in Revolution Square

7
00:00:23,000 --> 00:00:27,000
His final speech from the balcony
A moment that ended his reign

8
00:00:27,000 --> 00:00:31,000
The Palace of Parliament stands today
A monument to Romania's past

9
00:00:31,000 --> 00:00:35,000
Nicolae Ceaușescu
Power, revolution, and the price of control
SRT

echo "✅ Captions created: ceausescu_captions.srt"

# Step 4: Install ffmpeg if needed
echo ""
echo "🔧 Step 4: Checking ffmpeg installation..."
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing ffmpeg..."
    apt-get update && apt-get install -y ffmpeg
fi

# Step 5: Concatenate videos
echo ""
echo "🎞️  Step 5: Concatenating video scenes..."
cd /home/ubuntu/Open-Sora/samples/ion_story/video_256px

# Create concat list
ls -1 *.mp4 | sort | awk '{print "file '\''" $0 "'\''"}' > concat_list.txt
echo "Video files to concatenate:"
cat concat_list.txt

# Concatenate videos
ffmpeg -f concat -safe 0 -i concat_list.txt -c copy combined_raw.mp4 -y

echo "✅ Videos concatenated"

# Step 6: Upscale to 1080p (TikTok quality)
echo ""
echo "📺 Step 6: Upscaling to 1080p for TikTok..."

ffmpeg -i combined_raw.mp4 \
  -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1" \
  -c:v libx264 -preset slow -crf 18 \
  upscaled_1080p.mp4 -y

echo "✅ Video upscaled to 1080x1920 (TikTok format)"

# Step 7: Add audio
echo ""
echo "🎵 Step 7: Adding audio narration..."

ffmpeg -i upscaled_1080p.mp4 \
  -i /home/ubuntu/test/ceausescu_narration.wav \
  -c:v copy -c:a aac -b:a 192k \
  -shortest \
  with_audio.mp4 -y

echo "✅ Audio added"

# Step 8: Add TikTok-style captions
echo ""
echo "💬 Step 8: Adding TikTok-style captions..."

ffmpeg -i with_audio.mp4 \
  -vf "subtitles=/home/ubuntu/test/ceausescu_captions.srt:force_style='FontName=Arial Black,FontSize=24,Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=3,Shadow=2,Alignment=2,MarginV=50'" \
  -c:a copy \
  /home/ubuntu/ceausescu_tiktok_reel.mp4 -y

echo ""
echo "=================================================="
echo "✅ TIKTOK REEL CREATED SUCCESSFULLY!"
echo "=================================================="
echo ""
echo "📍 Location: /home/ubuntu/ceausescu_tiktok_reel.mp4"
echo ""

# Get video info
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 /home/ubuntu/ceausescu_tiktok_reel.mp4 | awk '{print "⏱️  Duration: " int($1) " seconds"}'
ls -lh /home/ubuntu/ceausescu_tiktok_reel.mp4 | awk '{print "💾 File size: " $5}'
echo ""
echo "🎬 Format: 1080x1920 (TikTok vertical)"
echo "🎤 Audio: Professional narration"
echo "💬 Captions: TikTok-style subtitles"
echo "✨ Quality: High (CRF 18)"
echo ""
echo "Ready to upload to TikTok, Instagram Reels, or YouTube Shorts! 🚀"

