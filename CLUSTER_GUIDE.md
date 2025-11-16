# 🚀 Multi-GPU Cluster Video Generation System

Complete guide for generating high-quality AI videos across multiple GPUs with state-of-the-art models.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Available Models](#available-models)
3. [Usage Examples](#usage-examples)
4. [Model Comparison](#model-comparison)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

---

## 🎯 Quick Start

### Installation

```bash
# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate bitsandbytes torchao
pip install moviepy librosa soundfile
```

### Basic Usage (9 GPUs)

```bash
# Generate 9 videos on 9 GPUs with CogVideoX
python cluster_launcher.py --num-gpus 9 --model cogvideox

# Use your own prompts
python cluster_launcher.py --num-gpus 9 --model hunyuan --prompts my_prompts.txt

# Compare all models side-by-side
python cluster_launcher.py --compare-models --num-gpus 5
```

---

## 🎨 Available Models

### Open-Sora 1.3 (70B) - **NEW!** 🔥
- **Quality**: ⭐⭐⭐⭐⭐⭐ OUTSTANDING (True Sora-level)
- **Speed**: 🐌🐌 Very Slow (~30-60 min per video)
- **VRAM**: 80GB (H200 SXM required)
- **Best for**: Ultimate quality, production-grade videos

```bash
python cluster_launcher.py --num-gpus 8 --model opensora
```

**⚠️ Requires H200 SXM GPUs with 80GB VRAM**  
See `OPENSORA_H200_SETUP.md` for installation guide.

### CogVideoX-5B (THUDM)
- **Quality**: ⭐⭐⭐⭐⭐ Excellent (Sora-like)
- **Speed**: 🐌 Slow (~15-25 min per video)
- **VRAM**: 24GB+
- **Best for**: Highest quality, photorealistic videos

```bash
python cluster_launcher.py --num-gpus 9 --model cogvideox
```

### HunyuanVideo (Tencent)
- **Quality**: ⭐⭐⭐⭐⭐ Excellent (State-of-the-art)
- **Speed**: 🐌 Slow (~15-20 min per video)
- **VRAM**: 24GB+
- **Best for**: Professional-grade video generation

```bash
python cluster_launcher.py --num-gpus 9 --model hunyuan
```

### Mochi-1 (Genmo)
- **Quality**: ⭐⭐⭐⭐ Very Good
- **Speed**: 🚀 Medium (~8-12 min per video)
- **VRAM**: 16GB+
- **Best for**: Balance of quality and speed

```bash
python cluster_launcher.py --num-gpus 9 --model mochi
```

### LTX-Video (Lightricks)
- **Quality**: ⭐⭐⭐ Good
- **Speed**: ⚡ Fast (~5-8 min per video)
- **VRAM**: 12GB+
- **Best for**: Quick iterations, testing

```bash
python cluster_launcher.py --num-gpus 9 --model ltx
```

### Stable Video Diffusion (Stability AI)
- **Quality**: ⭐⭐⭐⭐ Very Good
- **Speed**: 🚀 Medium (~10-15 min per video)
- **VRAM**: 16GB+
- **Best for**: Proven quality, image-to-video

```bash
python cluster_launcher.py --num-gpus 9 --model svd
```

---

## 💡 Usage Examples

### Example 1: Generate 9 Videos on 9 GPUs

```bash
python cluster_launcher.py \
  --num-gpus 9 \
  --model cogvideox \
  --prompts prompts.txt
```

**What happens:**
- Loads 9 prompts from `prompts.txt`
- Assigns one video to each GPU
- Generates all 9 videos in parallel
- Saves to `cluster_output/`

**Expected time:** ~15-25 minutes (wall-clock) for CogVideoX

---

### Example 2: Use Specific GPUs Only

```bash
python cluster_launcher.py \
  --gpu-ids 0 2 4 6 8 \
  --model hunyuan \
  --prompts my_prompts.txt
```

**What happens:**
- Uses only GPUs 0, 2, 4, 6, 8
- Skips GPUs 1, 3, 5, 7
- Useful if some GPUs are busy

---

### Example 3: Single Prompt, Multiple Videos

```bash
python cluster_launcher.py \
  --num-gpus 9 \
  --model svd \
  --prompt "A cat playing with a ball of yarn in slow motion"
```

**What happens:**
- Uses the same prompt for all 9 videos
- Each video will be slightly different (random seed)
- Good for exploring variations

---

### Example 4: Custom Generation Parameters

```bash
python cluster_launcher.py \
  --num-gpus 4 \
  --model cogvideox \
  --num-frames 81 \
  --fps 16 \
  --steps 60 \
  --prompts prompts.txt
```

**What happens:**
- Generates 81 frames per video (longer)
- 16 FPS output
- 60 inference steps (higher quality)

---

## 🔬 Model Comparison

### Compare All Models with Same Prompt

```bash
python cluster_launcher.py \
  --compare-models \
  --num-gpus 5 \
  --prompt "Cinematic footage of ocean waves at sunset"
```

**What happens:**
- Generates the same prompt with 5 different models
- Each model runs on a separate GPU
- Creates comparison directory with all videos
- Saves timing and quality metrics

**Output:**
```
cluster_output/comparison_20250116_143022/
├── video_cogvideox_gpu0_id0_20250116_143022.mp4
├── video_hunyuan_gpu1_id1_20250116_143025.mp4
├── video_mochi_gpu2_id2_20250116_143028.mp4
├── video_ltx_gpu3_id3_20250116_143030.mp4
├── video_svd_gpu4_id4_20250116_143032.mp4
└── comparison_results.json
```

### Analyze Comparison Results

```bash
cat cluster_output/comparison_*/comparison_results.json
```

Shows:
- Generation time for each model
- Success/failure status
- GPU utilization
- Prompt used

---

## ⚙️ Configuration

### Custom Prompts File

Create `my_prompts.txt`:

```
A photorealistic scene of a programmer working at night
Cinematic drone footage of a mountain landscape
Close-up of raindrops falling on a window
Time-lapse of clouds moving across the sky
```

Then run:

```bash
python cluster_launcher.py --num-gpus 4 --model cogvideox --prompts my_prompts.txt
```

---

### Adjust GPU Count Based on Cluster Size

**For 4 GPUs:**
```bash
python cluster_launcher.py --num-gpus 4 --model hunyuan
```

**For 8 GPUs:**
```bash
python cluster_launcher.py --num-gpus 8 --model cogvideox
```

**For 16 GPUs:**
```bash
python cluster_launcher.py --num-gpus 16 --model svd --prompts large_batch.txt
```

---

### Model-Specific Defaults

Each model has optimized defaults:

| Model      | Frames | FPS | Steps | Notes                    |
|------------|--------|-----|-------|--------------------------|
| cogvideox  | 49     | 8   | 50    | High quality, slow       |
| hunyuan    | 61     | 15  | 30    | Best quality             |
| mochi      | 84     | 30  | 64    | Smooth motion            |
| ltx        | 121    | 25  | 50    | Fast generation          |
| svd        | 25     | 7   | 25    | Requires initial image   |

Override with `--num-frames`, `--fps`, `--steps`.

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1:** Use fewer GPUs or smaller models

```bash
python cluster_launcher.py --num-gpus 4 --model ltx  # Lighter model
```

**Solution 2:** Reduce frames/steps

```bash
python cluster_launcher.py --num-gpus 9 --model cogvideox --num-frames 25 --steps 30
```

**Solution 3:** Use specific GPUs with more VRAM

```bash
python cluster_launcher.py --gpu-ids 0 1 2  # Only A40s, skip smaller GPUs
```

---

### Issue: "device >= 0 && device < num_gpus" Error

**Solution:** Pin to specific GPU before running

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
python cluster_launcher.py --num-gpus 9 --model cogvideox
```

---

### Issue: Slow Generation

**Check GPU utilization:**

```bash
watch -n 1 nvidia-smi
```

**Use faster model:**

```bash
python cluster_launcher.py --num-gpus 9 --model ltx  # Fastest
```

**Reduce quality settings:**

```bash
python cluster_launcher.py --num-gpus 9 --model cogvideox --steps 25 --num-frames 25
```

---

### Issue: Model Download Fails

**Solution:** Pre-download models

```python
from diffusers import CogVideoXPipeline
pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b")
```

Or set cache directory:

```bash
export HF_HOME=/path/to/large/disk
python cluster_launcher.py --num-gpus 9 --model cogvideox
```

---

## 📊 Performance Benchmarks

### Single Video Generation Time (A40 GPU)

| Model      | Time (min) | Quality | VRAM (GB) |
|------------|------------|---------|-----------|
| CogVideoX  | 15-25      | ⭐⭐⭐⭐⭐   | 24        |
| HunyuanVideo | 15-20    | ⭐⭐⭐⭐⭐   | 24        |
| Mochi-1    | 8-12       | ⭐⭐⭐⭐    | 16        |
| LTX-Video  | 5-8        | ⭐⭐⭐     | 12        |
| SVD        | 10-15      | ⭐⭐⭐⭐    | 16        |

### Cluster Throughput (9 GPUs)

| Model      | Videos/Hour | Total GPU-Hours |
|------------|-------------|-----------------|
| CogVideoX  | ~27         | 3.75            |
| HunyuanVideo | ~30       | 3.0             |
| Mochi-1    | ~54         | 1.5             |
| LTX-Video  | ~81         | 1.0             |
| SVD        | ~36         | 2.25            |

---

## 🎓 Advanced Usage

### Sequential vs Parallel

**Parallel (recommended):**
```bash
python cluster_launcher.py --num-gpus 9 --model cogvideox
# 9 videos in ~20 minutes
```

**Sequential (for comparison):**
```bash
for i in {0..8}; do
  python video_worker.py --model cogvideox --gpu-id 0 --video-id $i --prompt "..."
done
# 9 videos in ~180 minutes (3 hours)
```

**Speedup: 9x faster with parallel!**

---

### Mixed Model Generation

Generate different videos with different models:

```bash
# Terminal 1: CogVideoX on GPUs 0-2
python cluster_launcher.py --gpu-ids 0 1 2 --model cogvideox --prompts batch1.txt &

# Terminal 2: Hunyuan on GPUs 3-5
python cluster_launcher.py --gpu-ids 3 4 5 --model hunyuan --prompts batch2.txt &

# Terminal 3: LTX on GPUs 6-8
python cluster_launcher.py --gpu-ids 6 7 8 --model ltx --prompts batch3.txt &

wait
```

---

## 📁 Output Structure

```
cluster_output/
├── video_cogvideox_gpu0_id0_20250116_143022.mp4
├── video_cogvideox_gpu1_id1_20250116_143025.mp4
├── ...
├── generation_results.json
└── comparison_20250116_143022/
    ├── video_cogvideox_gpu0_id0_*.mp4
    ├── video_hunyuan_gpu1_id1_*.mp4
    ├── ...
    └── comparison_results.json
```

---

## 🔗 Integration with Original Pipeline

### Use cluster output with original TTS pipeline

```bash
# 1. Generate videos with cluster
python cluster_launcher.py --num-gpus 9 --model cogvideox

# 2. Add audio/subtitles with original pipeline
python generate_subtitles.py
# Choose Option 1 (Use Existing Video)
# Point to: cluster_output/video_cogvideox_gpu0_id0_*.mp4
```

---

## 📝 Tips & Best Practices

1. **Start small:** Test with 1-2 GPUs first
2. **Monitor VRAM:** Use `nvidia-smi` to watch memory
3. **Use prompts.txt:** Easier to manage many prompts
4. **Compare models:** Run `--compare-models` first to find best model
5. **Save results:** Check `generation_results.json` for metrics
6. **Backup outputs:** Videos are large, plan storage accordingly

---

## 🆘 Support

**List available models:**
```bash
python cluster_launcher.py --list-models
```

**Check GPU status:**
```bash
nvidia-smi
```

**View generation results:**
```bash
cat cluster_output/generation_results.json | python -m json.tool
```

---

**Made for high-performance AI video generation on multi-GPU clusters** 🚀

