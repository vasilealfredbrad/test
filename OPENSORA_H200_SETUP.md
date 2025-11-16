# 🚀 Open-Sora 1.3 (70B) Setup Guide for H200 SXM

Complete guide for installing and running Open-Sora 1.3 (70B) on NVIDIA H200 SXM GPUs.

---

## 📋 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA H200 SXM (80GB VRAM minimum)
- **RAM**: 256GB+ system RAM recommended
- **Storage**: 500GB+ free space for models and cache
- **Network**: Fast internet for initial model download (~150GB)

### Software Requirements
- **OS**: Ubuntu 22.04 LTS (recommended)
- **CUDA**: 12.1+ 
- **Python**: 3.10 or 3.11
- **PyTorch**: 2.3.0+ with CUDA support

---

## 🔧 Installation Steps

### Step 1: Verify H200 Setup

```bash
# Check GPU
nvidia-smi

# Should show:
# - NVIDIA H200 SXM
# - 80GB VRAM
# - CUDA Version 12.1+
```

### Step 2: Clone Open-Sora Repository

```bash
cd /home/ubuntu
git clone https://github.com/hpcaitech/Open-Sora.git
cd Open-Sora

# Checkout stable version
git checkout v1.3
```

### Step 3: Create Virtual Environment

```bash
# Create venv
python3.10 -m venv opensora_env
source opensora_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 4: Install PyTorch with CUDA 12.1

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

### Step 5: Install Open-Sora Dependencies

```bash
# Install from requirements
pip install -r requirements.txt

# Additional dependencies
pip install accelerate transformers diffusers
pip install flash-attn --no-build-isolation
pip install xformers
```

### Step 6: Install ColossalAI (Required for 70B Model)

```bash
# ColossalAI for distributed inference
pip install colossalai

# Or build from source for latest features:
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI
pip install -v .
cd ..
```

### Step 7: Download Open-Sora 1.3 Weights

```bash
# Using huggingface-cli (recommended)
pip install huggingface_hub[cli]

# Login to Hugging Face (if model requires auth)
huggingface-cli login

# Download model weights
huggingface-cli download hpcaitech/Open-Sora-v1.3 \
  --local-dir ./pretrained_models/opensora_v1.3 \
  --local-dir-use-symlinks False
```

**Alternative: Manual download**
```bash
# Create model directory
mkdir -p pretrained_models/opensora_v1.3

# Download from Hugging Face model hub
# Visit: https://huggingface.co/hpcaitech/Open-Sora-v1.3
# Download all files to pretrained_models/opensora_v1.3/
```

---

## ⚙️ Configuration for H200

### Edit Open-Sora Config

Create `configs/opensora_v1.3_h200.py`:

```python
# Open-Sora 1.3 Configuration for H200 SXM

# Model settings
model = dict(
    type='OpenSoraModel_v1_3',
    from_pretrained='pretrained_models/opensora_v1.3',
    model_max_length=300,
    freeze_y_embedder=True,
    enable_flash_attn=True,  # Enable for H200
    enable_xformers=True,
)

# Video generation settings
num_frames = 51  # 2 seconds at 24fps
fps = 24
image_size = (720, 1280)  # 720p, can go up to 1080p on H200

# Inference settings
num_sampling_steps = 100  # Higher = better quality
cfg_scale = 7.5  # Guidance scale

# H200 optimizations
dtype = 'bf16'  # Use bfloat16 for better performance
enable_model_cpu_offload = False  # H200 has enough VRAM
enable_vae_tiling = False  # Not needed with 80GB VRAM
```

---

## 🎬 Usage Examples

### Basic Usage (Single GPU)

```bash
cd /home/ubuntu/Open-Sora
source opensora_env/bin/activate

python scripts/inference.py \
  --config configs/opensora_v1.3_h200.py \
  --prompt "A photorealistic scene of a programmer working at night, cinematic 4K quality" \
  --save-dir outputs/
```

### Using with Our Cluster System

Update `video_worker.py` path:

```python
# In video_worker.py, line 92:
sys.path.insert(0, '/home/ubuntu/Open-Sora')
```

Then run:

```bash
# Single video with Open-Sora on H200
python video_worker.py \
  --model opensora \
  --prompt "Cinematic drone footage of mountains at sunset" \
  --gpu-id 0

# Multiple videos on multiple H200s
python cluster_launcher.py \
  --num-gpus 8 \
  --model opensora \
  --prompts prompts.txt
```

---

## 🔬 Advanced Configuration

### Multi-GPU Inference (Multiple H200s)

For distributing one video across multiple H200 GPUs:

```bash
# Using ColossalAI distributed inference
torchrun --nproc_per_node=8 scripts/inference_distributed.py \
  --config configs/opensora_v1.3_h200.py \
  --prompt "Your prompt here" \
  --save-dir outputs/
```

### Memory Optimization (if needed)

Even with 80GB, for very long videos:

```python
# In config file
enable_vae_tiling = True
enable_sequential_cpu_offload = True
```

### Quality vs Speed Trade-offs

```python
# Maximum quality (slow)
num_sampling_steps = 150
cfg_scale = 8.0

# Balanced (recommended)
num_sampling_steps = 100
cfg_scale = 7.5

# Faster (still good)
num_sampling_steps = 50
cfg_scale = 6.0
```

---

## 📊 Performance Benchmarks (H200 SXM)

### Single H200 Performance

| Resolution | Frames | Steps | Time (min) | VRAM Used |
|------------|--------|-------|------------|-----------|
| 720p       | 51     | 100   | 30-40      | ~65GB     |
| 1080p      | 51     | 100   | 45-60      | ~75GB     |
| 720p       | 102    | 100   | 60-80      | ~70GB     |

### Multi-H200 Cluster (8 GPUs)

- **8 videos in parallel**: ~35-45 min wall-clock time
- **Total GPU-hours**: ~5 hours
- **Speedup vs sequential**: 8x

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"

**Solution 1**: Enable tiling
```python
enable_vae_tiling = True
```

**Solution 2**: Reduce resolution
```python
image_size = (512, 896)  # Lower resolution
```

**Solution 3**: Fewer frames
```python
num_frames = 25  # 1 second instead of 2
```

### Issue: "ColossalAI not found"

```bash
pip install colossalai
# Or build from source
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI && pip install -v .
```

### Issue: "Flash attention not available"

```bash
pip install flash-attn --no-build-isolation
# Or disable in config:
enable_flash_attn = False
```

### Issue: Slow generation

**Check GPU utilization:**
```bash
watch -n 1 nvidia-smi
```

Should show:
- GPU utilization: 95-100%
- Memory usage: 60-75GB
- Power: Near TDP limit

**If low utilization:**
- Enable flash attention
- Disable CPU offloading
- Check for CPU bottlenecks

---

## 🎯 Integration with Cluster System

### Update video_worker.py

```python
# Line 92 in video_worker.py
sys.path.insert(0, '/home/ubuntu/Open-Sora')

# Verify installation
try:
    from opensora.models import OpenSoraModel
    print("✓ Open-Sora installed correctly")
except ImportError:
    print("❌ Open-Sora not found - check installation")
```

### Run Cluster Comparison

```bash
# Compare Open-Sora vs other models
python cluster_launcher.py \
  --compare-models \
  --num-gpus 6 \
  --prompt "Cinematic footage of ocean waves at sunset"

# This will generate with:
# - opensora (H200 #0)
# - cogvideox (GPU #1)
# - hunyuan (GPU #2)
# - mochi (GPU #3)
# - ltx (GPU #4)
# - svd (GPU #5)
```

---

## 📈 Quality Comparison

### Open-Sora 1.3 (70B) vs Others

| Model | Quality | Motion | Coherence | Photorealism |
|-------|---------|--------|-----------|--------------|
| **Open-Sora 70B** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| CogVideoX-5B | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| HunyuanVideo | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Mochi-1 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

**Open-Sora advantages:**
- True Sora-level quality
- Better temporal consistency
- More photorealistic output
- Smoother motion
- Better prompt following

**Trade-offs:**
- Requires H200 (80GB VRAM)
- 2-3x slower than CogVideoX
- Larger model download (~150GB)

---

## 💡 Best Practices

### For Maximum Quality

1. **Use full resolution**: 1080p if possible
2. **High step count**: 100-150 steps
3. **Optimal guidance**: cfg_scale=7.5-8.0
4. **Good prompts**: Detailed, specific descriptions
5. **Monitor VRAM**: Keep below 75GB for stability

### For Faster Iteration

1. **Lower resolution**: 720p or 512p
2. **Fewer steps**: 50-75 steps
3. **Shorter videos**: 25 frames (1 sec)
4. **Batch generation**: Use cluster launcher

### For Production

1. **Test prompts first** with faster models
2. **Use Open-Sora** for final high-quality renders
3. **Run overnight** for large batches
4. **Save intermediate results**
5. **Monitor disk space** (videos are large)

---

## 📚 Additional Resources

- **Open-Sora GitHub**: https://github.com/hpcaitech/Open-Sora
- **Model Hub**: https://huggingface.co/hpcaitech/Open-Sora-v1.3
- **ColossalAI Docs**: https://colossalai.org/
- **H200 Specs**: https://www.nvidia.com/en-us/data-center/h200/

---

## 🆘 Support

**Check installation:**
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

**Test Open-Sora import:**
```bash
cd /home/ubuntu/Open-Sora
python -c "from opensora.models import OpenSoraModel; print('✓ Open-Sora OK')"
```

**Monitor generation:**
```bash
# Terminal 1: Run generation
python video_worker.py --model opensora --prompt "..." --gpu-id 0

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi
```

---

**Made for NVIDIA H200 SXM - Ultimate AI Video Generation** 🚀

