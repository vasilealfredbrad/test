"""
Multi-GPU Cluster Video Generation Launcher

Launches multiple video generation workers in parallel across available GPUs.
Supports configurable GPU count and multiple video generation models.

Usage:
    python cluster_launcher.py --num-gpus 9 --model cogvideox --prompts prompts.txt
    python cluster_launcher.py --num-gpus 4 --model hunyuan --prompt "A single prompt"
    python cluster_launcher.py --compare-models --num-gpus 5
"""

import os
import sys
import argparse
import subprocess
import time
import torch
import json
from pathlib import Path
from datetime import datetime
import multiprocessing as mp

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_PROMPTS = [
    "A photorealistic documentary scene of a young programmer working at his computer, natural home office lighting, authentic setting, cinematic 4K quality",
    "Cinematic footage of ocean waves crashing on a rocky shore at sunset, golden hour lighting, slow motion, 4K quality",
    "A fluffy teddy bear sits on a bed of soft pillows surrounded by children's toys, warm cozy lighting, shallow depth of field",
    "Time-lapse of a bustling city street at night, car lights trailing, neon signs glowing, urban atmosphere, 4K cinematic",
    "Close-up of hands crafting pottery on a spinning wheel, clay texture visible, artistic lighting, documentary style",
    "Aerial drone footage flying through a misty forest at dawn, sunbeams through trees, cinematic nature documentary",
    "A cat playing with a ball of yarn in slow motion, natural window lighting, shallow depth of field, 4K quality",
    "Underwater footage of colorful tropical fish swimming around coral reef, clear blue water, nature documentary style",
    "Steam rising from a hot cup of coffee on a wooden table, morning sunlight, cozy atmosphere, macro photography style"
]

AVAILABLE_MODELS = {
    "cogvideox": "CogVideoX-5B (High quality, Sora-like)",
    "hunyuan": "HunyuanVideo (State-of-the-art)",
    "mochi": "Mochi-1 (Fast, high quality)",
    "ltx": "LTX-Video (Fast, lightweight)",
    "svd": "Stable Video Diffusion (Proven quality)"
}

OUTPUT_DIR = Path("cluster_output")
RESULTS_FILE = OUTPUT_DIR / "generation_results.json"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_available_gpus():
    """Get list of available GPU IDs."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def load_prompts_from_file(filepath):
    """Load prompts from a text file (one per line)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def run_worker(gpu_id, video_id, model, prompt, output_dir, num_frames=None, fps=None, steps=None):
    """Run a single video generation worker on a specific GPU."""
    
    # Set environment variable to restrict to this GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Build command
    cmd = [
        sys.executable,  # python
        "video_worker.py",
        "--model", model,
        "--prompt", prompt,
        "--gpu-id", str(gpu_id),
        "--video-id", str(video_id)
    ]
    
    if num_frames:
        cmd.extend(["--num-frames", str(num_frames)])
    if fps:
        cmd.extend(["--fps", str(fps)])
    if steps:
        cmd.extend(["--steps", str(steps)])
    
    # Run worker
    print(f"[GPU {gpu_id}] Starting worker for video {video_id} with model {model}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        
        print(f"[GPU {gpu_id}] ✓ Video {video_id} completed in {elapsed/60:.1f} minutes")
        
        return {
            "gpu_id": gpu_id,
            "video_id": video_id,
            "model": model,
            "prompt": prompt,
            "status": "success",
            "time_minutes": elapsed / 60,
            "stdout": result.stdout[-500:] if result.stdout else "",  # Last 500 chars
        }
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"[GPU {gpu_id}] ❌ Video {video_id} failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e.stderr[-500:]}")
        
        return {
            "gpu_id": gpu_id,
            "video_id": video_id,
            "model": model,
            "prompt": prompt,
            "status": "failed",
            "time_minutes": elapsed / 60,
            "error": e.stderr[-500:] if e.stderr else str(e)
        }


def parallel_generation(gpu_ids, prompts, model, num_frames=None, fps=None, steps=None):
    """Generate multiple videos in parallel across GPUs."""
    
    print("\n" + "="*70)
    print("PARALLEL VIDEO GENERATION")
    print("="*70)
    print(f"GPUs: {len(gpu_ids)} ({gpu_ids})")
    print(f"Videos: {len(prompts)}")
    print(f"Model: {model}")
    print("="*70 + "\n")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Prepare tasks
    tasks = []
    for video_id, prompt in enumerate(prompts):
        gpu_id = gpu_ids[video_id % len(gpu_ids)]  # Round-robin GPU assignment
        tasks.append((gpu_id, video_id, model, prompt, OUTPUT_DIR, num_frames, fps, steps))
    
    # Run in parallel using multiprocessing
    with mp.Pool(processes=len(gpu_ids)) as pool:
        results = pool.starmap(run_worker, tasks)
    
    # Save results
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "num_gpus": len(gpu_ids),
        "num_videos": len(prompts),
        "results": results
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    total_time = sum(r["time_minutes"] for r in results)
    avg_time = total_time / len(results) if results else 0
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE!")
    print("="*70)
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    print(f"Average time per video: {avg_time:.1f} minutes")
    print(f"Total GPU-time: {total_time:.1f} minutes")
    print(f"Wall-clock time: ~{total_time/len(gpu_ids):.1f} minutes (parallel)")
    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Videos saved to: {OUTPUT_DIR}/")
    print("="*70 + "\n")
    
    return results


def compare_models(gpu_ids, prompt, models=None):
    """Generate the same prompt with multiple models for comparison."""
    
    if models is None:
        models = list(AVAILABLE_MODELS.keys())
    
    # Limit to available GPUs
    models = models[:len(gpu_ids)]
    
    print("\n" + "="*70)
    print("MODEL COMPARISON MODE")
    print("="*70)
    print(f"Comparing {len(models)} models on {len(gpu_ids)} GPUs")
    print(f"Models: {', '.join(models)}")
    print(f"Prompt: {prompt[:100]}...")
    print("="*70 + "\n")
    
    # Create comparison directory
    comparison_dir = OUTPUT_DIR / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate with each model
    results = []
    for i, model in enumerate(models):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        result = run_worker(
            gpu_id=gpu_id,
            video_id=i,
            model=model,
            prompt=prompt,
            output_dir=comparison_dir
        )
        results.append(result)
    
    # Save comparison results
    comparison_file = comparison_dir / "comparison_results.json"
    comparison_data = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "models": models,
        "results": results
    }
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    # Print comparison summary
    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS")
    print("="*70)
    for result in results:
        status_icon = "✓" if result["status"] == "success" else "❌"
        print(f"{status_icon} {result['model']:15s} - {result['time_minutes']:.1f} min - GPU {result['gpu_id']}")
    print(f"\nComparison results: {comparison_file}")
    print(f"Videos saved to: {comparison_dir}/")
    print("="*70 + "\n")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU Cluster Video Generation Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 9 videos on 9 GPUs with CogVideoX
  python cluster_launcher.py --num-gpus 9 --model cogvideox
  
  # Use custom prompts from file
  python cluster_launcher.py --num-gpus 4 --model hunyuan --prompts my_prompts.txt
  
  # Compare multiple models side-by-side
  python cluster_launcher.py --compare-models --num-gpus 5
  
  # Use specific GPUs only
  python cluster_launcher.py --gpu-ids 0 2 4 6 --model svd
        """
    )
    
    # GPU selection
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--num-gpus", type=int, default=None,
                          help="Number of GPUs to use (uses first N GPUs)")
    gpu_group.add_argument("--gpu-ids", type=int, nargs="+", default=None,
                          help="Specific GPU IDs to use (e.g., --gpu-ids 0 2 4)")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--compare-models", action="store_true",
                           help="Compare multiple models with the same prompt")
    mode_group.add_argument("--model", type=str, choices=list(AVAILABLE_MODELS.keys()),
                           help="Model to use for generation")
    
    # Prompts
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", type=str,
                             help="Single prompt to use for all videos")
    prompt_group.add_argument("--prompts", type=str,
                             help="Path to text file with prompts (one per line)")
    
    # Generation parameters
    parser.add_argument("--num-frames", type=int, default=None,
                       help="Number of frames to generate")
    parser.add_argument("--fps", type=int, default=None,
                       help="Frames per second")
    parser.add_argument("--steps", type=int, default=None,
                       help="Number of inference steps")
    
    # Info
    parser.add_argument("--list-models", action="store_true",
                       help="List available models and exit")
    
    args = parser.parse_args()
    
    # List models
    if args.list_models:
        print("\nAvailable Models:")
        print("="*70)
        for key, desc in AVAILABLE_MODELS.items():
            print(f"  {key:12s} - {desc}")
        print("="*70 + "\n")
        return 0
    
    # Detect GPUs
    available_gpus = get_available_gpus()
    if not available_gpus:
        print("❌ No GPUs detected! This script requires CUDA-capable GPUs.")
        return 1
    
    print(f"✓ Detected {len(available_gpus)} GPUs: {available_gpus}")
    
    # Select GPUs to use
    if args.gpu_ids:
        gpu_ids = args.gpu_ids
        # Validate
        for gid in gpu_ids:
            if gid not in available_gpus:
                print(f"❌ GPU {gid} not available. Available: {available_gpus}")
                return 1
    elif args.num_gpus:
        gpu_ids = available_gpus[:args.num_gpus]
    else:
        # Default: use all GPUs
        gpu_ids = available_gpus
    
    print(f"✓ Using {len(gpu_ids)} GPUs: {gpu_ids}")
    
    # Compare models mode
    if args.compare_models:
        prompt = args.prompt or DEFAULT_PROMPTS[0]
        compare_models(gpu_ids, prompt)
        return 0
    
    # Normal generation mode
    if not args.model:
        print("❌ Error: --model is required (or use --compare-models)")
        print("Run with --list-models to see available models")
        return 1
    
    # Load prompts
    if args.prompts:
        prompts = load_prompts_from_file(args.prompts)
        print(f"✓ Loaded {len(prompts)} prompts from {args.prompts}")
    elif args.prompt:
        # Repeat single prompt for each GPU
        prompts = [args.prompt] * len(gpu_ids)
    else:
        # Use default prompts
        prompts = DEFAULT_PROMPTS[:len(gpu_ids)]
        print(f"✓ Using {len(prompts)} default prompts")
    
    # Run parallel generation
    parallel_generation(
        gpu_ids=gpu_ids,
        prompts=prompts,
        model=args.model,
        num_frames=args.num_frames,
        fps=args.fps,
        steps=args.steps
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

