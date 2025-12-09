import os
import sys
import json
import argparse
import time
import torch
import torch.nn.functional as F
import PIL.Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any
from dotenv import load_dotenv
import wandb

load_dotenv()
EVAL_PATH = Path(__file__).parent / "eval" / "common_metrics_on_video_quality"
sys.path.insert(0, str(EVAL_PATH))
from calculate_fvd import calculate_fvd
from calculate_ssim import calculate_ssim
from calculate_psnr import calculate_psnr
from calculate_lpips import calculate_lpips
from train import WanFLF2VDatasetFromHF, WanFLF2VDatasetFromLocal
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from transformers import CLIPVisionModel


def choose_device(name: str) -> torch.device:
    name = name.lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[WARN] CUDA requested but not available, falling back to CPU.")
        return torch.device("cpu")
    if name == "mps":
        if hasattr(torch, "mps") and torch.mps.is_available():
            return torch.device("mps")
        print("[WARN] MPS requested but not available, falling back to CPU.")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "mps") and torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resize_video(video_tensor: torch.Tensor, target_size: int) -> torch.Tensor:
    T, C, H, W = video_tensor.shape
    resized_frames = []
    for t in range(T):
        frame = video_tensor[t]
        resized = F.interpolate(
            frame.unsqueeze(0),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        resized_frames.append(resized.squeeze(0))
    return torch.stack(resized_frames, dim=0)


def pil_list_to_tensor(pil_images: List[PIL.Image.Image]) -> torch.Tensor:
    import torchvision.transforms.functional as TF

    frames = [TF.to_tensor(img) for img in pil_images]
    return torch.stack(frames, dim=0)


def prepare_videos_for_eval(
    videos_list: List[torch.Tensor],
    target_size: int = None,
    desc: str = "Preparing videos",
) -> torch.Tensor:
    processed = []
    for video in tqdm(videos_list, desc=desc, leave=False):
        if video.dim() == 4 and video.shape[1] == 3:
            video_tensor = video
        else:
            raise ValueError(f"Unexpected video shape: {video.shape}")
        if video_tensor.max() > 1.0:
            video_tensor = video_tensor / 255.0
        if target_size is not None:
            video_tensor = resize_video(video_tensor, target_size)
        processed.append(video_tensor)
    batch = torch.stack(processed, dim=0)
    return batch


def load_pipeline_with_lora(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    flow_shift: float = 3.0,
    lora_path: str = None,
    hf_lora_repo: str = None,
) -> WanImageToVideoPipeline:
    print(f"Loading base model: {model_id}")
    image_encoder = CLIPVisionModel.from_pretrained(
        model_id,
        subfolder="image_encoder",
        torch_dtype=torch.float32,
    )
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        vae=vae,
        image_encoder=image_encoder,
        dtype=dtype,
    )
    if hf_lora_repo:
        print(f"Loading LoRA weights from HuggingFace: {hf_lora_repo}")
        pipe.load_lora_weights(hf_lora_repo)
        print("✓ LoRA weights loaded from HuggingFace")
    elif lora_path:
        lora_path_obj = Path(lora_path)
        if lora_path_obj.exists():
            print(f"Loading LoRA weights from local: {lora_path}")
            pipe.load_lora_weights(lora_path)
            print("✓ LoRA weights loaded from local path")
        else:
            print(f"[WARN] Local LoRA path not found: {lora_path}")
            print("[WARN] Using base model without LoRA")
    else:
        print("No LoRA weights specified - using base model")
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=flow_shift,
    )
    if device.type == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


def generate_videos_from_dataset(
    pipe: WanImageToVideoPipeline,
    dataset,
    max_samples: int,
    num_frames: int,
    device: torch.device,
    height: int = 480,
    width: int = 832,
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
) -> List[torch.Tensor]:
    generated_videos = []
    print(f"\nGenerating {max_samples} videos...")
    for i in tqdm(range(max_samples), desc="Generating videos"):
        sample = dataset[i]
        first_frame = sample["first"].unsqueeze(0)
        last_frame = sample["last"].unsqueeze(0)
        import torchvision.transforms.functional as TF

        first_pil = TF.to_pil_image(first_frame.squeeze(0))
        last_pil = TF.to_pil_image(last_frame.squeeze(0))
        first_pil = first_pil.resize((width, height), PIL.Image.BICUBIC)
        last_pil = last_pil.resize((width, height), PIL.Image.BICUBIC)
        output = pipe(
            image=first_pil,
            last_image=last_pil,
            prompt=sample["positive_prompt"],
            negative_prompt=sample["negative_prompt"],
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        gen_video = pil_list_to_tensor(output.frames[0])
        generated_videos.append(gen_video)
    return generated_videos


def calculate_all_metrics(
    gt_videos: torch.Tensor,
    gen_videos: torch.Tensor,
    device: torch.device,
    only_final: bool = True,
) -> Dict[str, Any]:
    print("\nCalculating metrics...")
    results = {}
    print("  Computing FVD...")
    results["fvd"] = calculate_fvd(
        gt_videos, gen_videos, device, method="styleganv", only_final=only_final
    )
    print("  Computing SSIM...")
    results["ssim"] = calculate_ssim(gt_videos, gen_videos, only_final=only_final)
    print("  Computing PSNR...")
    results["psnr"] = calculate_psnr(gt_videos, gen_videos, only_final=only_final)
    print("  Computing LPIPS...")
    results["lpips"] = calculate_lpips(
        gt_videos, gen_videos, device, only_final=only_final
    )
    return results


def print_metrics_summary(results: Dict[str, Any]):
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    if "fvd" in results and "value" in results["fvd"]:
        fvd_val = (
            results["fvd"]["value"][0]
            if isinstance(results["fvd"]["value"], list)
            else results["fvd"]["value"]
        )
        print(f"  FVD:   {fvd_val:.2f} (lower is better)")
    if "ssim" in results and "value" in results["ssim"]:
        ssim_val = (
            results["ssim"]["value"][0]
            if isinstance(results["ssim"]["value"], list)
            else results["ssim"]["value"]
        )
        print(f"  SSIM:  {ssim_val:.4f} (higher is better, 0-1 range)")
    if "psnr" in results and "value" in results["psnr"]:
        psnr_val = (
            results["psnr"]["value"][0]
            if isinstance(results["psnr"]["value"], list)
            else results["psnr"]["value"]
        )
        print(f"  PSNR:  {psnr_val:.2f} dB (higher is better)")
    if "lpips" in results and "value" in results["lpips"]:
        lpips_val = (
            results["lpips"]["value"][0]
            if isinstance(results["lpips"]["value"], list)
            else results["lpips"]["value"]
        )
        print(f"  LPIPS: {lpips_val:.4f} (lower is better, 0-1 range)")
    print("=" * 60)


def _extract_metric_value(metric_dict: Dict[str, Any]) -> float:
    if not metric_dict:
        return None
    if "value" not in metric_dict:
        return None
    val = metric_dict["value"]
    if isinstance(val, list) and len(val) > 0:
        return val[0]
    return val


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate video generation model with quality metrics"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
        help="Base model ID",
    )
    parser.add_argument(
        "--lora_path", type=str, default=None, help="Local path to trained LoRA weights"
    )
    parser.add_argument(
        "--hf_lora_repo",
        type=str,
        default=None,
        help="HuggingFace repo for LoRA weights (e.g., 'attack-on-genai/wan-finetune')",
    )
    parser.add_argument(
        "--eval_base",
        action="store_true",
        help="Also evaluate base model without LoRA for comparison",
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default=None,
        help="Path to local test directory (e.g., checkpoint_setup/test). Recommended.",
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        default="attack-on-genai/video-frames",
        help="HuggingFace dataset repository ID (only needed if not using --test_data_dir)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Number of samples to evaluate (only used with --dataset_repo_id)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=17,
        help="Number of frames to generate (must be 4k+1)",
    )
    parser.add_argument(
        "--eval_frame_counts",
        type=str,
        default="1,5,9,17,45,81",
        help="Comma-separated list of frame counts to evaluate (default: 1,5,9,17,45,81)",
    )
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument(
        "--num_inference_steps", type=int, default=30, help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=5.0, help="Guidance scale"
    )
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=3.0,
        help="Flow shift (3.0 for 480P, 5.0 for 720P)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=256,
        help="Resize videos to this size for evaluation (saves memory)",
    )
    parser.add_argument(
        "--only_final",
        action="store_true",
        help="Only compute final aggregate metrics (not per-frame)",
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip generation if videos already exist",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="evaluation_results.json",
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cpu', 'cuda', or 'mps'",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for inference",
    )
    args = parser.parse_args()
    if (args.num_frames - 1) % 4 != 0:
        raise ValueError("num_frames must be 4k+1 (1, 5, 9, 13, 17, ..., 81)")
    device = choose_device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    print("\n" + "=" * 60)
    print("EVALUATION CONFIGURATION")
    print("=" * 60)
    print(f"  Model: {args.model_id}")
    if args.hf_lora_repo:
        print(f"  LoRA: {args.hf_lora_repo} (HuggingFace)")
    elif args.lora_path:
        print(f"  LoRA: {args.lora_path} (local)")
    else:
        print(f"  LoRA: None (base model only)")
    print(f"  Eval Base: {args.eval_base}")
    if args.test_data_dir:
        print(f"  Dataset: {args.test_data_dir} (local test split)")
    else:
        print(f"  Dataset: {args.dataset_repo_id} (HuggingFace)")
        print(f"  Samples: {args.max_samples}")
    print(f"  Frames: {args.num_frames}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Resize for eval: {args.resize}x{args.resize}")
    print(f"  Device: {device}")
    print(f"  Dtype: {args.dtype}")
    print("=" * 60 + "\n")
    eval_frame_counts = [int(x.strip()) for x in args.eval_frame_counts.split(",")]
    print(f"Frame counts to evaluate: {eval_frame_counts}")
    valid_frames = [
        1,
        5,
        9,
        13,
        17,
        21,
        25,
        29,
        33,
        37,
        41,
        45,
        49,
        53,
        57,
        61,
        65,
        69,
        73,
        77,
        81,
    ]
    for fc in eval_frame_counts:
        if fc not in valid_frames:
            raise ValueError(f"Frame count {fc} is not valid (must be 4k+1)")
    print("Loading dataset...")
    if args.test_data_dir:
        print(f"  Source: Local test directory ({args.test_data_dir})")
        dataset = WanFLF2VDatasetFromLocal(
            data_dir=args.test_data_dir,
            num_frames=17,
            dynamic_frames=False,
        )
        num_samples = len(dataset)
    else:
        print(f"  Source: HuggingFace ({args.dataset_repo_id})")
        dataset = WanFLF2VDatasetFromHF(
            dataset_repo_id=args.dataset_repo_id,
            num_frames=17,
            max_samples=args.max_samples,
            dynamic_frames=False,
        )
        num_samples = min(args.max_samples, len(dataset))
    print(f"✓ Loaded {len(dataset)} samples (evaluating {num_samples})")
    wandb_run = None
    try:
        wandb_project = os.getenv("WANDB_PROJECT", "attack-on-genai")
        wandb_entity = os.getenv("WANDB_ENTITY") or None
        wandb_run_name = os.getenv("WANDB_RUN_NAME") or f"eval-{int(time.time())}"
        wandb_config = {
            "eval/model_id": args.model_id,
            "eval/lora_path": args.lora_path,
            "eval/hf_lora_repo": args.hf_lora_repo,
            "eval/eval_base": args.eval_base,
            "eval/test_data_dir": args.test_data_dir,
            "eval/dataset_repo_id": args.dataset_repo_id,
            "eval/num_samples": num_samples,
            "eval/num_frames": args.num_frames,
            "eval/height": args.height,
            "eval/width": args.width,
            "eval/resize": args.resize,
            "eval/num_inference_steps": args.num_inference_steps,
            "eval/guidance_scale": args.guidance_scale,
            "eval/flow_shift": args.flow_shift,
            "eval/only_final": args.only_final,
            "eval/dtype": args.dtype,
            "eval/device": str(device),
        }
        wandb_run_id = os.getenv("WANDB_RUN_ID")
        if wandb_run_id:
            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                id=wandb_run_id,
                resume="must",
                config=wandb_config,
            )
            print(
                f"[wandb] Resuming eval on project={wandb_project}, run_id={wandb_run_id}"
            )
        else:
            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                config=wandb_config,
            )
            print(
                f"[wandb] Logging eval to project={wandb_project}, run={wandb_run_name} (no training run to resume)"
            )
    except Exception as e:
        print(f"[WARN] W&B init skipped or failed: {e}")
    all_results = {}
    total_frames = len(eval_frame_counts)
    start_eval_time = time.time()
    for frame_idx, eval_frames in enumerate(eval_frame_counts, 1):
        elapsed = time.time() - start_eval_time
        avg_time_per_frame = elapsed / frame_idx if frame_idx > 0 else 0
        remaining_frames = total_frames - frame_idx
        eta_seconds = avg_time_per_frame * remaining_frames
        eta_mins = eta_seconds / 60
        print("\n" + "=" * 80)
        print(f"FRAME COUNT {frame_idx}/{total_frames}: {eval_frames} FRAMES")
        print(
            f"Progress: {frame_idx}/{total_frames} | Elapsed: {elapsed/60:.1f}m | ETA: {eta_mins:.1f}m"
        )
        print("=" * 80)
        dataset.num_frames = eval_frames
        print(f"\nCollecting ground truth videos ({eval_frames} frames)...")
        gt_videos_list = []
        for i in tqdm(range(num_samples), desc="Loading GT", leave=False):
            sample = dataset[i]
            gt_videos_list.append(sample["video"])
        print(f"Preparing ground truth videos ({eval_frames} frames)...")
        gt_batch = prepare_videos_for_eval(
            gt_videos_list, target_size=args.resize, desc="Preparing GT videos"
        )
        print(f"✓ GT videos shape: {gt_batch.shape}")
        frame_key = f"frames_{eval_frames}"
        all_results[frame_key] = {}
        if args.eval_base:
            print(f"\n  Evaluating BASE MODEL with {eval_frames} frames...")
            if not args.skip_generation:
                pipe_base = load_pipeline_with_lora(
                    model_id=args.model_id,
                    device=device,
                    dtype=dtype,
                    flow_shift=args.flow_shift,
                    lora_path=None,
                    hf_lora_repo=None,
                )
                gen_videos_base = generate_videos_from_dataset(
                    pipe_base,
                    dataset,
                    num_samples,
                    eval_frames,
                    device,
                    args.height,
                    args.width,
                    args.num_inference_steps,
                    args.guidance_scale,
                )
                del pipe_base
                torch.cuda.empty_cache()
            else:
                gen_videos_base = gt_videos_list
            print(f"  Preparing base model videos ({eval_frames} frames)...")
            gen_batch_base = prepare_videos_for_eval(
                gen_videos_base, target_size=args.resize, desc="Preparing base videos"
            )
            print(f"Base model videos shape: {gen_batch_base.shape}")
            all_results[frame_key]["base"] = calculate_all_metrics(
                gt_batch, gen_batch_base, device, only_final=args.only_final
            )
            print(f"BASE MODEL RESULTS ({eval_frames} frames):")
            print_metrics_summary(all_results[frame_key]["base"])
            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"eval/base/fvd/{eval_frames}frames": _extract_metric_value(
                            all_results[frame_key]["base"].get("fvd")
                        ),
                        f"eval/base/ssim/{eval_frames}frames": _extract_metric_value(
                            all_results[frame_key]["base"].get("ssim")
                        ),
                        f"eval/base/psnr/{eval_frames}frames": _extract_metric_value(
                            all_results[frame_key]["base"].get("psnr")
                        ),
                        f"eval/base/lpips/{eval_frames}frames": _extract_metric_value(
                            all_results[frame_key]["base"].get("lpips")
                        ),
                    }
                )
        if args.hf_lora_repo or args.lora_path:
            print(f"\n  Evaluating FINE-TUNED MODEL with {eval_frames} frames...")
            if not args.skip_generation:
                pipe_finetuned = load_pipeline_with_lora(
                    model_id=args.model_id,
                    device=device,
                    dtype=dtype,
                    flow_shift=args.flow_shift,
                    lora_path=args.lora_path,
                    hf_lora_repo=args.hf_lora_repo,
                )
                gen_videos_finetuned = generate_videos_from_dataset(
                    pipe_finetuned,
                    dataset,
                    num_samples,
                    eval_frames,
                    device,
                    args.height,
                    args.width,
                    args.num_inference_steps,
                    args.guidance_scale,
                )
                del pipe_finetuned
                torch.cuda.empty_cache()
            else:
                gen_videos_finetuned = gt_videos_list
            print(f"  Preparing fine-tuned model videos ({eval_frames} frames)...")
            gen_batch_finetuned = prepare_videos_for_eval(
                gen_videos_finetuned,
                target_size=args.resize,
                desc="Preparing fine-tuned videos",
            )
            print(f"Fine-tuned model videos shape: {gen_batch_finetuned.shape}")
            all_results[frame_key]["finetuned"] = calculate_all_metrics(
                gt_batch, gen_batch_finetuned, device, only_final=args.only_final
            )
            print(f"FINE-TUNED MODEL RESULTS ({eval_frames} frames):")
            print_metrics_summary(all_results[frame_key]["finetuned"])
            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"eval/finetuned/fvd/{eval_frames}frames": _extract_metric_value(
                            all_results[frame_key]["finetuned"].get("fvd")
                        ),
                        f"eval/finetuned/ssim/{eval_frames}frames": _extract_metric_value(
                            all_results[frame_key]["finetuned"].get("ssim")
                        ),
                        f"eval/finetuned/psnr/{eval_frames}frames": _extract_metric_value(
                            all_results[frame_key]["finetuned"].get("psnr")
                        ),
                        f"eval/finetuned/lpips/{eval_frames}frames": _extract_metric_value(
                            all_results[frame_key]["finetuned"].get("lpips")
                        ),
                    }
                )
    print("\n" + "=" * 100)
    print("FRAME-BY-FRAME RESULTS: BASE vs FINE-TUNED")
    print("=" * 100)
    metric_info = {
        "fvd": (True, "0-∞", "< 100 good", "> 300 bad"),
        "ssim": (False, "0-1", "> 0.9 good", "< 0.5 bad"),
        "psnr": (False, "0-∞ dB", "> 30 good", "< 20 bad"),
        "lpips": (True, "0-1", "< 0.1 good", "> 0.5 bad"),
    }
    avg_metrics = {
        "base": {m: [] for m in ["fvd", "ssim", "psnr", "lpips"]},
        "finetuned": {m: [] for m in ["fvd", "ssim", "psnr", "lpips"]},
    }
    for frame_key in sorted(all_results.keys()):
        frame_count = int(frame_key.split("_")[1])
        frame_results = all_results[frame_key]
        if "base" in frame_results and "finetuned" in frame_results:
            print(f"\n{frame_key.upper()} ({frame_count} frames):")
            print(f"{'Metric':<8} {'Base':<12} {'FT':<12} {'Δ':<10} {'Winner':<8}")
            print("-" * 50)
            for metric in ["fvd", "ssim", "psnr", "lpips"]:
                if (
                    metric in frame_results["base"]
                    and metric in frame_results["finetuned"]
                ):
                    base_val = frame_results["base"][metric]["value"][0]
                    ft_val = frame_results["finetuned"][metric]["value"][0]
                    delta = ft_val - base_val
                    avg_metrics["base"][metric].append(base_val)
                    avg_metrics["finetuned"][metric].append(ft_val)
                    lower_better, _, _, _ = metric_info[metric]
                    if lower_better:
                        better = (
                            "✓ FT" if delta < 0 else ("✓ Base" if delta > 0 else "Same")
                        )
                    else:
                        better = (
                            "✓ FT" if delta > 0 else ("✓ Base" if delta < 0 else "Same")
                        )
                    print(
                        f"{metric.upper():<8} {base_val:<12.4f} {ft_val:<12.4f} {delta:+10.4f} {better:<8}"
                    )
        elif "finetuned" in frame_results:
            print(f"\n{frame_key.upper()} ({frame_count} frames - FINE-TUNED ONLY):")
            print(f"{'Metric':<8} {'Value':<12}")
            print("-" * 30)
            for metric in ["fvd", "ssim", "psnr", "lpips"]:
                if metric in frame_results["finetuned"]:
                    val = frame_results["finetuned"][metric]["value"][0]
                    avg_metrics["finetuned"][metric].append(val)
                    print(f"{metric.upper():<8} {val:<12.4f}")
    print("\n" + "=" * 100)
    print("📈 AVERAGES ACROSS ALL FRAME COUNTS")
    print("=" * 100)
    print(
        f"{'Metric':<8} {'Base Avg':<15} {'FT Avg':<15} {'Δ Avg':<12} {'Overall Winner':<12}"
    )
    print("-" * 60)
    for metric in ["fvd", "ssim", "psnr", "lpips"]:
        if avg_metrics["base"][metric] and avg_metrics["finetuned"][metric]:
            base_avg = np.mean(avg_metrics["base"][metric])
            ft_avg = np.mean(avg_metrics["finetuned"][metric])
            delta_avg = ft_avg - base_avg
            lower_better, _, _, _ = metric_info[metric]
            if lower_better:
                winner = (
                    "✓ FT" if delta_avg < 0 else ("✓ Base" if delta_avg > 0 else "Tie")
                )
            else:
                winner = (
                    "✓ FT" if delta_avg > 0 else ("✓ Base" if delta_avg < 0 else "Tie")
                )
            print(
                f"{metric.upper():<8} {base_avg:<15.4f} {ft_avg:<15.4f} {delta_avg:+12.4f} {winner:<12}"
            )
    print("\n" + "=" * 100)
    print("\nMetric Guide:")
    print(
        "  FVD:   Fréchet Video Distance - measures distribution similarity (LOWER = better)"
    )
    print(
        "  SSIM:  Structural Similarity - measures structural quality (HIGHER = better)"
    )
    print(
        "  PSNR:  Peak Signal-to-Noise Ratio - measures pixel accuracy (HIGHER = better)"
    )
    print(
        "  LPIPS: Learned Perceptual Similarity - measures perceptual quality (LOWER = better)"
    )
    output_path = Path(args.output_json)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n✓ Results saved to: {output_path}")
    if wandb_run is not None:
        try:
            wandb_run.save(str(output_path))
        except Exception as e:
            print(f"[WARN] Could not save results JSON to W&B: {e}")
        wandb_run.finish()
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
