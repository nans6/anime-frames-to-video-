import torch
import argparse
import PIL.Image
from pathlib import Path
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from transformers import CLIPVisionModel
from diffusers.utils import export_to_video, load_image


def choose_device(name: str) -> torch.device:
    # Choose device for inference
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


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained LoRA weights")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
                        help="Base model ID")
    parser.add_argument("--lora_path", type=str, default="wan_flf2v_lora",
                        help="Path to trained LoRA weights")
    parser.add_argument("--first_frame", type=str, required=True,
                        help="Path to first frame image")
    parser.add_argument("--last_frame", type=str, required=True,
                        help="Path to last frame image")
    parser.add_argument("--output", type=str, default="output.mp4",
                        help="Output video path")
    parser.add_argument("--prompt", type=str, default="high quality anime style, smooth motion, consistent characters",
                        help="Positive prompt")
    parser.add_argument("--negative_prompt", type=str, 
                        default="blurry, low-res, distorted faces, extra limbs, jittery motion",
                        help="Negative prompt")
    parser.add_argument("--height", type=int, default=480,
                        help="Video height")
    parser.add_argument("--width", type=int, default=832,
                        help="Video width")
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Number of frames")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                        help="Guidance scale")
    parser.add_argument("--flow_shift", type=float, default=3.0,
                        help="Flow shift (3.0 for 480P, 5.0 for 720P)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cpu', 'cuda', or 'mps'")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Data type for inference")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    
    args = parser.parse_args()
    
    device = choose_device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    print(f"Loading base model from: {args.model_id}")
    image_encoder = CLIPVisionModel.from_pretrained(
        args.model_id,
        subfolder="image_encoder",
        torch_dtype=torch.float32,
    )
    vae = AutoencoderKLWan.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    
    pipe = WanImageToVideoPipeline.from_pretrained(
        args.model_id,
        vae=vae,
        image_encoder=image_encoder,
        dtype=dtype,
    )
    
    lora_path = Path(args.lora_path)
    if lora_path.exists():
        print(f"Loading LoRA weights from: {args.lora_path}")
        pipe.load_lora_weights(args.lora_path)
        print("LoRA weights loaded successfully")
    else:
        print(f"[WARN] LoRA path not found: {args.lora_path}, running without LoRA")
    print()
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=args.flow_shift,
    )
    
    if device.type == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    
    print(f"Loading first frame from: {args.first_frame}")
    if args.first_frame.startswith("http"):
        first_frame = load_image(args.first_frame)
    else:
        first_frame = PIL.Image.open(args.first_frame).convert("RGB")
    
    print(f"Loading last frame from: {args.last_frame}")
    if args.last_frame.startswith("http"):
        last_frame = load_image(args.last_frame)
    else:
        last_frame = PIL.Image.open(args.last_frame).convert("RGB")
    
    first_frame = first_frame.resize((args.width, args.height), PIL.Image.BICUBIC)
    last_frame = last_frame.resize((args.width, args.height), PIL.Image.BICUBIC)
    
    print(f"\nGenerating video:")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Frames: {args.num_frames}")
    print(f"  Steps: {args.num_inference_steps}")
    print(f"  Guidance: {args.guidance_scale}")
    print(f"  Prompt: {args.prompt[:50]}...")
    print()
    
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        print(f"Using seed: {args.seed}")
    
    output = pipe(
        image=first_frame,
        last_image=last_frame,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).frames[0]
    
    print(f"\nExporting video to: {args.output}")
    export_to_video(output, args.output, fps=16)
    print("Done!")


if __name__ == "__main__":
    main()
