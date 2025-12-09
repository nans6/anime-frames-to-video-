from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from transformers import AutoModelForCausalLM


def load_moondream2(device: str = "auto") -> AutoModelForCausalLM:
    if device == "auto":
        device_map = "auto"
    else:
        device_map = {"": device}

    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        trust_remote_code=True,
        dtype="auto",
        device_map=device_map,
    )
    return model


def encode_image(model, image_path: Path):
    img = Image.open(image_path).convert("RGB")
    return model.encode_image(img)


def md_frame_summary(model, encoded_image, temperature: float = 0.35) -> str:
    settings = {"temperature": temperature, "max_tokens": 256, "top_p": 0.3}
    prompt = (
        "In 1-2 sentences, describe what is happening in this frame from an animation. "
        "Mention the main characters (if any), their actions, the setting, and the emotional tone. "
        "Do NOT invent dialogue; describe only what you can infer visually."
    )
    result = model.query(encoded_image, prompt, settings=settings)
    return result.get("answer", "").strip()


def md_motion_synopsis(model, middle_summaries: List[str], temperature: float = 0.3) -> str:
    settings = {"temperature": temperature, "max_tokens": 220, "top_p": 0.3}

    if not middle_summaries:
        return "• No intermediate frames available."

    joined = "\n".join(f"- {s}" for s in middle_summaries[:200])

    prompt = (
        "You are given brief descriptions of consecutive intermediate frames from a short sequence.\n"
        "Summarize the key motion beats and changes across time in 4-8 bullet points.\n"
        "Focus on character movement (if any), camera/scene evolution, lighting changes, and emotional progression.\n"
        "Do not add dialogue.\n\n"
        f"Intermediate frame descriptions:\n{joined}"
    )

    try:
        result = model.query(None, prompt, settings=settings)
        text = result.get("answer", "").strip()
        if text:
            return text
    except Exception:
        pass

    k = min(6, len(middle_summaries))
    idxs = torch.linspace(0, len(middle_summaries) - 1, steps=k).round().to(torch.int64).tolist()
    sampled = [middle_summaries[i] for i in idxs]
    bullets = "\n".join(f"• {s}" for s in sampled)
    return bullets

FRAME_RE = re.compile(r"^frame_(\d+)\.(png|jpg|jpeg|webp|bmp|tiff)$", re.IGNORECASE)

def numeric_frame_key(p: Path) -> int:
    m = FRAME_RE.match(p.name)
    return int(m.group(1)) if m else 10**9


def collect_clip_frames(frames_dir: Path, middle_stride: int = 9) -> Tuple[Path, Path, List[Path]]:

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    all_imgs = [p for p in frames_dir.iterdir() if p.suffix.lower() in exts]

    first_explicit = frames_dir / "first.png"
    last_explicit  = frames_dir / "last.png"

    numeric = [p for p in all_imgs if FRAME_RE.match(p.name)]
    numeric.sort(key=numeric_frame_key)

    if not numeric and not (first_explicit.exists() and last_explicit.exists()):
        raise ValueError(f"No numeric frames found in {frames_dir}")

    # Choose first/last frames for summaries
    if first_explicit.exists():
        first_frame = first_explicit
    else:
        first_frame = numeric[0]

    if last_explicit.exists():
        last_frame = last_explicit
    else:
        last_frame = numeric[-1]

    # Build strided motion frames from numeric list
    middle: List[Path] = []
    if len(numeric) >= 3:
        stride = max(1, int(middle_stride))
        sampled = numeric[::stride]

        # Exclude numeric endpoints to avoid duplicating first/last effort
        first_numeric = numeric[0]
        last_numeric = numeric[-1]
        middle = [p for p in sampled if p != first_numeric and p != last_numeric]

        # If stride is too large and we got nothing, fall back to a tiny core set
        if not middle:
            middle = numeric[1:-1: max(1, len(numeric)//9)]

    return first_frame, last_frame, middle



def build_transition_prompt(
    first_summary: str,
    last_summary: str,
    motion_synopsis: str,
) -> str:

    has_dark_to_bright = ("dark" in first_summary.lower()) and (
        "bright" in last_summary.lower() or "sun" in last_summary.lower()
    )
    has_flowers = ("flower" in motion_synopsis.lower()) or ("flower" in last_summary.lower())

    lines = []
    lines.append(
        "A 3–7 second, 16fps continuous cinematic shot in the same illustrated art style."
    )
    lines.append(f"Start frame context: {first_summary}")
    lines.append("Implied motion/story beats from intermediate frames:")
    lines.append(motion_synopsis.strip())
    lines.append(f"End frame context: {last_summary}")

    lines.append(
        "Use smooth continuous motion with no hard cuts. Maintain consistent art style, "
        "color language, and environmental continuity unless the synopsis implies a natural transformation."
    )

    if has_flowers:
        lines.append(
            "Emphasize gentle natural movement: petals subtly unfurling, a soft breeze, "
            "and gradual clarifying focus as the scene becomes more lush and readable."
        )

    if has_dark_to_bright:
        lines.append(
            "Let the lighting evolve gradually from moody darkness to warm, sunlit serenity, "
            "supporting an emotional arc from unease to tranquility."
        )

    lines.append(
        "Camera motion should be slow and stable—e.g., a gentle push-in, glide, or slight tilt—"
        "with occasional subtle rack focus to reveal detail."
    )

    return " ".join(lines).strip()


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def discover_clips(data_root: Path) -> List[Path]:
    clips = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and p.name.startswith("clip_") and (p / "frames").exists():
            clips.append(p)
    return clips


def process_clip(model, clip_dir: Path, output_root: Path, middle_stride: int):
    frames_dir = clip_dir
    first_path, last_path, middle_paths = collect_clip_frames(frames_dir, middle_stride=middle_stride)

    print(f"\n=== {clip_dir.name} ===")
    print("First :", first_path.name)
    print("Last  :", last_path.name)
    print(f"Middle sampled numeric frames (stride={middle_stride}):", len(middle_paths))

    enc_first = encode_image(model, first_path)
    enc_last = encode_image(model, last_path)

    first_summary = md_frame_summary(model, enc_first)
    last_summary = md_frame_summary(model, enc_last)

    middle_summaries: List[str] = []
    for i, p in enumerate(middle_paths, start=1):
        enc = encode_image(model, p)
        s = md_frame_summary(model, enc)
        middle_summaries.append(s)
        print(f"  middle summarized {i}/{len(middle_paths)} -> {p.name}")

    motion_synopsis = md_motion_synopsis(model, middle_summaries)

    final_prompt = build_transition_prompt(
        first_summary=first_summary,
        last_summary=last_summary,
        motion_synopsis=motion_synopsis,
    )

    clip_name = clip_dir.name
    write_text(output_root / f"{clip_name}_first_frame_summary.txt", first_summary)
    write_text(output_root / f"{clip_name}_last_frame_summary.txt", last_summary)
    write_text(output_root / f"{clip_name}_transition_prompt.txt", final_prompt)

    print("Wrote:")
    print(" ", output_root / f"{clip_name}_first_frame_summary.txt")
    print(" ", output_root / f"{clip_name}_last_frame_summary.txt")
    print(" ", output_root / f"{clip_name}_transition_prompt.txt")


def main():
    parser = argparse.ArgumentParser(
        description="Moondream2 minimal clip pipeline: save only first/last summaries + final transition prompt."
    )
    parser.add_argument("--data_root", required=True, help="Root containing clip_*/frames/")
    parser.add_argument("--output_root", default="generated_prompts", help="One parent output folder for all clips.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument(
        "--middle_stride",
        type=int,
        default=9,
        help="Stride over numeric frames used to infer motion (default=9).",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading Moondream2 on device={args.device}...")
    model = load_moondream2(device=args.device)

    clips = discover_clips(data_root)
    if not clips:
        raise ValueError(f"No clip_* directories with frames/ found under {data_root}")

    print(f"Discovered {len(clips)} clips")

    for clip_dir in clips:
        process_clip(model, clip_dir, output_root, middle_stride=args.middle_stride)

    print(f"\nAll outputs flattened into: {output_root.resolve()}")

if __name__ == "__main__":
    main()
