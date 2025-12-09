import os
import csv
import io
import base64
import argparse
from pathlib import Path
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 

def encode_image(path, max_size=768):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

SYSTEM_PROMPT = """
You are an animation director and prompt engineer for a first-and-last-frame-to-video diffusion model (Wan FLF2V).

Given FIRST and LAST frames of a clip, output a JSON object with:

- reasoning: short explanation of what happens between the frames.
- positive_prompt: visual and motion prompt (style, characters, scene, lighting, motion).
- negative_prompt: list of things to avoid (e.g. 'blurry, low-res, deformed faces, extra limbs, inconsistent character design, jittery motion').

Return ONLY valid JSON.
"""

def build_user_content(first_b64, last_b64, clip_id):
    return [
        {"type": "text", "text": f"These two images are the FIRST and LAST frames for clip {clip_id}."},
        {"type": "image_url", "image_url": {"url": first_b64}},
        {"type": "text", "text": "FIRST frame."},
        {"type": "image_url", "image_url": {"url": last_b64}},
        {"type": "text", "text": "LAST frame."}
    ]

def get_prompts_for_clip(first_path, last_path, clip_id, model="gpt-5.1"):
    first_b64 = encode_image(first_path)
    last_b64 = encode_image(last_path)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(first_b64, last_b64, clip_id)},
        ],
        response_format={"type": "json_object"}
    )
    obj = resp.choices[0].message.content
    import json
    data = json.loads(obj)
    pos = data.get("positive_prompt", "")
    neg = data.get("negative_prompt", "")
    return pos, neg

def prepare_metadata(dataset_root, out_csv, model="gpt-5.1", use_gpt=True):
    clip_dirs = [
        d for d in os.listdir(dataset_root)
        if d.startswith("clip_") and os.path.isdir(os.path.join(dataset_root, d))
    ]
    clip_dirs = natsorted(clip_dirs)

    rows = []
    for clip_dir in tqdm(clip_dirs, desc="Processing clips", unit="clip"):
        clip_id = clip_dir
        clip_path = os.path.join(dataset_root, clip_dir)
        frames_dir = os.path.join(clip_path, "frames")

        first_path = os.path.join(clip_path, "first.png")
        last_path  = os.path.join(clip_path, "last.png")

        if not (os.path.isdir(frames_dir) and os.path.isfile(first_path) and os.path.isfile(last_path)):
            tqdm.write(f"Skipping {clip_id}: missing frames/ or first/last frame.")
            continue

        if use_gpt:
            tqdm.write(f"[GPT] Generating prompts for {clip_id}...")
            positive_prompt, negative_prompt = get_prompts_for_clip(first_path, last_path, clip_id, model=model)
        else:
            positive_prompt = "high quality anime style, smooth motion, consistent characters"
            negative_prompt = "blurry, low-res, distorted faces, extra limbs, jittery motion"

        rows.append({
            "clip_id": clip_id,
            "frames_dir": frames_dir,
            "first_frame": first_path,
            "last_frame": last_path,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
        })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["clip_id", "frames_dir", "first_frame", "last_frame", "positive_prompt", "negative_prompt"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {out_csv}")

def find_training_directory(data_root: Path = Path("data")):
    if not data_root.exists():
        raise ValueError(f"Data root directory does not exist: {data_root}")
    
    training_dirs = []
    for item in data_root.iterdir():
        if item.is_dir():
            training_path = item / "training"
            if training_path.exists() and training_path.is_dir():
                training_dirs.append(training_path)
    
    if not training_dirs:
        raise ValueError(f"No training directories found in {data_root}")
    
    if len(training_dirs) > 1:
        print(f"Found multiple training directories: {[str(d) for d in training_dirs]}")
        print(f"Using first one: {training_dirs[0]}")
    
    return training_dirs[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    data_root = Path("data")
    training_dir = find_training_directory(data_root)
    dataset_root = str(training_dir)
    out_csv = str(training_dir / "metadata.csv")
    
    print(f"Dataset root: {dataset_root}")
    print(f"Output CSV: {out_csv}")
    
    prepare_metadata(
        dataset_root=dataset_root,
        out_csv=out_csv
    )