import cv2
import os
from pathlib import Path
from typing import List, Dict, Any
import PIL.Image
from huggingface_hub import HfApi
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
TARGET_WIDTH = 832
TARGET_HEIGHT = 480
CLIP_LEN = 81
STRIDE = 16
OUTPUT_ROOT = Path("data")


def load_frames_from_video(
    video_path: str,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
) -> List[PIL.Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    with tqdm(total=total_frames, desc="Loading frames", unit="frame") as pbar:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(frame_rgb)
            img = img.resize((target_width, target_height), PIL.Image.BICUBIC)
            frames.append(img)
            pbar.update(1)
    cap.release()
    return frames


def generate_all_4k1_clips(
    frames: List[PIL.Image.Image],
    clip_len: int = CLIP_LEN,
    stride: int = STRIDE,
) -> List[Dict[str, Any]]:
    N = len(frames)
    if N < clip_len:
        return []
    results = []
    num_clips = (N - clip_len) // stride + 1
    for start in tqdm(
        range(0, N - clip_len + 1, stride),
        desc="Generating clips",
        total=num_clips,
        unit="clip",
    ):
        end = start + clip_len - 1
        clip = frames[start : end + 1]
        first = clip[0]
        last = clip[-1]
        results.append(
            {
                "first": first,
                "last": last,
                "clip": clip,
                "length": clip_len,
                "start_idx": start,
                "end_idx": end,
            }
        )
    return results


def save_clip_to_disk(
    clip_info: Dict[str, Any],
    base_dir: Path,
):
    L = clip_info["length"]
    start_idx = clip_info["start_idx"]
    first = clip_info["first"]
    last = clip_info["last"]
    frames = clip_info["clip"]
    clip_dir = base_dir / f"clip_{start_idx}"
    frames_dir = clip_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    first.save(clip_dir / "first.png")
    last.save(clip_dir / "last.png")
    for i, frame in enumerate(frames):
        frame.save(frames_dir / f"frame_{i:03d}.png")


def process_video_to_all_4k1_clips(video_path: str, output_root: Path = OUTPUT_ROOT):
    video_path = Path(video_path)
    frames = load_frames_from_video(str(video_path))
    clips = generate_all_4k1_clips(frames, clip_len=CLIP_LEN, stride=STRIDE)
    print(
        f"{video_path.name}: {len(frames)} frames -> "
        f"{len(clips)} clips of length {CLIP_LEN} (stride={STRIDE})"
    )
    video_base = output_root / video_path.stem / "training"
    for clip_info in tqdm(clips, desc="Saving clips", unit="clip"):
        save_clip_to_disk(clip_info, video_base)


def upload_to_huggingface(
    folder_path: str,
    repo_id: str = "attack-on-genai/video-frames",
    repo_type: str = "dataset",
):
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.upload_large_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Successfully uploaded {folder_path} to {repo_id}")


if __name__ == "__main__":
    INPUT_VIDEO = "video.mp4"
    OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)
    process_video_to_all_4k1_clips(INPUT_VIDEO, OUTPUT_ROOT)
    print("Done.")
    upload_to_huggingface(
        folder_path=str(OUTPUT_ROOT),
        repo_id="attack-on-genai/video-frames",
        repo_type="dataset",
    )
