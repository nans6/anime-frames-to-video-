import os
import pickle
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from pathlib import Path
import torch
import torch.nn.functional as FNN
from torch.utils.data import Dataset, DataLoader
import PIL.Image
import torchvision.transforms.functional as TF
from natsort import natsorted
from tqdm import tqdm
import wandb
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from transformers import CLIPVisionModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse
import time

try:
    from torchvision.models.optical_flow import raft_small

    RAFT_AVAILABLE = True
except ImportError:
    print(
        "[WARN] torchvision.models.optical_flow not available. Optical flow loss will be disabled."
    )
    RAFT_AVAILABLE = False


class WanFLF2VDatasetFromHF(Dataset):
    def __init__(
        self,
        dataset_repo_id: str,
        num_frames: int = 81,
        max_samples: int = None,
        dynamic_frames: bool = False,
        valid_num_frames: List[int] = None,
        frame_sampling_seed: int = 42,
    ):
        from huggingface_hub import hf_hub_download, HfApi

        self.dataset_repo_id = dataset_repo_id
        self.num_frames = num_frames
        self.max_samples = max_samples
        self.dynamic_frames = dynamic_frames
        self.frame_sampling_seed = frame_sampling_seed
        self.samples: List[Dict[str, Any]] = []
        if valid_num_frames is None:
            valid_num_frames = [5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81]
        self.valid_num_frames = valid_num_frames
        self._rng = None
        if not dynamic_frames:
            if (num_frames - 1) % 4 != 0:
                raise ValueError(f"num_frames must be 4k+1, got {num_frames}")
            if num_frames < 1 or num_frames > 81:
                raise ValueError(
                    f"num_frames must be between 1 and 81, got {num_frames}"
                )
        print(f"Loading dataset from Hugging Face: {dataset_repo_id}")
        api = HfApi()
        try:
            files = api.list_repo_files(repo_id=dataset_repo_id, repo_type="dataset")
            clip_dirs = set()
            for file_path in files:
                if "/training/clip_" in file_path and file_path.endswith("/first.png"):
                    clip_dir = file_path.rsplit("/", 1)[0]
                    clip_dirs.add(clip_dir)
            clip_dirs = sorted(
                clip_dirs,
                key=lambda x: int(x.split("clip_")[-1]) if "clip_" in x else 0,
            )
            if self.max_samples is not None:
                clip_dirs = clip_dirs[: self.max_samples]
            for clip_dir in clip_dirs:
                clip_id = clip_dir.split("/")[-1]
                first_frame_path = f"{clip_dir}/first.png"
                last_frame_path = f"{clip_dir}/last.png"
                frames_dir = f"{clip_dir}/frames"
                if first_frame_path not in files or last_frame_path not in files:
                    print(f"[WARN] Skipping {clip_id}: missing first.png or last.png")
                    continue
                frame_files = []
                for i in range(81):
                    frame_path = f"{frames_dir}/frame_{i:03d}.png"
                    if frame_path in files:
                        frame_files.append(frame_path)
                    else:
                        print(f"[WARN] Missing frame {i} in {clip_id}")
                if len(frame_files) == 0:
                    print(f"[WARN] No frames found in {clip_id}, skipping")
                    continue
                positive_prompt = (
                    "high quality anime style, smooth motion, consistent characters"
                )
                negative_prompt = (
                    "blurry, low-res, distorted faces, extra limbs, jittery motion"
                )
                self.samples.append(
                    {
                        "clip_id": clip_id,
                        "frames_dir_path": frames_dir,
                        "first_frame_path": first_frame_path,
                        "last_frame_path": last_frame_path,
                        "frame_files": frame_files,
                        "length": len(frame_files),
                        "positive_prompt": positive_prompt,
                        "negative_prompt": negative_prompt,
                    }
                )
        except Exception as e:
            raise ValueError(
                f"Failed to scan Hugging Face dataset {dataset_repo_id}: {e}"
            )
        self.hf_hub_download = hf_hub_download
        print(
            f"Loaded {len(self.samples)} samples from Hugging Face dataset {dataset_repo_id}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        if self.dynamic_frames and self._rng is None:
            import numpy as np

            self._rng = np.random.RandomState(self.frame_sampling_seed)
        if self.dynamic_frames:
            num_frames_to_load = self._rng.choice(self.valid_num_frames)
        else:
            num_frames_to_load = self.num_frames
        if "first_frame_local" in s and Path(s["first_frame_local"]).exists():
            first_img_path = s["first_frame_local"]
        else:
            first_img_path = self.hf_hub_download(
                repo_id=self.dataset_repo_id,
                filename=s["first_frame_path"],
                repo_type="dataset",
            )
        first_img = PIL.Image.open(first_img_path).convert("RGB")
        if "last_frame_local" in s and Path(s["last_frame_local"]).exists():
            last_img_path = s["last_frame_local"]
        else:
            last_img_path = self.hf_hub_download(
                repo_id=self.dataset_repo_id,
                filename=s["last_frame_path"],
                repo_type="dataset",
            )
        last_img = PIL.Image.open(last_img_path).convert("RGB")
        first = TF.to_tensor(first_img)
        last = TF.to_tensor(last_img)
        frame_tensors: List[torch.Tensor] = []
        if "frame_files_local" in s and s["frame_files_local"]:
            frame_paths = s["frame_files_local"]
        else:
            frame_paths = s["frame_files"]
        total_available = len(frame_paths)
        if num_frames_to_load == total_available:
            frame_indices = list(range(total_available))
        elif num_frames_to_load == 1:
            frame_indices = [0]
        else:
            stride = (total_available - 1) / (num_frames_to_load - 1)
            frame_indices = [int(round(i * stride)) for i in range(num_frames_to_load)]
            frame_indices[0] = 0
            frame_indices[-1] = total_available - 1
        for frame_idx in frame_indices:
            frame_path = frame_paths[frame_idx]
            if "frame_files_local" in s and Path(frame_path).exists():
                frame_file_path = frame_path
            else:
                frame_file_path = self.hf_hub_download(
                    repo_id=self.dataset_repo_id,
                    filename=frame_path,
                    repo_type="dataset",
                )
            img = PIL.Image.open(frame_file_path).convert("RGB")
            frame_tensors.append(TF.to_tensor(img))
        video = torch.stack(frame_tensors, dim=0)
        return {
            "video": video,
            "first": first,
            "last": last,
            "length": len(frame_indices),
            "num_frames": num_frames_to_load,
            "positive_prompt": s["positive_prompt"],
            "negative_prompt": s["negative_prompt"],
        }


class WanFLF2VDatasetFromLocal(Dataset):
    def __init__(
        self,
        data_dir: str,
        num_frames: int = 81,
        dynamic_frames: bool = False,
        valid_num_frames: List[int] = None,
        frame_sampling_seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.dynamic_frames = dynamic_frames
        self.frame_sampling_seed = frame_sampling_seed
        if valid_num_frames is None:
            valid_num_frames = [5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81]
        self.valid_num_frames = valid_num_frames
        self._rng = None
        if not dynamic_frames:
            if (num_frames - 1) % 4 != 0:
                raise ValueError(f"num_frames must be 4k+1, got {num_frames}")
            if num_frames < 1 or num_frames > 81:
                raise ValueError(
                    f"num_frames must be between 1 and 81, got {num_frames}"
                )
        samples_file = self.data_dir / "dataset_samples.pkl"
        if not samples_file.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {samples_file}\n"
                f"Run train_setup.py first to download and split the dataset."
            )
        print(f"Loading dataset from: {self.data_dir}")
        with open(samples_file, "rb") as f:
            self.samples = pickle.load(f)
        print(f"✓ Loaded {len(self.samples)} samples from local cache")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        if self.dynamic_frames and self._rng is None:
            import numpy as np

            self._rng = np.random.RandomState(self.frame_sampling_seed)
        if self.dynamic_frames:
            num_frames_to_load = self._rng.choice(self.valid_num_frames)
        else:
            num_frames_to_load = self.num_frames
        first = PIL.Image.open(s["first_frame_local"]).convert("RGB")
        last = PIL.Image.open(s["last_frame_local"]).convert("RGB")
        all_frames = []
        for frame_path in s["frame_files_local"]:
            img = PIL.Image.open(frame_path).convert("RGB")
            all_frames.append(img)
        total_frames = len(all_frames)
        if total_frames < num_frames_to_load:
            raise ValueError(
                f"Clip {s['clip_id']} has only {total_frames} frames, need {num_frames_to_load}"
            )
        if num_frames_to_load == total_frames:
            sampled_indices = list(range(total_frames))
        elif num_frames_to_load == 1:
            sampled_indices = [0]
        else:
            stride = (total_frames - 1) / (num_frames_to_load - 1)
            sampled_indices = [
                int(round(i * stride)) for i in range(num_frames_to_load)
            ]
            sampled_indices[0] = 0
            sampled_indices[-1] = total_frames - 1
        frames_tensors = []
        for frame_idx in sampled_indices:
            img = all_frames[frame_idx]
            img_tensor = TF.to_tensor(img)
            frames_tensors.append(img_tensor)
        video_tensor = torch.stack(frames_tensors, dim=0)
        first = TF.to_tensor(first)
        last = TF.to_tensor(last)
        return {
            "video": video_tensor,
            "first": first,
            "last": last,
            "length": len(sampled_indices),
            "num_frames": num_frames_to_load,
            "positive_prompt": s["positive_prompt"],
            "negative_prompt": s["negative_prompt"],
        }


@dataclass
class TrainConfig:
    model_id: str = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    train_data_dir: str = None
    dataset_repo_id: str = "attack-on-genai/video-frames"
    output_dir: str = "wan_flf2v_lora"
    num_frames: int = 9
    dynamic_frames: bool = True
    max_samples: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 1
    max_train_steps: int = 1000
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "no"
    log_every: int = 1
    save_every: int = 1000
    quantization: str = "no"
    flow_shift: float = 3.0
    use_text: bool = True
    use_flow_loss: bool = True
    flow_loss_weight: float = 0.2
    flow_downsample: int = 2
    valid_num_frames: List[int] = None
    frame_sampling_seed: int = 42

    def __post_init__(self):
        if self.valid_num_frames is None:
            self.valid_num_frames = [
                5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81]
        if not self.dynamic_frames:
            if (self.num_frames - 1) % 4 != 0:
                raise ValueError(
                    f"num_frames must be 4k+1 (1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81), "
                    f"got {self.num_frames}"
                )
            if self.num_frames < 1 or self.num_frames > 81:
                raise ValueError(
                    f"num_frames must be between 1 and 81 (got {self.num_frames})"
                )


def encode_text(
    pipe: WanImageToVideoPipeline,
    prompts: List[str],
    device: torch.device,
    dtype: torch.dtype,
):
    tokenizer = pipe.tokenizer
    raw_max_len = getattr(tokenizer, "model_max_length", 77)
    try:
        raw_max_len_int = int(raw_max_len)
    except Exception:
        raw_max_len_int = 77
    max_len = min(max(raw_max_len_int, 1), 256)
    if isinstance(prompts, str):
        prompts = [prompts]
    else:
        prompts = [str(p) for p in prompts]
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        text_embeds = pipe.text_encoder(text_inputs.input_ids)[0]
    return text_embeds.to(device=device, dtype=dtype)


class OpticalFlowLoss(torch.nn.Module):
    def __init__(self, device: torch.device, downsample: int = 1):
        super().__init__()
        if not RAFT_AVAILABLE:
            raise ImportError(
                "torchvision optical flow models not available. Update torchvision."
            )
        self.flow_model = raft_small(weights="DEFAULT", progress=False)
        self.flow_model.to(device)
        self.flow_model.requires_grad_(False)
        self.flow_model.eval()
        self.device = device
        self.downsample = max(1, int(downsample))
        print("[INFO] Loaded RAFT optical flow model")

    def forward(
        self,
        frames_gt: torch.Tensor,
        frames_gen: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        B, F, C, H, W = frames_gt.shape
        flow_dtype = next(self.flow_model.parameters()).dtype
        autocast_device = "cuda" if self.device.type == "cuda" else "cpu"
        if F < 2:
            return torch.tensor(0.0, device=self.device, dtype=dtype)
        losses = []
        with torch.no_grad():
            for f in range(F - 1):
                gt_frame_t = frames_gt[:, f].to(device=self.device, dtype=flow_dtype)
                gt_frame_t1 = frames_gt[:, f + 1].to(
                    device=self.device, dtype=flow_dtype
                )
                if self.downsample > 1:
                    new_h = max(1, H // self.downsample)
                    new_w = max(1, W // self.downsample)
                    gt_frame_t = FNN.interpolate(
                        gt_frame_t,
                        size=(new_h, new_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    gt_frame_t1 = FNN.interpolate(
                        gt_frame_t1,
                        size=(new_h, new_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                with torch.autocast(device_type=autocast_device, enabled=False):
                    flow_gt = self.flow_model(gt_frame_t, gt_frame_t1)
                if isinstance(flow_gt, (list, tuple)):
                    flow_gt = flow_gt[-1]
        for f in range(F - 1):
            gen_frame_t = frames_gen[:, f].to(device=self.device, dtype=flow_dtype)
            gen_frame_t1 = frames_gen[:, f + 1].to(device=self.device, dtype=flow_dtype)
            if self.downsample > 1:
                new_h = max(1, H // self.downsample)
                new_w = max(1, W // self.downsample)
                gen_frame_t = FNN.interpolate(
                    gen_frame_t,
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                )
                gen_frame_t1 = FNN.interpolate(
                    gen_frame_t1,
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                )
            try:
                with torch.autocast(device_type=autocast_device, enabled=False):
                    flow_gen = self.flow_model(gen_frame_t, gen_frame_t1)
                if isinstance(flow_gen, (list, tuple)):
                    flow_gen = flow_gen[-1]
                gt_frame_t = frames_gt[:, f].to(device=self.device, dtype=flow_dtype)
                gt_frame_t1 = frames_gt[:, f + 1].to(
                    device=self.device, dtype=flow_dtype
                )
                if self.downsample > 1:
                    new_h = max(1, H // self.downsample)
                    new_w = max(1, W // self.downsample)
                    gt_frame_t = FNN.interpolate(
                        gt_frame_t,
                        size=(new_h, new_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    gt_frame_t1 = FNN.interpolate(
                        gt_frame_t1,
                        size=(new_h, new_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                with torch.no_grad():
                    with torch.autocast(device_type=autocast_device, enabled=False):
                        flow_gt = self.flow_model(gt_frame_t, gt_frame_t1)
                    if isinstance(flow_gt, (list, tuple)):
                        flow_gt = flow_gt[-1]
                pair_loss = torch.mean((flow_gen - flow_gt) ** 2)
                losses.append(pair_loss)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("[WARN] Flow loss skipped due to OOM on RAFT forward.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return torch.tensor(0.0, device=self.device, dtype=dtype)
                raise
        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device, dtype=dtype)
        total_loss = torch.stack(losses).mean()
        h_norm = max(1, H // self.downsample)
        w_norm = max(1, W // self.downsample)
        spatial_norm = torch.sqrt(torch.tensor(h_norm * w_norm, dtype=torch.float32))
        normalized_loss = total_loss / spatial_norm
        return normalized_loss


def encode_first_last_images(
    pipe: WanImageToVideoPipeline,
    first: torch.Tensor,
    last: torch.Tensor,
    target_device: torch.device,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    B, C, H, W = first.shape
    images_stacked = []
    for b in range(B):
        images_stacked.append(first[b])
        images_stacked.append(last[b])
    images_tensor = torch.stack(images_stacked, dim=0)
    from torchvision.transforms import Normalize, Resize

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    clip_size = 224
    if H != clip_size or W != clip_size:
        images_tensor = torch.nn.functional.interpolate(
            images_tensor,
            size=(clip_size, clip_size),
            mode="bicubic",
            align_corners=False,
        )
    images_tensor = (images_tensor - mean) / std
    original_device = next(pipe.image_encoder.parameters()).device
    if original_device != target_device:
        pipe.image_encoder.to(target_device)
    pixel_values = images_tensor.to(
        device=target_device, dtype=pipe.image_encoder.dtype
    )
    with torch.no_grad():
        image_embeds = pipe.image_encoder(pixel_values).last_hidden_state
    if original_device != target_device:
        pipe.image_encoder.to(original_device)
    return image_embeds.to(device=target_device, dtype=target_dtype)


def compute_wan_flf2v_loss(
    pipe: WanImageToVideoPipeline,
    transformer,
    frames: torch.Tensor,
    first: torch.Tensor,
    last: torch.Tensor,
    positive_prompts,
    negative_prompts,
    device: torch.device,
    dtype: torch.dtype,
    flow_loss_fn: OpticalFlowLoss = None,
    flow_loss_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    B, F, C, H, W = frames.shape
    if positive_prompts is not None and hasattr(pipe, "text_encoder"):
        if isinstance(positive_prompts, str):
            prompts = [positive_prompts] * B
        else:
            prompts = list(positive_prompts)
            if len(prompts) != B:
                prompts = [prompts[0]] * B
        encoder_hidden_states = encode_text(pipe, prompts, device, dtype)
    else:
        encoder_hidden_states = None
    image_hidden_states = encode_first_last_images(
        pipe, first, last, target_device=device, target_dtype=dtype
    )
    with torch.no_grad():
        scaling_factor = getattr(pipe.vae.config, "scaling_factor", None)
        if scaling_factor is None:
            scaling_factor = pipe.vae.config.get("scaling_factor", 0.18215)
        vae_original_device = next(pipe.vae.parameters()).device
        if vae_original_device != device:
            pipe.vae.to(device)
        frames_gpu = frames.to(device)
        first_gpu = first.to(device)
        last_gpu = last.to(device)
        video_in = frames_gpu * 2.0 - 1.0
        video_5d = video_in.permute(0, 2, 1, 3, 4).contiguous()
        vae_out = pipe.vae.encode(video_5d)
        video_latents = vae_out.latent_dist.sample() * scaling_factor
        B, C_vid, T_lat, H_lat, W_lat = video_latents.shape
        first_in = (first_gpu * 2.0 - 1.0).unsqueeze(2)
        last_in = (last_gpu * 2.0 - 1.0).unsqueeze(2)
        first_last_in = torch.cat([first_in, last_in], dim=0)
        first_last_vae_out = pipe.vae.encode(first_last_in)
        first_last_latents = first_last_vae_out.latent_dist.sample() * scaling_factor
        first_latents = first_last_latents[:B]
        last_latents = first_last_latents[B:]
        latents = torch.cat(
            [first_latents, video_latents, last_latents],
            dim=2,
        )
        if vae_original_device != device:
            pipe.vae.to(vae_original_device)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    latents = pipe.channel_proj(latents)
    noise_scheduler = pipe.scheduler
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (B,),
        device=device,
        dtype=torch.long,
    )
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    model_out = transformer(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_image=image_hidden_states,
        return_dict=True,
    )
    model_pred = model_out.sample
    if noise.shape[1] != model_pred.shape[1]:
        noise_for_loss = noise[:, : model_pred.shape[1]]
    else:
        noise_for_loss = noise
    mse_loss = FNN.mse_loss(model_pred, noise_for_loss)
    flow_loss = None
    total_loss = mse_loss
    if flow_loss_fn is not None and flow_loss_weight > 0:
        try:
            model_pred_16 = model_pred[:, :16, :, :, :]
            video_latents_pred = model_pred_16[:, :, 1:-1, :, :]
            scaling_factor = getattr(pipe.vae.config, "scaling_factor", 0.18215)
            video_latents_pred_decoded = video_latents_pred / scaling_factor
            video_latents_pred_5d = video_latents_pred_decoded.contiguous()
            vae_original_device = next(pipe.vae.parameters()).device
            if vae_original_device != device:
                pipe.vae.to(device)
            with torch.no_grad():
                vae_out_pred = pipe.vae.decode(video_latents_pred_5d)
                frames_gen = vae_out_pred.sample.clamp(-1, 1)
                frames_gen = frames_gen.permute(0, 2, 1, 3, 4)
                frames_gen = (frames_gen + 1.0) / 2.0
            if vae_original_device != device:
                pipe.vae.to(vae_original_device)
            frames_gt = frames
            flow_loss = flow_loss_fn(frames_gt, frames_gen, dtype=dtype)
            total_loss = mse_loss + flow_loss_weight * flow_loss
        except Exception as e:
            print(f"[WARN] Flow loss computation failed: {e}")
            flow_loss = None
    return {"loss": total_loss, "mse_loss": mse_loss, "flow_loss": flow_loss}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="Path to local train directory (e.g., checkpoint_setup/train). Recommended for fast loading.",
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        default=None,
        help="Hugging Face dataset repo ID (default from TrainConfig). Only needed if not using --train_data_dir.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="HF model id (default from TrainConfig)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output dir for LoRA weights"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=None,
        help="LoRA rank (default from TrainConfig)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="LoRA alpha (default from TrainConfig)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=None,
        help="LoRA dropout (default from TrainConfig)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Weight decay (default from TrainConfig)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size (default from TrainConfig)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (default from TrainConfig)",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=None,
        help="Log every N steps (default from TrainConfig)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=None,
        help="Save checkpoint every N steps (default from TrainConfig)",
    )
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=None,
        help="Flow shift for scheduler (default from TrainConfig)",
    )
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Number of frames to train on (must be 4k+1: 1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,69,73,77,81). Ignored if --dynamic_frames is True.",
    )
    parser.add_argument(
        "--dynamic_frames",
        action="store_true",
        default=None,
        help="If set, sample random 4k+1 frames (1-81) per batch instead of fixed num_frames",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of clips to use (only used with --dataset_repo_id, default: 10)",
    )
    parser.add_argument(
        "--mixed_precision", type=str, default=None, choices=["no", "bf16", "fp16"]
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["no", "4bit", "8bit"],
        help="Quantization: 'no', '4bit' (QLoRA), or '8bit'",
    )
    parser.add_argument(
        "--use_flow_loss",
        action="store_true",
        default=None,
        help="Enable optical flow loss during training",
    )
    parser.add_argument(
        "--flow_loss_weight",
        type=float,
        default=None,
        help="Weight for optical flow loss (default: 0.05)",
    )
    parser.add_argument(
        "--flow_downsample",
        type=int,
        default=None,
        help="Downsample factor before RAFT (1 = no downsample, 2 = half-res)",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="'auto', 'cpu', 'cuda', or 'mps'"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory from train_setup.py",
    )
    args = parser.parse_args()
    cfg = TrainConfig()
    if args.train_data_dir is not None:
        cfg.train_data_dir = args.train_data_dir
    if args.dataset_repo_id is not None:
        cfg.dataset_repo_id = args.dataset_repo_id
    if args.model_id is not None:
        cfg.model_id = args.model_id
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.max_train_steps is not None:
        cfg.max_train_steps = args.max_train_steps
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        cfg.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.log_every is not None:
        cfg.log_every = args.log_every
    if args.save_every is not None:
        cfg.save_every = args.save_every
    if args.num_frames is not None:
        cfg.num_frames = args.num_frames
    if args.dynamic_frames is not None:
        cfg.dynamic_frames = args.dynamic_frames
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
    if args.lora_rank is not None:
        cfg.lora_rank = args.lora_rank
    if args.lora_alpha is not None:
        cfg.lora_alpha = args.lora_alpha
    if args.lora_dropout is not None:
        cfg.lora_dropout = args.lora_dropout
    if args.mixed_precision is not None:
        cfg.mixed_precision = args.mixed_precision
    if args.quantization is not None:
        cfg.quantization = args.quantization
    if args.flow_shift is not None:
        cfg.flow_shift = args.flow_shift
    if args.use_flow_loss is not None:
        cfg.use_flow_loss = args.use_flow_loss
    if args.flow_loss_weight is not None:
        cfg.flow_loss_weight = args.flow_loss_weight
    if args.flow_downsample is not None:
        cfg.flow_downsample = args.flow_downsample
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = choose_device(args.device)
    print("Using device:", device)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    print()
    dtype = (
        torch.bfloat16
        if cfg.mixed_precision == "bf16"
        else torch.float16 if cfg.mixed_precision == "fp16" else torch.float32
    )
    wandb_run = None
    try:
        wandb_project = os.getenv("WANDB_PROJECT", "attack-on-genai")
        wandb_entity = os.getenv("WANDB_ENTITY") or None
        wandb_run_name = os.getenv("WANDB_RUN_NAME") or f"wan-train-{int(time.time())}"
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config={**asdict(cfg), "device": str(device)},
        )
        wandb.config.update(
            {"mixed_precision": cfg.mixed_precision, "quantization": cfg.quantization},
            allow_val_change=True,
        )
        print(f"[wandb] Logging to project={wandb_project}, run={wandb_run_name}")
    except Exception as e:
        print(f"[WARN] W&B init skipped or failed: {e}")
    checkpoint_path = (
        Path(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    )
    use_checkpoint = checkpoint_path and checkpoint_path.exists()
    if use_checkpoint:
        print("=" * 60)
        print(f"RESUMING FROM CHECKPOINT: {checkpoint_path}")
        print("=" * 60)
        print("Loading saved config, dataset, and model state...")
        print()
    if use_checkpoint:
        print("Loading model from checkpoint...")
    else:
        print("Loading model from:", cfg.model_id)
    image_encoder = CLIPVisionModel.from_pretrained(
        cfg.model_id,
        subfolder="image_encoder",
        torch_dtype=torch.float32,
    )
    vae = AutoencoderKLWan.from_pretrained(
        cfg.model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    load_kwargs = {"dtype": dtype}
    if cfg.quantization in ("4bit", "8bit") and device.type == "cuda":
        pass
    pipe = WanImageToVideoPipeline.from_pretrained(
        cfg.model_id,
        vae=vae,
        image_encoder=image_encoder,
        **load_kwargs,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=cfg.flow_shift,
    )
    pipe.vae.requires_grad_(False)
    if hasattr(pipe, "text_encoder"):
        pipe.text_encoder.requires_grad_(False)
    if hasattr(pipe, "image_encoder"):
        pipe.image_encoder.requires_grad_(False)
    transformer = pipe.transformer
    use_quantization = False
    if cfg.quantization in ("4bit", "8bit"):
        if device.type != "cuda":
            print(f"[WARN] Quantization ({cfg.quantization}) requires CUDA.")
            print(f"[WARN] Current device: {device}. Skipping quantization.")
            print(
                f"[WARN] Consider using --mixed_precision bf16 or fp16 for memory savings."
            )
        else:
            try:
                import bitsandbytes as bnb

                print(f"[INFO] Quantization requested: {cfg.quantization}")
                print(
                    "[WARN] Direct quantization of diffusers transformers is complex."
                )
                print(
                    "[WARN] Diffusers models don't support BitsAndBytesConfig directly."
                )
                print("[WARN] For now, quantization is not fully implemented.")
                print(
                    "[WARN] Recommendation: Use --mixed_precision bf16 or fp16 instead."
                )
                print("[WARN] This provides ~2x memory savings and is fully supported.")
                print("[WARN] Continuing without quantization...")
                use_quantization = False
            except ImportError:
                print(
                    "[WARN] bitsandbytes not installed. Install with: pip install bitsandbytes"
                )
                print("[WARN] Skipping quantization. Using full precision.")
                use_quantization = False
    print("Adding LoRA to transformer...")
    lora_conf = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_conf)
    pipe.transformer = transformer
    if hasattr(pipe.transformer, "enable_gradient_checkpointing"):
        print("Enabling gradient checkpointing on transformer...")
        pipe.transformer.enable_gradient_checkpointing()
    else:
        print("Gradient checkpointing method not found; setting flags.")
        pipe.transformer.gradient_checkpointing = True
        if hasattr(pipe.transformer, "config"):
            pipe.transformer.config.use_gradient_checkpointing = True
    pipe.vae.to("cpu")
    if hasattr(pipe, "image_encoder"):
        pipe.image_encoder.to("cpu")
    if hasattr(pipe, "text_encoder"):
        pipe.text_encoder.to(device, dtype=dtype)
    pipe.transformer.to(device, dtype=dtype)
    pipe.channel_proj = torch.nn.Conv3d(
        in_channels=16,
        out_channels=36,
        kernel_size=1,
        stride=1,
        padding=0,
    )
    pipe.channel_proj.to(device, dtype=dtype)
    pipe.channel_proj.requires_grad_(False)
    flow_loss_fn = None
    if cfg.use_flow_loss:
        if RAFT_AVAILABLE:
            try:
                flow_loss_fn = OpticalFlowLoss(
                    device=device,
                    downsample=cfg.flow_downsample,
                )
                print(f"✓ Optical flow loss enabled (weight={cfg.flow_loss_weight})")
            except Exception as e:
                print(f"[WARN] Failed to initialize optical flow loss: {e}")
                print("[WARN] Continuing without flow loss...")
        else:
            print(
                "[WARN] Optical flow not available (torchvision >= 0.14). Skipping flow loss."
            )
    trainable_params = [p for p in pipe.transformer.parameters() if p.requires_grad]
    print(f"Trainable parameters (LoRA): {sum(p.numel() for p in trainable_params):,}")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    if cfg.train_data_dir:
        print(f"Loading dataset from local directory: {cfg.train_data_dir}")
        dataset = WanFLF2VDatasetFromLocal(
            data_dir=cfg.train_data_dir,
            num_frames=cfg.num_frames,
            dynamic_frames=cfg.dynamic_frames,
            valid_num_frames=cfg.valid_num_frames,
            frame_sampling_seed=cfg.frame_sampling_seed,
        )
        print(f"✓ Loaded {len(dataset)} training samples from local split")
    elif use_checkpoint:
        dataset_checkpoint = checkpoint_path / "dataset_samples.pkl"
        if dataset_checkpoint.exists():
            print(f"Loading dataset from checkpoint: {dataset_checkpoint}")
            with open(dataset_checkpoint, "rb") as f:
                samples = pickle.load(f)
            dataset = WanFLF2VDatasetFromHF.__new__(WanFLF2VDatasetFromHF)
            dataset.samples = samples
            dataset.num_frames = cfg.num_frames
            dataset.dynamic_frames = cfg.dynamic_frames
            dataset.valid_num_frames = cfg.valid_num_frames
            dataset.frame_sampling_seed = cfg.frame_sampling_seed
            dataset._rng = None
            dataset.max_samples = cfg.max_samples
            from huggingface_hub import hf_hub_download

            dataset.hf_hub_download = hf_hub_download
            dataset.dataset_repo_id = cfg.dataset_repo_id
            print(f"✓ Loaded {len(dataset.samples)} samples from checkpoint")
        else:
            print(f"[WARN] Dataset checkpoint not found, loading from Hugging Face...")
            dataset = WanFLF2VDatasetFromHF(
                cfg.dataset_repo_id,
                num_frames=cfg.num_frames,
                max_samples=cfg.max_samples,
                dynamic_frames=cfg.dynamic_frames,
                valid_num_frames=cfg.valid_num_frames,
                frame_sampling_seed=cfg.frame_sampling_seed,
            )
    else:
        print("Building dataset from Hugging Face:", cfg.dataset_repo_id)
        dataset = WanFLF2VDatasetFromHF(
            cfg.dataset_repo_id,
            num_frames=cfg.num_frames,
            max_samples=cfg.max_samples,
            dynamic_frames=cfg.dynamic_frames,
            valid_num_frames=cfg.valid_num_frames,
            frame_sampling_seed=cfg.frame_sampling_seed,
        )
    if cfg.dynamic_frames:
        print(f"✓ Training with DYNAMIC frames: {cfg.valid_num_frames}")
    else:
        print(f"Training with FIXED {cfg.num_frames} frames per sample")
    if not cfg.train_data_dir and cfg.max_samples:
        print(
            f"Limited to {cfg.max_samples} clips (out of 1,238 total = {cfg.max_samples * 81:,} frames)"
        )
    use_pin_memory = device.type == "cuda"
    num_workers = 4 if device.type == "cuda" else 0
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=True if device.type == "cuda" and num_workers > 0 else False,
    )
    if cfg.mixed_precision == "fp16" and device.type in ("cuda", "mps"):
        if device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler()
        elif device.type == "mps" and hasattr(torch, "mps"):
            scaler = torch.mps.amp.GradScaler()
        else:
            scaler = None
    else:
        scaler = None
    global_step = 0
    pipe.transformer.train()
    use_autocast = cfg.mixed_precision in ("bf16", "fp16")
    autocast_device = device.type
    start_time = time.time()
    pbar = tqdm(total=cfg.max_train_steps, desc="Training", unit="step")
    while global_step < cfg.max_train_steps:
        for batch in dataloader:
            if global_step >= cfg.max_train_steps:
                break
            video = batch["video"]
            first = batch["first"]
            last = batch["last"]
            lengths = batch["length"]
            pos_prompts = batch["positive_prompt"]
            neg_prompts = batch["negative_prompt"]
            B, F, C, H, W = video.shape
            actual_frame_count = batch.get("num_frames", [F])
            if isinstance(actual_frame_count, (list, tuple)):
                frame_info = f"frames={actual_frame_count[0] if len(actual_frame_count) > 0 else F}"
            else:
                frame_info = f"frames={actual_frame_count}"
            tqdm.write(
                f"[step] starting step {global_step+1}/{cfg.max_train_steps} "
                f"with batch_size={B}, {frame_info}"
            )
            loss_dict = None
            if scaler is not None and autocast_device in ("cuda", "mps"):
                with torch.autocast(device_type=autocast_device, dtype=dtype):
                    loss_dict = compute_wan_flf2v_loss(
                        pipe=pipe,
                        transformer=pipe.transformer,
                        frames=video,
                        first=first,
                        last=last,
                        positive_prompts=pos_prompts,
                        negative_prompts=neg_prompts,
                        device=device,
                        dtype=dtype,
                        flow_loss_fn=flow_loss_fn,
                        flow_loss_weight=(
                            cfg.flow_loss_weight if cfg.use_flow_loss else 0.0
                        ),
                    )
                loss = loss_dict["loss"] / cfg.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                if use_autocast and autocast_device in ("cuda", "mps"):
                    ctx = torch.autocast(device_type=autocast_device, dtype=dtype)
                else:

                    class NullCtx:
                        def __enter__(self):
                            return None

                        def __exit__(self, exc_type, exc_val, exc_tb):
                            return False

                    ctx = NullCtx()
                with ctx:
                    loss_dict = compute_wan_flf2v_loss(
                        pipe=pipe,
                        transformer=pipe.transformer,
                        frames=video,
                        first=first,
                        last=last,
                        positive_prompts=pos_prompts,
                        negative_prompts=neg_prompts,
                        device=device,
                        dtype=dtype,
                        flow_loss_fn=flow_loss_fn,
                        flow_loss_weight=(
                            cfg.flow_loss_weight if cfg.use_flow_loss else 0.0
                        ),
                    )
                loss = loss_dict["loss"] / cfg.gradient_accumulation_steps
                loss.backward()
            if (global_step + 1) % cfg.gradient_accumulation_steps == 0:
                if scaler is not None and autocast_device in ("cuda", "mps"):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                optimizer.zero_grad()
            global_step += 1
            pbar.update(1)
            mse_val = (
                loss_dict["mse_loss"].item() if loss_dict["mse_loss"] is not None else 0
            )
            flow_val = (
                loss_dict["flow_loss"].item()
                if loss_dict["flow_loss"] is not None
                else 0
            )
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.6f}",
                    "mse": f"{mse_val:.6f}",
                    "flow": f"{flow_val:.6f}" if flow_val > 0 else "N/A",
                    "frames": F,
                    "step": f"{global_step}/{cfg.max_train_steps}",
                }
            )
            if global_step % cfg.log_every == 0:
                elapsed = time.time() - start_time
                steps_per_sec = global_step / max(elapsed, 1e-6)
                steps_per_min = steps_per_sec * 60.0
                remaining_steps = max(cfg.max_train_steps - global_step, 0)
                log_msg = (
                    f"[step] finished step {global_step}/{cfg.max_train_steps}, "
                    f"loss={loss.item():.6f}, mse={mse_val:.6f}, F={F}, "
                    f"speed={steps_per_sec:.4f} steps/s (~{steps_per_min:.1f} steps/min), "
                    f"remaining_steps={remaining_steps}"
                )
                if flow_val > 0:
                    log_msg += f", flow={flow_val:.6f}"
                tqdm.write(log_msg)
                log_dict = {
                    "train/loss": loss.item(),
                    "train/mse_loss": mse_val,
                    "data/frames": F,
                    "train/global_step": global_step,
                    "train/steps_per_sec": steps_per_sec,
                    "train/steps_per_min": steps_per_min,
                    "train/remaining_steps": remaining_steps,
                }
                if flow_val > 0:
                    log_dict["train/flow_loss"] = flow_val
                if wandb_run is not None:
                    wandb_run.log(log_dict, step=global_step)
            if global_step % cfg.save_every == 0:
                ckpt_dir = os.path.join(
                    cfg.output_dir, f"checkpoint_step_{global_step}"
                )
                os.makedirs(ckpt_dir, exist_ok=True)
                pipe.transformer.save_pretrained(ckpt_dir)
                tqdm.write(f"Saved LoRA checkpoint to: {ckpt_dir}")
            if global_step >= cfg.max_train_steps:
                break
    pbar.close()
    pipe.transformer.save_pretrained(cfg.output_dir)
    import json

    adapter_config_path = os.path.join(cfg.output_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, "r") as f:
            config = json.load(f)
        config["base_model_name_or_path"] = cfg.model_id
        if "task_type" in config and config["task_type"] is None:
            config["task_type"] = "image_to_video"
        with open(adapter_config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"✓ Fixed adapter metadata for upload")
    readme_path = os.path.join(cfg.output_dir, "README.md")
    if os.path.exists(readme_path):
        try:
            import re

            readme_txt = None
            with open(readme_path, "r", encoding="utf-8") as f:
                readme_txt = f.read()
            if readme_txt:
                updated = re.sub(
                    r"^base_model:.*$",
                    f"base_model: {cfg.model_id}",
                    readme_txt,
                    flags=re.MULTILINE,
                )
                updated = re.sub(
                    r"^base_model_name_or_path:.*$",
                    f"base_model_name_or_path: {cfg.model_id}",
                    updated,
                    flags=re.MULTILINE,
                )
                if updated != readme_txt:
                    with open(readme_path, "w", encoding="utf-8") as f:
                        f.write(updated)
                    print("✓ Fixed README base_model metadata for upload")
        except Exception as e:
            print(f"[WARN] Could not update README metadata: {e}")
    total_time = time.time() - start_time
    print("Training complete. LoRA saved to:", cfg.output_dir)
    if wandb_run is not None:
        wandb_run.log(
            {
                "train/final_step": global_step,
                "train/total_time_sec": total_time,
            },
            step=global_step,
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
