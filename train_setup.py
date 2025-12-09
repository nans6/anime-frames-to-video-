import os
import sys
import pickle
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from train import WanFLF2VDatasetFromHF


def main():
    parser = argparse.ArgumentParser(description="Setup: Pre-download and cache dataset with train/test split")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint_setup")
    parser.add_argument("--dataset_repo_id", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of clips to download (default: all)")
    parser.add_argument("--test_split", type=float, default=0.2,
                        help="Fraction of data to use for testing (default: 0.2 = 20%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible train/test split")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SETUP PHASE - Dataset Download & Split")
    print("=" * 60)
    print(f"  Dataset: {args.dataset_repo_id}")
    print(f"  Cache: {checkpoint_dir}")
    if args.max_samples:
        print(f"  Max Samples: {args.max_samples}")
    print(f"  Test Split: {args.test_split:.1%} (seed={args.seed})")
    print("=" * 60 + "\n")

    print("Scanning dataset from Hugging Face...")
    dataset = WanFLF2VDatasetFromHF(
        dataset_repo_id=args.dataset_repo_id,
        max_samples=args.max_samples
    )
    print(f"\n✓ Found {len(dataset)} samples\n")

    print("Downloading all images locally...")
    print("   This will take 10-20 minutes but makes training 10x faster!")
    print()
    
    images_dir = checkpoint_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    total_images = sum(len(s["frame_files"]) + 2 for s in dataset.samples)  # frames + first + last
    
    with tqdm(total=total_images, desc="Downloading images", unit="img") as pbar:
        for sample_idx, sample in enumerate(dataset.samples):
            clip_id = sample["clip_id"]
            clip_dir = images_dir / clip_id
            clip_dir.mkdir(exist_ok=True)
            
            first_remote = sample["first_frame_path"]
            first_local = clip_dir / "first.png"
            if not first_local.exists():
                first_downloaded = dataset.hf_hub_download(
                    repo_id=dataset.dataset_repo_id,
                    filename=first_remote,
                    repo_type="dataset"
                )
                shutil.copy2(first_downloaded, first_local)
            pbar.update(1)
            
            last_remote = sample["last_frame_path"]
            last_local = clip_dir / "last.png"
            if not last_local.exists():
                last_downloaded = dataset.hf_hub_download(
                    repo_id=dataset.dataset_repo_id,
                    filename=last_remote,
                    repo_type="dataset"
                )
                shutil.copy2(last_downloaded, last_local)
            pbar.update(1)
            
            frames_dir = clip_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            
            for frame_remote in sample["frame_files"]:
                frame_name = Path(frame_remote).name  
                frame_local = frames_dir / frame_name
                if not frame_local.exists():
                    frame_downloaded = dataset.hf_hub_download(
                        repo_id=dataset.dataset_repo_id,
                        filename=frame_remote,
                        repo_type="dataset"
                    )
                    shutil.copy2(frame_downloaded, frame_local)
                pbar.update(1)
            
            sample["first_frame_local"] = str(first_local)
            sample["last_frame_local"] = str(last_local)
            sample["frame_files_local"] = [str(frames_dir / Path(f).name) for f in sample["frame_files"]]
    
    print(f"\n✓ Downloaded {total_images} images to: {images_dir}\n")

    print("Splitting dataset into train/test...")
    import random
    import json
    
    random.seed(args.seed)
    indices = list(range(len(dataset.samples)))
    random.shuffle(indices)
    
    split_idx = int(len(indices) * (1 - args.test_split))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_samples = [dataset.samples[i] for i in train_indices]
    test_samples = [dataset.samples[i] for i in test_indices]
    
    print(f"  Train: {len(train_samples)} clips")
    print(f"  Test:  {len(test_samples)} clips")
    print()
    
    train_dir = checkpoint_dir / "train"
    train_dir.mkdir(exist_ok=True)
    
    train_checkpoint = train_dir / "dataset_samples.pkl"
    with open(train_checkpoint, "wb") as f:
        pickle.dump(train_samples, f)
    print(f"✓ Saved train split to: {train_checkpoint}")
    
   
    test_dir = checkpoint_dir / "test"
    test_dir.mkdir(exist_ok=True)
    
    test_checkpoint = test_dir / "dataset_samples.pkl"
    with open(test_checkpoint, "wb") as f:
        pickle.dump(test_samples, f)
    print(f"✓ Saved test split to: {test_checkpoint}")
    
   
    split_info = {
        "total_samples": len(dataset.samples),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "test_split_ratio": args.test_split,
        "seed": args.seed,
        "dataset_repo_id": args.dataset_repo_id,
        "max_samples": args.max_samples,
        "clip_ids_train": [s["clip_id"] for s in train_samples],
        "clip_ids_test": [s["clip_id"] for s in test_samples],
    }
    
    split_info_path = checkpoint_dir / "split_info.json"
    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"✓ Saved split info to: {split_info_path}")
    
    print("\n" + "=" * 60)
    print("✓ SETUP COMPLETE!")
    print("=" * 60)
    print(f"\nDataset split and cached to: {checkpoint_dir}")
    print(f"  Train: {checkpoint_dir}/train/ ({len(train_samples)} clips)")
    print(f"  Test:  {checkpoint_dir}/test/ ({len(test_samples)} clips)")
    print("\nYou can now run training and evaluation!")
    print("\nNext steps:")
    print(f"  # Training:")
    print(f"  python train.py --train_data_dir {checkpoint_dir}/train [other args...]")
    print(f"  # Evaluation:")
    print(f"  python evaluation.py --test_data_dir {checkpoint_dir}/test [other args...]")
    

if __name__ == "__main__":
    main()
