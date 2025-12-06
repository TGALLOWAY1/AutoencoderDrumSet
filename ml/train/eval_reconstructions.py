import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
from typing import Tuple
import torch
import soundfile as sf
from ml.models.autoencoder import AutoencoderConfig, KickConvAutoencoder
from ml.datasets.kick_dataset import KickDataset, KickDatasetConfig
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate autoencoder reconstructions")
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="ml/data/preprocessed_manifest.json",
        help="Path to preprocessed manifest JSON",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="ml/eval/reconstructions",
        help="Directory to save reconstruction audio files",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of samples to reconstruct",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cuda/cpu)",
    )
    
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: Path, device: str) -> Tuple[KickConvAutoencoder, AutoencoderConfig]:
    """Load model from checkpoint file."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    config_dict = checkpoint["config"]
    config = AutoencoderConfig(**config_dict)
    
    # Instantiate model
    model = KickConvAutoencoder(config)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint loss: {checkpoint.get('loss', 'unknown'):.6f}")
    
    return model, config


def run_eval(args):
    # Resolve paths
    project_root = Path(__file__).resolve().parents[2]
    manifest_path = project_root / args.manifest_path
    checkpoint_path = project_root / args.checkpoint_path
    out_dir = project_root / args.out_dir
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    
    # Load model
    model, config = load_model_from_checkpoint(checkpoint_path, args.device)
    
    # Create dataset and dataloader (no shuffle, batch_size = num_samples)
    dataset_config = KickDatasetConfig(
        manifest_path=manifest_path,
        n_samples=config.n_samples,
    )
    dataset = KickDataset(dataset_config)
    dataloader = DataLoader(
        dataset,
        batch_size=args.num_samples,
        shuffle=False,
        num_workers=0,
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Evaluating {args.num_samples} samples...")
    
    # Get first batch
    batch = next(iter(dataloader))
    waveform = batch["waveform"]
    ids = batch.get("id", [f"sample_{i:02d}" for i in range(args.num_samples)])
    
    # Move to device
    waveform = waveform.to(args.device)
    
    # Forward pass
    print("Running inference...")
    with torch.no_grad():
        recon, z = model(waveform)
    
    # Move back to CPU and convert to numpy
    waveform_np = waveform.detach().cpu().numpy()  # (batch, 1, n_samples)
    recon_np = recon.detach().cpu().numpy()  # (batch, 1, n_samples)
    
    # Save audio files
    print(f"Saving audio files to {out_dir}...")
    for i in range(args.num_samples):
        # Extract original and reconstructed arrays
        orig_audio = waveform_np[i, 0, :]  # Flatten to (n_samples,)
        recon_audio = recon_np[i, 0, :]  # Flatten to (n_samples,)
        
        # Get ID for filename
        sample_id = ids[i] if ids[i] is not None else f"sample_{i:02d}"
        # Sanitize ID for filename (remove invalid characters)
        safe_id = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in str(sample_id))
        
        # Save original
        orig_path = out_dir / f"orig_{i:02d}_{safe_id}.wav"
        sf.write(str(orig_path), orig_audio, config.sample_rate)
        
        # Save reconstruction
        recon_path = out_dir / f"recon_{i:02d}_{safe_id}.wav"
        sf.write(str(recon_path), recon_audio, config.sample_rate)
        
        print(f"  Saved: {orig_path.name} and {recon_path.name}")
    
    print(f"\n✓ Evaluation complete! Files saved to: {out_dir}")


def main():
    args = parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
