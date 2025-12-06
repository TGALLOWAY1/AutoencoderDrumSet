import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import json
import torch
from ml.models.autoencoder import AutoencoderConfig, KickConvAutoencoder


def parse_args():
    parser = argparse.ArgumentParser(description="Export checkpoint to final model format")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--out-model-path",
        type=str,
        default="ml/models/kick_autoencoder_v1.pt",
        help="Path to save exported model state dict",
    )
    parser.add_argument(
        "--out-config-path",
        type=str,
        default="ml/models/kick_autoencoder_v1_config.json",
        help="Path to save exported config JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for loading checkpoint (cuda/cpu)",
    )
    
    return parser.parse_args()


def export_model(args):
    # Resolve paths
    project_root = Path(__file__).resolve().parents[2]
    checkpoint_path = project_root / args.checkpoint_path
    out_model_path = project_root / args.out_model_path
    out_config_path = project_root / args.out_config_path
    
    # Create output directories if needed
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    out_config_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    
    # Extract config and state dict
    config_dict = checkpoint["config"]
    state_dict = checkpoint["model_state"]
    
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint loss: {checkpoint.get('loss', 'unknown'):.6f}")
    
    # Recreate config and model
    config = AutoencoderConfig(**config_dict)
    model = KickConvAutoencoder(config)
    model.load_state_dict(state_dict)
    
    # Move to CPU and set eval mode
    model = model.to("cpu")
    model.eval()
    
    # Save model state dict only
    print(f"\nSaving model state dict to: {out_model_path}")
    torch.save(model.state_dict(), out_model_path)
    
    # Save config as JSON
    print(f"Saving config to: {out_config_path}")
    with open(out_config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n✓ Export complete!")
    print(f"  Model: {out_model_path}")
    print(f"  Config: {out_config_path}")


def main():
    args = parse_args()
    export_model(args)


if __name__ == "__main__":
    main()
