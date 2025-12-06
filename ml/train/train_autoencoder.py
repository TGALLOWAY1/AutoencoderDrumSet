import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import time
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from ml.models.autoencoder import AutoencoderConfig, create_model
from ml.datasets.kick_dataset import create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train KickConvAutoencoder")
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="ml/data/preprocessed_manifest.json",
        help="Path to preprocessed manifest JSON",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=64,
        help="Latent dimension",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=16384,
        help="Number of samples per waveform",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Defaults to cuda if available, else cpu",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="ml/models/checkpoints",
        help="Directory to save checkpoints",
    )
    
    args = parser.parse_args()
    
    # Set device if not specified
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return args


def train(args):
    # Resolve paths
    project_root = Path(__file__).resolve().parents[2]
    manifest_path = project_root / args.manifest_path
    save_dir = project_root / args.save_dir
    
    # Create save directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {manifest_path}")
    print(f"Saving checkpoints to: {save_dir}")
    print(f"Using device: {args.device}")
    
    # Create dataloader
    train_loader = create_dataloader(
        manifest_path=manifest_path,
        batch_size=args.batch_size,
        shuffle=True,
        n_samples=args.n_samples,
    )
    
    print(f"Dataset size: {len(train_loader.dataset)} samples")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Create model config
    config = AutoencoderConfig(
        n_samples=args.n_samples,
        latent_dim=args.latent_dim,
    )
    
    # Create model
    model = create_model(config)
    model = model.to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # Loss function
    criterion = nn.L1Loss()
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Get waveform and move to device
            waveform = batch["waveform"].to(args.device)
            
            # Forward pass
            recon, z = model(waveform)
            
            # Compute loss
            loss = criterion(recon, waveform)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        # Compute mean loss for epoch
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch}/{args.epochs} - Loss: {mean_loss:.6f} - Time: {epoch_time:.2f}s")
        
        # Save checkpoint
        checkpoint = {
            "model_state": model.state_dict(),
            "config": {
                "sample_rate": config.sample_rate,
                "n_samples": config.n_samples,
                "latent_dim": config.latent_dim,
                "base_channels": config.base_channels,
                "num_down_blocks": config.num_down_blocks,
            },
            "epoch": epoch,
            "loss": mean_loss,
        }
        checkpoint_path = save_dir / f"kick_autoencoder_epoch{epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")
    
    print("\nTraining complete!")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
