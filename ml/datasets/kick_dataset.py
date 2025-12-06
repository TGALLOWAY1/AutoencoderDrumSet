from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Union
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf


@dataclass
class KickDatasetConfig:
    manifest_path: Path
    n_samples: int = 16384


class KickDataset(Dataset):
    def __init__(self, config: KickDatasetConfig):
        # Load JSON manifest
        with open(config.manifest_path, 'r') as f:
            self.records = json.load(f)
        
        self.n_samples = config.n_samples
        
        # Set root_dir: if manifest is at ml/data/preprocessed_manifest.json,
        # then root_dir should be the project root
        # file_path in manifest is like "ml/data/preprocessed/..." (relative to project root)
        # So we need to go up 3 levels: ml/data/ -> ml/ -> project root
        # But task says parent.parent, so let's try that first and adjust if needed
        self.root_dir = config.manifest_path.parent.parent.parent
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        
        # Resolve full path
        # file_path in manifest is like "ml/data/preprocessed/..."
        # root_dir is project root, so we can use file_path directly
        wave_path = self.root_dir / record["file_path"]
        
        # Load waveform
        waveform, sample_rate = sf.read(str(wave_path), dtype='float32')
        
        # Convert to mono if stereo
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        
        # Ensure correct length
        if len(waveform) > self.n_samples:
            waveform = waveform[:self.n_samples]
        elif len(waveform) < self.n_samples:
            pad_length = self.n_samples - len(waveform)
            waveform = np.pad(waveform, (0, pad_length), mode='constant')
        
        # Convert to torch tensor with shape (1, n_samples) and values in [-1, 1]
        # Ensure values are in [-1, 1] range
        waveform = np.clip(waveform, -1.0, 1.0)
        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)  # (1, n_samples)
        
        return {
            "waveform": waveform_tensor,
            "id": record.get("id"),
            "pitch_shift_semitones": record.get("pitch_shift_semitones"),
        }


def create_dataloader(
    manifest_path: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = True,
    n_samples: int = 16384,
):
    config = KickDatasetConfig(manifest_path=Path(manifest_path), n_samples=n_samples)
    dataset = KickDataset(config)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    return loader


if __name__ == "__main__":
    # Test the dataset
    manifest_path = Path("ml/data/preprocessed_manifest.json")
    
    # Resolve to absolute path
    if not manifest_path.is_absolute():
        # Assume we're running from project root
        project_root = Path(__file__).parent.parent.parent
        manifest_path = project_root / manifest_path
    
    loader = create_dataloader(
        manifest_path=manifest_path,
        batch_size=4,
        shuffle=True,
        n_samples=16384,
    )
    
    # Get one batch
    batch = next(iter(loader))
    
    print(f"Batch keys: {batch.keys()}")
    print(f"Waveform shape: {batch['waveform'].shape}")
    print(f"IDs: {batch['id'][:4]}")
    print(f"Pitch shifts: {batch['pitch_shift_semitones'][:4]}")
    print("✓ Dataset test passed!")
