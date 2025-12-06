from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import argparse
import json
import librosa
import soundfile as sf
import numpy as np


@dataclass
class Config:
    raw_dir: Path
    out_dir: Path
    manifest_path: Path
    sample_rate: int
    n_samples: int
    pitch_steps: List[int] = field(default_factory=lambda: [-3, -2, -1, 0, 1, 2, 3])


def load_audio_mono(path: Path, sample_rate: int) -> np.ndarray:
    """Load audio file as mono waveform at specified sample rate.
    
    Args:
        path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        1D numpy array containing mono waveform
    """
    waveform, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return waveform


def trim_silence(waveform: np.ndarray, top_db: float = 40.0) -> np.ndarray:
    """Remove leading and trailing silence from waveform.
    
    Args:
        waveform: Input audio waveform
        top_db: Threshold in dB below peak for silence detection
        
    Returns:
        Trimmed waveform, or original if trimming fails
    """
    try:
        trimmed, _ = librosa.effects.trim(waveform, top_db=top_db)
        if len(trimmed) == 0:
            return waveform
        return trimmed
    except Exception:
        return waveform


def normalize_peak(waveform: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    """Normalize waveform to target peak amplitude in dB.
    
    Args:
        waveform: Input audio waveform
        peak_db: Target peak level in dB (default: -1.0 dB)
        
    Returns:
        Normalized waveform, or original if too quiet
    """
    max_abs = np.max(np.abs(waveform))
    
    # If waveform is too quiet, return unchanged
    if max_abs < 1e-6:
        return waveform
    
    # Convert peak_db to linear amplitude
    target_peak = 10.0 ** (peak_db / 20.0)
    
    # Scale waveform to target peak
    scale_factor = target_peak / max_abs
    return waveform * scale_factor


def pad_or_truncate(waveform: np.ndarray, n_samples: int) -> np.ndarray:
    """Pad or truncate waveform to exact length.
    
    Args:
        waveform: Input audio waveform
        n_samples: Target number of samples
        
    Returns:
        Waveform of exactly n_samples length
    """
    current_len = len(waveform)
    
    if current_len > n_samples:
        # Truncate to first n_samples
        return waveform[:n_samples]
    elif current_len < n_samples:
        # Pad with zeros at the end
        padded = np.zeros(n_samples, dtype=waveform.dtype)
        padded[:current_len] = waveform
        return padded
    else:
        # Already correct length
        return waveform


def align_transient(
    waveform: np.ndarray,
    sample_rate: int,
    max_search_ms: float = 50.0,
    target_index: int = 0,
) -> np.ndarray:
    """Align the main transient towards the beginning of the waveform.
    
    This is a heuristic to get the main transient consistently located.
    It finds the peak in the first max_search_ms milliseconds and shifts
    the waveform so that peak is at target_index. This is not perfect,
    but good enough for an MVP.
    
    Args:
        waveform: Input audio waveform
        sample_rate: Sample rate in Hz
        max_search_ms: Maximum window in milliseconds after start where
                       we expect the transient peak
        target_index: Target index where the transient should be located
        
    Returns:
        Shifted waveform with transient aligned to target_index
    """
    # Handle edge cases: empty or very short waveforms
    if len(waveform) == 0:
        return waveform
    
    # Compute max_search_samples, limiting to waveform length
    max_search_samples = int(sample_rate * max_search_ms / 1000.0)
    max_search_samples = min(max_search_samples, len(waveform))
    
    # If search window is too small, return original
    if max_search_samples <= 0:
        return waveform
    
    # Find the index of the maximum absolute sample in the search window
    search_window = waveform[:max_search_samples]
    peak_index = np.argmax(np.abs(search_window))
    
    # If peak is already at target_index, no shift needed
    if peak_index == target_index:
        return waveform
    
    # Compute shift amount
    shift = peak_index - target_index
    
    # Handle left shift (peak_index > target_index): discard samples at start, pad zeros at end
    if shift > 0:
        if shift >= len(waveform):
            # Shift is larger than waveform, return zeros
            return np.zeros_like(waveform)
        shifted = np.zeros_like(waveform)
        shifted[:-shift] = waveform[shift:]
        return shifted
    
    # Handle right shift (peak_index < target_index): pad zeros at beginning, discard at end
    else:  # shift < 0
        shift_abs = abs(shift)
        if shift_abs >= len(waveform):
            # Shift is larger than waveform, return zeros
            return np.zeros_like(waveform)
        shifted = np.zeros_like(waveform)
        shifted[shift_abs:] = waveform[:-shift_abs]
        return shifted


def preprocess_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    n_samples: int,
) -> np.ndarray:
    """Apply full preprocessing pipeline to a waveform.
    
    Steps:
    1. Trim silence from beginning and end
    2. Align transient to the beginning
    3. Pad or truncate to exact length
    4. Normalize peak amplitude
    
    Args:
        waveform: Input audio waveform
        sample_rate: Sample rate in Hz
        n_samples: Target number of samples for output
        
    Returns:
        Preprocessed waveform of exactly n_samples length
    """
    waveform = trim_silence(waveform)
    waveform = align_transient(waveform, sample_rate=sample_rate)
    waveform = pad_or_truncate(waveform, n_samples=n_samples)
    waveform = normalize_peak(waveform)
    return waveform


def pitch_shift_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    semitones: float,
) -> np.ndarray:
    """Apply pitch shifting to a waveform.
    
    Args:
        waveform: Input audio waveform
        sample_rate: Sample rate in Hz
        semitones: Number of semitones to shift (positive = higher, negative = lower)
        
    Returns:
        Pitch-shifted waveform, or original if semitones == 0
    """
    if semitones == 0:
        return waveform.copy()
    
    return librosa.effects.pitch_shift(
        y=waveform,
        sr=sample_rate,
        n_steps=semitones
    )


def generate_augmented_variants(
    waveform: np.ndarray,
    sample_rate: int,
    n_samples: int,
    pitch_steps: List[int],
) -> List[Tuple[int, np.ndarray]]:
    """Generate pitch-augmented variants of a waveform.
    
    For each pitch step, applies pitch shifting and then runs the result
    through the full preprocessing pipeline (trim, align, pad/truncate, normalize).
    
    Args:
        waveform: Input audio waveform
        sample_rate: Sample rate in Hz
        n_samples: Target number of samples for output
        pitch_steps: List of semitone steps to apply (e.g., [-3, -2, -1, 0, 1, 2, 3])
        
    Returns:
        List of tuples (semitones, processed_waveform) for each pitch step
    """
    variants = []
    
    for step in pitch_steps:
        # Apply pitch shifting
        shifted = pitch_shift_waveform(waveform, sample_rate, semitones=step)
        
        # Run through full preprocessing pipeline
        processed = preprocess_waveform(shifted, sample_rate, n_samples)
        
        # Store as (semitones, waveform) tuple
        variants.append((step, processed))
    
    return variants


def run_preprocessing(args):
    """End-to-end preprocessing pipeline for kick drum audio data.
    
    Walks through raw audio files, generates pitch-augmented variants,
    saves preprocessed WAV files, and writes a JSON manifest.
    
    Args:
        args: Parsed command-line arguments
    """
    # Build Config from args
    config = Config(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        manifest_path=Path(args.manifest_path),
        sample_rate=args.sample_rate,
        n_samples=args.n_samples
    )
    
    # Ensure out_dir exists (create with parents if needed)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Print the config to confirm
    print("Configuration:")
    print(f"  Raw directory: {config.raw_dir}")
    print(f"  Output directory: {config.out_dir}")
    print(f"  Manifest path: {config.manifest_path}")
    print(f"  Sample rate: {config.sample_rate}")
    print(f"  Number of samples: {config.n_samples}")
    print(f"  Pitch steps: {config.pitch_steps}")
    print()
    
    # Collect all audio files recursively from raw_dir
    audio_extensions = {".wav", ".aif", ".aiff", ".WAV", ".AIF", ".AIFF"}
    raw_files = []
    for ext in audio_extensions:
        raw_files.extend(config.raw_dir.rglob(f"*{ext}"))
    
    # Remove duplicates (Path objects handle case-insensitive filesystems correctly)
    raw_files = list(set(raw_files))
    raw_files.sort()  # Sort for consistent processing order
    
    print(f"Found {len(raw_files)} audio file(s) to process")
    print()
    
    # Initialize manifest list
    manifest = []
    files_processed = 0
    total_samples_written = 0
    
    # Process each file
    for raw_path in raw_files:
        try:
            # Load the raw waveform
            waveform = load_audio_mono(raw_path, config.sample_rate)
        except Exception as e:
            print(f"Warning: Failed to load {raw_path}: {e}")
            continue
        
        # Generate augmented variants
        variants = generate_augmented_variants(
            waveform,
            config.sample_rate,
            config.n_samples,
            config.pitch_steps
        )
        
        # Get source file stem for naming
        source_stem = raw_path.stem
        
        # Get relative paths for manifest (relative to project root)
        # Assuming we're running from project root
        try:
            source_file_rel = str(raw_path.relative_to(Path.cwd()))
        except ValueError:
            # If not relative, use absolute path as string
            source_file_rel = str(raw_path)
        
        # Process each variant
        for step, processed_waveform in variants:
            # Construct output filename
            output_filename = f"{source_stem}_ps{step:+d}.wav"
            output_path = config.out_dir / output_filename
            
            # Save the processed waveform
            sf.write(str(output_path), processed_waveform, config.sample_rate)
            
            # Get relative path for manifest
            try:
                file_path_rel = str(output_path.relative_to(Path.cwd()))
            except ValueError:
                file_path_rel = str(output_path)
            
            # Create manifest record
            manifest_record = {
                "id": f"{source_stem}_ps{step:+d}",
                "file_path": file_path_rel,
                "source_file": source_file_rel,
                "sample_rate": config.sample_rate,
                "n_samples": config.n_samples,
                "pitch_shift_semitones": step
            }
            manifest.append(manifest_record)
            total_samples_written += 1
        
        files_processed += 1
        if files_processed % 10 == 0:
            print(f"Processed {files_processed}/{len(raw_files)} files...")
    
    # Write manifest to JSON file
    with open(config.manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Print summary
    print()
    print("=" * 50)
    print("Preprocessing complete!")
    print(f"  Raw files processed: {files_processed}")
    print(f"  Total augmented samples written: {total_samples_written}")
    print(f"  Manifest saved to: {config.manifest_path}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare kick drum audio dataset for training")
    
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="ml/data/raw",
        help="Directory containing raw audio files"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="ml/data/preprocessed",
        help="Directory to save preprocessed audio files"
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="ml/data/preprocessed_manifest.json",
        help="Path to save the preprocessing manifest"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Target sample rate for audio processing"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=16384,
        help="Number of samples per audio segment"
    )
    
    args = parser.parse_args()
    run_preprocessing(args)

