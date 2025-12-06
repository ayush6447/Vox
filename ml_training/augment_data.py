
"""
Data augmentation script for Vox.
Reads existing .npy files and generates augmented versions (noise, shift, scale).
"""
import numpy as np
from pathlib import Path
import shutil

def augment_data(data_dir: Path):
    files = sorted(data_dir.glob("*.npy"))
    print(f"Found {len(files)} original files. Generating augmentations...")

    for f in files:
        # Skip already augmented files if re-running
        if "_aug_" in f.name:
            continue

        data = np.load(f) # (30, 63)
        label = f.stem

        # 1. Jitter (Add random noise)
        noise = np.random.normal(0, 0.02, data.shape)
        data_noise = data + noise
        np.save(data_dir / f"{label}_aug_noise.npy", data_noise)

        # 2. Scale (Scaling the landmarks around center - simplish approx)
        # Landmarks are 0-1, so scaling might push them out, but small scale is fine.
        scale_factor = np.random.uniform(0.9, 1.1)
        mean = np.mean(data, axis=0) # (63,)
        data_scale = (data - mean) * scale_factor + mean
        np.save(data_dir / f"{label}_aug_scale.npy", data_scale)

        # 3. Time Shift (Shift sequence slightly)
        # Copy first/last frame to pad
        shift = np.random.randint(1, 4)
        data_shift_fwd = np.concatenate([np.repeat(data[:1], shift, axis=0), data[:-shift]])
        np.save(data_dir / f"{label}_aug_shift_fwd.npy", data_shift_fwd)
        
        # 4. Mix (Noise + Scale)
        data_mix = (data_noise - mean) * scale_factor + mean
        np.save(data_dir / f"{label}_aug_mix.npy", data_mix)

    print("Augmentation complete.")

if __name__ == "__main__":
    augment_data(Path("ml_training/data"))
