"""
Create sample training data for testing purposes.

This script generates synthetic hand landmark sequences that can be used
to test the training pipeline without needing MediaPipe/OpenCV.

Usage:
    python create_sample_data.py --output_dir data --num_signs 3 --sequences_per_sign 20
"""

import argparse
import numpy as np
from pathlib import Path


def generate_sample_sequence(label: str, sequence_id: int, seq_len: int = 30) -> np.ndarray:
    """
    Generate a synthetic hand landmark sequence.
    
    Each sign gets a slightly different pattern to make them distinguishable.
    """
    np.random.seed(hash(f"{label}_{sequence_id}") % 2**32)
    
    # Base pattern varies by label
    base_pattern = hash(label) % 10
    
    sequence = []
    for frame in range(seq_len):
        # Create a pattern that varies over time
        time_factor = frame / seq_len
        
        # Generate 63 values (21 landmarks * 3 coordinates)
        frame_data = []
        for i in range(21):
            # x, y, z coordinates
            x = 0.3 + 0.4 * np.sin(time_factor * np.pi + base_pattern * 0.1 + i * 0.05)
            y = 0.3 + 0.4 * np.cos(time_factor * np.pi + base_pattern * 0.15 + i * 0.03)
            z = 0.1 + 0.2 * np.sin(time_factor * np.pi * 2 + base_pattern * 0.2 + i * 0.02)
            
            frame_data.extend([x, y, z])
        
        sequence.append(frame_data)
    
    return np.array(sequence, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate sample training data")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save .npy files"
    )
    parser.add_argument(
        "--num_signs",
        type=int,
        default=3,
        help="Number of different signs to generate"
    )
    parser.add_argument(
        "--sequences_per_sign",
        type=int,
        default=20,
        help="Number of sequences per sign"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=30,
        help="Frames per sequence"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default sign names
    sign_names = ["hello", "thanks", "yes", "no", "please"][:args.num_signs]
    
    print(f"Generating sample data for {len(sign_names)} signs...")
    print(f"Signs: {', '.join(sign_names)}")
    print(f"Sequences per sign: {args.sequences_per_sign}")
    print()
    
    total_files = 0
    for sign_name in sign_names:
        for seq_id in range(args.sequences_per_sign):
            sequence = generate_sample_sequence(sign_name, seq_id, args.sequence_length)
            
            filename = f"{sign_name}_{seq_id:03d}.npy"
            filepath = output_dir / filename
            
            np.save(filepath, sequence)
            total_files += 1
            
            if (seq_id + 1) % 5 == 0:
                print(f"  Generated {seq_id + 1}/{args.sequences_per_sign} sequences for '{sign_name}'")
    
    print()
    print(f"✅ Generated {total_files} sequence files in {output_dir}")
    print(f"   Ready for training!")


if __name__ == "__main__":
    main()

