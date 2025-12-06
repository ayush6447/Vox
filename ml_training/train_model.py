"""
Training script for Vox sign recognition model.

Loads landmark sequences captured by collect_data.py, trains an LSTM-based
classifier, and saves the model as backend/model/sign_model.h5.

Expected data layout:
    data/
        hello_000.npy
        hello_001.npy
        thanks_000.npy
        ...

File naming convention:
    <label>_<index>.npy
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def load_sequences(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load all .npy files and build (X, y, class_names)."""
    files = sorted(data_dir.glob("*.npy"))
    if not files:
        raise RuntimeError(f"No .npy files found in {data_dir}")

    X_list: List[np.ndarray] = []
    labels: List[str] = []

    for f in files:
        seq = np.load(f)
        if seq.ndim != 2 or seq.shape[1] != 63:
            print(f"Skipping {f}, unexpected shape {seq.shape}")
            continue

        # Label is prefix before first underscore
        name = f.stem
        label = name.split("_")[0]
        X_list.append(seq)
        labels.append(label)

    if not X_list:
        raise RuntimeError("No valid sequences loaded.")

    X = np.stack(X_list, axis=0)  # (N, T, 63)

    # Build label mapping
    unique_labels = sorted(set(labels))
    label_to_idx: Dict[str, int] = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_idx[l] for l in labels], dtype=np.int32)

    return X, y, unique_labels


def build_lstm_model(sequence_length: int, feature_dim: int, num_classes: int) -> tf.keras.Model:
    """Create a simple LSTM-based classifier."""
    inputs = layers.Input(shape=(sequence_length, feature_dim))
    # Masking layer removed to avoid 'Unknown layer: NotEqual' serialization error
    # x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.LSTM(128, return_sequences=False)(inputs)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="sign_lstm_classifier")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Directory with .npy sequences")
    parser.add_argument(
        "--output_model",
        type=str,
        default="../backend/model/sign_model.h5",
        help="Output path for trained model",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    X, y, class_names = load_sequences(data_dir)

    num_classes = len(class_names)
    sequence_length = X.shape[1]
    feature_dim = X.shape[2]

    print(f"Loaded {X.shape[0]} sequences, {num_classes} classes")
    print("Classes:", class_names)

    model = build_lstm_model(sequence_length, feature_dim, num_classes)

    # Simple train/validation split
    num_samples = X.shape[0]
    val_size = max(1, int(0.2 * num_samples))

    # Shuffle data before splitting
    indices = np.random.permutation(num_samples)
    X = X[indices]
    y = y[indices]

    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
        ),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    # Attach class names to model for use in backend (if supported by TF version)
    model.class_names = class_names  # type: ignore[attr-defined]

    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"Saved trained model to {output_path}")

    # Save class names to JSON for backend
    import json
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(class_names, f)
    print(f"Saved class names to {json_path}")


if __name__ == "__main__":
    main()



