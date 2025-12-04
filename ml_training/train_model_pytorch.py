"""
Training script for SignSpeak sign recognition model using PyTorch.

This version works with Python 3.14+ since PyTorch supports newer Python versions.

Loads landmark sequences captured by collect_data.py, trains an LSTM-based
classifier, and saves the model as backend/model/sign_model.pt.

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
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SignLSTMClassifier(nn.Module):
    """LSTM-based sign language classifier."""

    def __init__(self, sequence_length: int, feature_dim: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, sequence_length, feature_dim)
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[:, -1, :]  # (batch, 128)
        x = self.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


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
    y = np.array([label_to_idx[l] for l in labels], dtype=np.int64)

    return X, y, unique_labels


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 6

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory with .npy sequences"
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="../backend/model/sign_model.pt",
        help="Output path for trained model",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    X, y, class_names = load_sequences(data_dir)

    num_classes = len(class_names)
    sequence_length = X.shape[1]
    feature_dim = X.shape[2]

    print(f"Loaded {X.shape[0]} sequences, {num_classes} classes")
    print("Classes:", class_names)

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # Simple train/validation split
    num_samples = X.shape[0]
    val_size = max(1, int(0.2 * num_samples))

    X_train, X_val = X_tensor[:-val_size], X_tensor[-val_size:]
    y_train, y_val = y_tensor[:-val_size], y_tensor[-val_size:]

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = SignLSTMClassifier(sequence_length, feature_dim, num_classes).to(device)

    train_model(model, train_loader, val_loader, args.epochs, device)

    # Save model and metadata
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model state dict and metadata
    save_dict = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "sequence_length": sequence_length,
        "feature_dim": feature_dim,
        "num_classes": num_classes,
    }
    torch.save(save_dict, output_path)
    print(f"Saved trained model to {output_path}")

    # Also save class names as JSON for easy access
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump({"class_names": class_names}, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()



