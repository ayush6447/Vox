"""
Train a model for a single sign (hello) with a background class.

This creates a binary classifier: "hello" vs "background"
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SignLSTMClassifier(nn.Module):
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
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        x = self.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def create_background_data(num_samples: int, seq_len: int = 30) -> np.ndarray:
    """Create random background sequences (no sign being performed)."""
    # Random hand positions (neutral/background)
    background = np.random.rand(num_samples, seq_len, 63).astype(np.float32)
    # Normalize to reasonable hand landmark ranges
    background = background * 0.5 + 0.25  # Scale to 0.25-0.75 range
    return background


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Directory with hello_*.npy files")
    parser.add_argument("--output_model", type=str, default="../backend/model/sign_model.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    
    # Load hello sequences
    hello_files = sorted(data_dir.glob("hello_*.npy"))
    if not hello_files:
        raise RuntimeError(f"No hello_*.npy files found in {data_dir}")
    
    hello_sequences = []
    for f in hello_files:
        seq = np.load(f)
        if seq.ndim == 2 and seq.shape[1] == 63:
            hello_sequences.append(seq)
    
    if not hello_sequences:
        raise RuntimeError("No valid hello sequences found")
    
    X_hello = np.stack(hello_sequences, axis=0)
    y_hello = np.zeros(len(hello_sequences), dtype=np.int64)  # Class 0 = hello
    
    print(f"Loaded {len(hello_sequences)} hello sequences")
    
    # Create background data (same number as hello)
    num_background = len(hello_sequences)
    X_background = create_background_data(num_background, X_hello.shape[1])
    y_background = np.ones(num_background, dtype=np.int64)  # Class 1 = background
    
    # Combine
    X = np.concatenate([X_hello, X_background], axis=0)
    y = np.concatenate([y_hello, y_background], axis=0)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"Total sequences: {len(X)} (hello: {len(hello_sequences)}, background: {num_background})")
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Train/val split
    val_size = max(1, int(0.2 * len(X)))
    X_train, X_val = X_tensor[:-val_size], X_tensor[-val_size:]
    y_train, y_val = y_tensor[:-val_size], y_tensor[-val_size:]
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    sequence_length = X.shape[1]
    feature_dim = X.shape[2]
    num_classes = 2  # hello vs background
    
    model = SignLSTMClassifier(sequence_length, feature_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
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
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    # Save model
    import json
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    class_names = ["hello", "background"]
    
    save_dict = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "sequence_length": sequence_length,
        "feature_dim": feature_dim,
        "num_classes": num_classes,
    }
    torch.save(save_dict, output_path)
    
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump({"class_names": class_names}, f, indent=2)
    
    print(f"\n✅ Model saved to {output_path}")
    print(f"✅ Metadata saved to {metadata_path}")
    print(f"Classes: {class_names}")


if __name__ == "__main__":
    main()



