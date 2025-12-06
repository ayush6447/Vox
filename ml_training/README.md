# ML Training Scripts

This directory contains scripts for collecting training data and training the sign recognition model.

## 📋 Requirements

**Python 3.11** is required for both data collection and training (TensorFlow compatibility).

### Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `opencv-python` - For webcam access
- `mediapipe` - For hand landmark detection
- `numpy` - For data processing
- `tensorflow` - For model training

## 🎬 Data Collection

Collect sign language sequences using your webcam:

```bash
python collect_data.py --output_dir data --label hello --num_sequences 20
```

**Parameters:**
- `--output_dir`: Directory to save `.npy` files (default: `data`)
- `--label`: Name of the sign (e.g., "hello", "thanks", "yes")
- `--num_sequences`: Number of sequences to collect (recommended: 20-30)
- `--sequence_length`: Frames per sequence (default: 30)

**Example:**
```bash
# Collect data for multiple signs
python collect_data.py --output_dir data --label hello --num_sequences 20
python collect_data.py --output_dir data --label thanks --num_sequences 20
python collect_data.py --output_dir data --label yes --num_sequences 20
```

**Output:** Files saved as `data/<label>_000.npy`, `data/<label>_001.npy`, etc.

## 🧠 Training

Train the LSTM model on collected data:

```bash
python train_model.py --data_dir data
```

**Parameters:**
- `--data_dir`: Directory containing `.npy` files (default: `data`)
- `--output_model`: Output path for trained model (default: `../backend/model/sign_model.h5`)
- `--epochs`: Number of training epochs (default: 25)
- `--batch_size`: Batch size for training (default: 16)

**Example:**
```bash
# Train with default settings
python train_model.py --data_dir data

# Train with custom settings
python train_model.py --data_dir data --epochs 50 --batch_size 32
```

**Output:** Model saved to `backend/model/sign_model.h5`

## 📝 Data Format

Each `.npy` file contains a NumPy array of shape `(30, 63)`:
- **30 frames** per sequence
- **63 features** per frame (21 hand landmarks × 3 coordinates: x, y, z)

## 🎯 Tips for Best Results

1. **Consistency**: Perform each sign the same way every time
2. **More data**: Collect 20-50 sequences per sign for better accuracy
3. **Variety**: Collect from slightly different angles/positions
4. **Good lighting**: Ensure hands are clearly visible
5. **Clean background**: Plain backgrounds help MediaPipe detect hands better

## 🔄 Workflow

1. **Collect data** for each sign you want to recognize
2. **Train model** on all collected data
3. **Model saved** to `backend/model/sign_model.h5`
4. **Restart backend** to load the new model

---

For more details, see the main [SETUP_GUIDE.md](../SETUP_GUIDE.md).
