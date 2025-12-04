# 📹 Data Collection Guide

## How to Collect Real Sign Language Data

### Step 1: Prepare Your Setup
- ✅ Webcam connected and working
- ✅ Good lighting (face your light source)
- ✅ Plain background (helps MediaPipe detect hands better)
- ✅ Enough space to perform signs clearly

### Step 2: Collect Data for Each Sign

For each sign you want to recognize, run:

```powershell
py -3.11 collect_data.py --output_dir data --label <SIGN_NAME> --num_sequences 20
```

**Example:**
```powershell
# Collect 20 sequences for "hello"
py -3.11 collect_data.py --output_dir data --label hello --num_sequences 20

# Collect 20 sequences for "thanks"
py -3.11 collect_data.py --output_dir data --label thanks --num_sequences 20

# Collect 20 sequences for "yes"
py -3.11 collect_data.py --output_dir data --label yes --num_sequences 20
```

### Step 3: What Happens During Collection

1. **Webcam opens** - You'll see a window with your video feed
2. **2-second countdown** - Get ready to perform the sign
3. **Recording** - Perform the sign clearly for ~1 second (30 frames)
4. **Auto-saves** - Saves as `data/<label>_000.npy`, `data/<label>_001.npy`, etc.
5. **Repeats** - Does this `num_sequences` times

### Tips for Best Results

✅ **Consistency**: Perform the sign the same way each time
✅ **Clear visibility**: Keep your hand fully visible in the camera frame
✅ **Smooth motion**: Perform the sign smoothly, not too fast
✅ **More data = better**: Aim for 20-30 sequences per sign minimum
✅ **Variety**: Collect from slightly different angles for robustness

### Step 4: After Collection

Once you've collected data for all your signs:
1. Check `ml_training/data/` folder - you should see `.npy` files
2. Train the model: `py train_model_pytorch.py --data_dir data`
3. The new model will be saved to `backend/model/sign_model.pt`

### Troubleshooting

**Problem**: Webcam not opening
- Check if another app is using the camera
- Try restarting the script

**Problem**: Hand not detected
- Improve lighting
- Move closer to camera
- Use plain background

**Problem**: Want to stop early
- Press `q` key to quit

---

**Ready to start?** Run the collection command for your first sign!

