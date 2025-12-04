# 🔧 SignSpeak Troubleshooting Guide

## ✅ Current Status
- **Backend**: ✅ Running on http://localhost:8000
- **Frontend**: ✅ Running (Node processes detected)
- **Model**: ✅ Trained and loaded (sign_model.pt)

## 🔍 Common Issues & Solutions

### 1. Frontend Not Loading
**Symptoms**: Blank page or error in browser

**Solutions**:
- Open http://localhost:5173 in your browser
- Check browser console (F12 → Console tab) for errors
- Make sure frontend dev server is running:
  ```powershell
  cd frontend
  npm run dev
  ```

### 2. Camera Not Working
**Symptoms**: "Unable to access camera" or black screen

**Solutions**:
- **Allow camera permissions** when browser prompts
- Check if another app is using the camera
- Try a different browser (Chrome/Edge work best)
- Check Windows camera privacy settings:
  - Settings → Privacy → Camera → Allow apps to access camera

### 3. Hand Not Detected
**Symptoms**: No landmarks detected, no predictions

**Solutions**:
- **Improve lighting** - Face a light source
- **Move closer** to camera
- **Use plain background** - Avoid cluttered backgrounds
- **Show hand clearly** - Keep hand fully visible in frame
- **Wait a moment** - MediaPipe needs a few frames to detect

### 4. No Predictions / Wrong Predictions
**Symptoms**: Predictions not appearing or always wrong

**Solutions**:
- **Check browser console** (F12) for API errors
- **Verify backend is running**: http://localhost:8000
- **Check CORS** - Backend should allow localhost:5173
- **Model might need retraining** with more/better data
- **Perform signs clearly** - Match how you trained them

### 5. Backend Connection Errors
**Symptoms**: "Failed to fetch" or CORS errors

**Solutions**:
- **Check backend is running**:
  ```powershell
  cd backend
  py -m uvicorn main_pytorch:app --reload --port 8000
  ```
- **Test backend directly**: http://localhost:8000
- **Check CORS settings** in backend/main_pytorch.py
- **Verify API URL** in frontend/src/api/api.js

### 6. Model Loading Errors
**Symptoms**: "Model not found" error

**Solutions**:
- **Check model file exists**: `backend/model/sign_model.pt`
- **Verify model was trained**: Check `sign_model.json` exists
- **Restart backend** after training new model

## 🧪 Quick Tests

### Test Backend
```powershell
# Should return: {"status":"ok","message":"SignSpeak backend is running (PyTorch)"}
Invoke-WebRequest http://localhost:8000
```

### Test Prediction Endpoint
```powershell
$body = @{ landmarks = @(0.0) * 63 } | ConvertTo-Json
Invoke-WebRequest -Uri "http://localhost:8000/predict" -Method POST -Body $body -ContentType "application/json"
```

### Test Frontend
1. Open http://localhost:5173
2. Open browser console (F12)
3. Check for errors
4. Allow camera when prompted

## 📊 Debug Checklist

- [ ] Backend running on port 8000?
- [ ] Frontend running on port 5173?
- [ ] Camera permissions granted?
- [ ] Browser console shows no errors?
- [ ] Hand visible in camera feed?
- [ ] MediaPipe detecting landmarks (check canvas)?
- [ ] API calls reaching backend (check Network tab)?
- [ ] Model file exists and is valid?

## 🆘 Still Not Working?

1. **Check browser console** (F12) - Look for red errors
2. **Check Network tab** (F12 → Network) - See if API calls are failing
3. **Check backend logs** - Look at terminal where backend is running
4. **Restart everything**:
   - Stop backend (Ctrl+C)
   - Stop frontend (Ctrl+C)
   - Restart both

## 💡 Tips

- **Use Chrome or Edge** - Best MediaPipe support
- **Good lighting** is crucial for hand detection
- **Plain background** helps MediaPipe work better
- **Wait 2-3 seconds** after showing hand for first detection
- **Perform signs consistently** - Match training data



