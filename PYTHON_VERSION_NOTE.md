# ⚠️ Python Version Compatibility

## Issue: TensorFlow Compatibility

**TensorFlow** currently supports **Python 3.8 - 3.11** only.

If you have **Python 3.12+** installed (like Python 3.14.1), TensorFlow will not install.

## ✅ Solutions

### Option 1: Install Python 3.11 (Recommended)

1. Download Python 3.11 from: https://www.python.org/downloads/
2. Install it (you can have multiple Python versions)
3. Use Python 3.11 specifically for this project:

```powershell
# Check available Python versions
py -0

# Use Python 3.11 specifically
py -3.11 -m pip install -r requirements.txt
py -3.11 -m uvicorn main:app --reload --port 8000
```

### Option 2: Use Virtual Environment with Python 3.11

```powershell
# Install Python 3.11 first, then:
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option 3: Use Docker (Advanced)

Use a Docker container with Python 3.11 pre-installed.

---

## 🔍 Check Your Python Version

```powershell
py --version
```

If it shows **3.12+**, you'll need to install Python 3.11 for TensorFlow compatibility.

---

## 📝 Updated Installation Commands

Once you have Python 3.11:

```powershell
# Backend setup
cd backend
py -3.11 -m pip install -r requirements.txt

# Training scripts
cd ml_training
py -3.11 -m pip install -r requirements.txt
```


