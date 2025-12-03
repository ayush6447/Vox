# ⚡ Quick Start Guide

## 🚀 Get Running in 5 Minutes

### 1️⃣ Install Dependencies

**Backend:**
```bash
cd backend
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

### 2️⃣ Collect Sample Data (Minimum)

```bash
cd ml_training
python collect_data.py --output_dir data --label hello --num_sequences 10
python collect_data.py --output_dir data --label thanks --num_sequences 10
```

### 3️⃣ Train Model

```bash
python train_model.py --data_dir data
```

### 4️⃣ Start Backend (Terminal 1)

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 5️⃣ Start Frontend (Terminal 2)

```bash
cd frontend
npm run dev
```

### 6️⃣ Open Browser

Go to: **http://localhost:5173**

**Done!** 🎉

---

For detailed instructions, see [SETUP_GUIDE.md](./SETUP_GUIDE.md)


