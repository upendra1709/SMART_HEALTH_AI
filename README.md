# 🏥 Smart Health AI — Diagnosis System

## ⚠️ IMPORTANT: Read Before Running

## 📁 Project Structure

'''
SMART_HEALTH_AI_v2/
│
├── backend/
│   ├── app.py                  ← Flask REST API server
│   ├── train_model.ipynb       ← Jupyter notebook for model training
│   ├── requirements.txt        ← Python dependencies
│   ├── Training.csv            ← Training dataset (kaushil268)
│   ├── Testing.csv             ← Testing dataset
│   └── model/
│       ├── disease_model.pkl       ← Trained Random Forest model
│       ├── label_encoder.pkl       ← Disease label encoder
│       ├── model_features.pkl      ← Symptom feature list (132 symptoms)
│       └── symptom_importances.csv ← Feature importance scores
│
├── frontend/
│   ├── index.html              ← Landing / Home page
│   ├── checker.html            ← Symptom checker interface
│   ├── result.html             ← Diagnosis result display page
│   ├── css/
│   │   └── style.css           ← Global styles
│   └── js/
│       └── script.js           ← Frontend logic & API calls
│
└── README.md

---

## 🚀 Setup (4 Steps)

### Step 1 — Create a virtual environment (recommended)
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

### Step 2 — Install dependencies
```bash
cd backend/
pip install -r requirements.txt
```

### Step 3 — Train the model (generates .pkl files)
```bash
cd backend/
jupyter notebook train_model.ipynb
```
This will print accuracy (~85%) and save 3 `.pkl` files into `backend/model/`.

### Step 4 — Start the backend
```bash
python app.py
# Running on http://127.0.0.1:5000
```

### Step 5 — Open the frontend
```bash
cd frontend/
python -m http.server 8080
# Visit: http://localhost:8080
```
--- 
 
## 🧠 Machine Learning Model
 
| Property | Value |
|----------|-------|
| Algorithm | Random Forest Classifier |
| Training Dataset | kaushil268 (Kaggle) |
| Total Diseases | 42 |
| Total Symptoms | 132 |
| Input Format | Binary symptom vector |
| Output | Disease name + probability scores |

---

## 🛠️ Troubleshooting
 
| Problem | Solution |
|---------|----------|
| `.pkl` version mismatch | Delete old `.pkl` files → re-train model |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Port 5000 already in use | Change `port=5000` to `port=5001` in `app.py` |
| `CORS error` in browser | Ensure Flask is running and CORS is enabled (it is by default) |
| Symptom not recognized | Call `GET /symptoms` to see all valid symptom names |
| Frontend not loading | Ensure you're serving from `frontend/` directory |
 
---

## ⚠️ Disclaimer
This system is for educational purposes only. Not a substitute for professional medical advice.
