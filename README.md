# 🏦 LoanNet — Loan Approval Prediction Using Neural Networks

A complete end-to-end deep learning project that predicts loan approval probability using a multi-layer neural network trained on applicant financial and demographic data.

---

## 🚀 Quick Start

```bash
# 1. Clone / download the project
cd loan_approval_nn

# 2. Create virtual environment (Python 3.11 required)
python -m venv venv

# 3. Activate
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the Streamlit app
streamlit run app.py
```

App opens at **http://localhost:8501**

---

## 📁 Project Structure

```
loan_approval_nn/
├── app.py                 ← Streamlit web application
├── requirements.txt       ← Python 3.11 dependencies
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py   ← Load, clean, engineer, encode, split, scale
│   ├── model.py           ← LoanNet neural network architecture
│   ├── train.py           ← Training script (CLI + programmatic)
│   └── predict.py         ← Inference for new applicants
├── models/                ← Auto-created during training
│   ├── loannet_best.h5
│   ├── scaler.pkl
│   └── encoders.pkl
└── data/
    └── loan_data.csv      ← Your dataset (or generate via app)
```

---

## 🗄️ Dataset

### Option A — Generate Synthetic Data (built into app)
Click "Generate Dataset" in the sidebar. Produces realistic synthetic loan records with correlated features and business-logic-based approval labels.

### Option B — Upload Your Own CSV
Required columns:

| Column | Type | Description |
|--------|------|-------------|
| `age` | int | Applicant age (18–80) |
| `income` | int | Annual income ($) |
| `loan_amount` | int | Requested loan ($) |
| `loan_term` | int | Term in months |
| `credit_score` | int | FICO score (300–850) |
| `employment_years` | float | Years at current job |
| `debt_to_income` | float | DTI ratio (0–1) |
| `num_credit_lines` | int | Open credit lines |
| `num_delinquencies` | int | Missed payments |
| `property_value` | int | Collateral value ($) |
| `education` | str | High School / Bachelor / Master / PhD / Associate |
| `employment_type` | str | Salaried / Self-Employed / Business / Contract |
| `loan_purpose` | str | Home / Car / Education / Personal / Business |
| `approved` | int | 1 = Approved, 0 = Rejected |

### Option C — Public Datasets
- [Kaggle: Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- [UCI: Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

---

## 🧠 Model Architecture

```
Input (17 features)
    ↓
Dense 256 + BatchNorm + Dropout(0.4)   [ReLU, L2 reg]
    ↓
Dense 128 + BatchNorm + Dropout(0.3)   [ReLU, L2 reg]
    ↓
Dense 64 + BatchNorm                   [ReLU]
    ↓
Dense 32                               [ReLU]
    ↓
Dense 1 (Sigmoid)                      → P(approved)
```

- **Loss**: Binary Cross-Entropy
- **Optimiser**: Adam (lr=1e-3, with ReduceLROnPlateau)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Total params**: ~48,000

---

## 📊 Typical Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~92% |
| AUC-ROC | ~0.97 |
| Precision | ~93% |
| Recall | ~91% |
| F1 Score | ~92% |

---

## 🖥️ Streamlit App Tabs

| Tab | Features |
|-----|----------|
| 📂 Dataset | Upload/generate data, statistics, approval distribution |
| 📊 EDA | Feature distributions, correlation heatmap, categorical analysis |
| 🧠 Model | Architecture viewer, hyperparameter reference |
| ⚡ Train | Configure and train with live loss/accuracy curves |
| 🔮 Predict | Real-time applicant prediction with risk profile |
| 💻 Code | Full source code for all modules |

---

## 🔧 CLI Training

```bash
python src/train.py \
    --data  data/loan_data.csv \
    --epochs 200 \
    --batch 256 \
    --dropout1 0.4 \
    --dropout2 0.3 \
    --lr 1e-3 \
    --patience 14
```

---

## 📦 Dependencies (Python 3.11)

```
streamlit>=1.32.0
tensorflow==2.15.0
numpy>=1.26.0
pandas>=2.1.0
scikit-learn>=1.4.0
plotly>=5.18.0
joblib>=1.3.0
```

---

## ⚠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| TensorFlow install fails | Try `pip install tensorflow-cpu==2.15.0` |
| Apple Silicon | Use `pip install tensorflow-macos==2.15.0` |
| Slow training | Reduce epochs to 50, increase batch size to 512 |
| Column not found | Check CSV has all required columns listed above |

---

## 📄 License
MIT License — free to use for academic and commercial projects.
