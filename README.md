# 🛡️ Credit Card Fraud Detection — End-to-End Machine Learning Project

This project builds an end-to-end machine-learning pipeline to detect fraudulent credit-card transactions.  
Because fraud cases are rare, the dataset is **highly imbalanced** — so the focus is not only accuracy, but improving **recall on fraud cases** while avoiding too many false alarms.

---

## 📂 Project Structure

```
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   └── fraud-analysis-personal.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
│
├── artifacts/
│   ├── train_data.joblib
│   └── scaler.joblib
│
├── models/
│   └── fraud_model.joblib
│
├── requirements.txt
└── README.md
```

> 🔒 Dataset not included due to licensing — download from Kaggle: *Credit Card Fraud Detection*.

---

## 🎯 Objectives

- Understand fraud patterns and transaction behavior  
- Handle severe class imbalance using **SMOTE** and class weights  
- Train and evaluate ML models  
- Build a reusable training + prediction pipeline  
- Save deployable model artifacts

---

## 📊 Dataset

- 284,807 transactions  
- 492 frauds (≈ 0.17%)  
- Features are anonymized (V1–V28) + `Amount` + `Time`

Target variable:

```
Class = 0 → Legitimate
Class = 1 → Fraud
```

---

## 🧰 Tech Stack

- Python
- pandas, numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- joblib
- matplotlib / seaborn

---

## ⚙️ Installation

Clone repository:

```bash
git clone <repo-url>
cd credit-card-fraud-detection
```

Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # Mac / Linux
venv\Scripts\activate         # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place dataset inside:

```
data/creditcard.csv
```

---

## 🚀 Run the pipeline

### 1️⃣ Preprocess data

```bash
python src/data_preprocessing.py
```

Outputs:

- `artifacts/train_data.joblib`
- `artifacts/scaler.joblib`

---

### 2️⃣ Train the model

```bash
python src/train_model.py
```

Saves model to:

```
models/fraud_model.joblib
```

---

### 3️⃣ Make predictions

Use inside Python:

```python
from src.predict import predict

sample = [values here...]
predict(sample)
```

Returns:

```json
{
  "prediction": 0,
  "fraud_probability": 0.0372
}
```

---

## 📈 Model Evaluation

Metrics considered:

- Precision
- Recall (fraud class)
- F1-Score
- ROC-AUC

Update with your trained metrics:

| Metric | Value |
|-------|-------|
| Accuracy |  |
| Recall (Fraud) |  |
| Precision (Fraud) |  |
| ROC-AUC |  |

---

## 🔍 Key Insights

- Class imbalance hurts baseline models
- SMOTE improves recall on fraud cases
- Time and amount-based signals are useful
- Balanced models reduce missed fraud events

---

## ➕ Future Improvements

- XGBoost / LightGBM
- Cost-sensitive learning
- FastAPI / Flask API
- Streamlit dashboard

---

## 👤 Author

**Sachin Patel**

---
