import joblib
import numpy as np


model = joblib.load("models/fraud_model.joblib")
scaler = joblib.load("artifacts/scaler.joblib")


def predict(transaction_list):
    arr = np.array(transaction_list).reshape(1, -1)
    arr_scaled = scaler.transform(arr)

    prob = model.predict_proba(arr_scaled)[0][1]
    pred = model.predict(arr_scaled)[0]

    return {
        "prediction": int(pred),
        "fraud_probability": round(float(prob), 4)
    }
