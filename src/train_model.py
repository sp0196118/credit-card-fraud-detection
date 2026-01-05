import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import os


if __name__ == "__main__":
    X, y = joblib.load("artifacts/train_data.joblib")

    model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced_subsample"
    )

    model.fit(X, y)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print(classification_report(y, y_pred))
    print("ROC-AUC:", roc_auc_score(y, y_prob))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fraud_model.joblib")

    print("Model saved.")
