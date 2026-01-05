import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os


def load_data(path):
    return pd.read_csv(path)


def preprocess(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    return X_resampled, y_resampled, scaler


if __name__ == "__main__":
    df = load_data("data/creditcard.csv")

    X, y, scaler = preprocess(df)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump((X, y), "artifacts/train_data.joblib")
    joblib.dump(scaler, "artifacts/scaler.joblib")

    print("Data preprocessing complete.")
