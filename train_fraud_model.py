"""
Trains a RandomForest fraud classifier on creditcard.csv and saves
the model + supporting files that app.py needs to serve predictions.

RUN THIS ONCE (or whenever you change the data/model). It creates:
    model/fraud_model.pkl
    model/feature_columns.pkl
    model/best_threshold.pkl

Usage:
    python train_fraud_model.py
"""

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

DATA_FILE = "creditcard.csv"
MODEL_DIR = "model"


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"'{DATA_FILE}' not found in this folder. "
            "Place your dataset here and run again."
        )

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load dataset
    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)

    # 2. Features / target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # 3. Train/test split (stratified so fraud ratio is preserved in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Model — kept light on purpose so training doesn't hang a laptop
    print("Training RandomForest (this can take 1-3 minutes)...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=2
    )
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    print("\nConfusion Matrix (threshold=0.50):")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))

    # 6. Sweep thresholds to find the best F1
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    results = []
    for t in thresholds:
        pred_t = (y_prob >= t).astype(int)
        results.append({
            "threshold": t,
            "precision": precision_score(y_test, pred_t, zero_division=0),
            "recall": recall_score(y_test, pred_t, zero_division=0),
            "f1": f1_score(y_test, pred_t, zero_division=0)
        })
    results_df = pd.DataFrame(results)
    print("\nThreshold sweep:")
    print(results_df)

    best = results_df.loc[results_df["f1"].idxmax()]
    print("\nBest threshold:")
    print(best)

    # 7. Save everything the API needs
    joblib.dump(model, os.path.join(MODEL_DIR, "fraud_model.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "feature_columns.pkl"))
    joblib.dump(float(best["threshold"]), os.path.join(MODEL_DIR, "best_threshold.pkl"))

    print(f"\nSaved model files to '{MODEL_DIR}/' — you can now run app.py")


if __name__ == "__main__":
    main()
