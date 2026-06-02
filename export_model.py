"""
Export model info and feature importance for inspection.
Run: python export_model.py
"""
import os, json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

MODEL_PATH    = "models/xgb_model.json"
FEATURES_PATH = "models/feature_cols.txt"
PARAMS_PATH   = "models/best_params.json"
OUT_PATH      = "models/feature_importance.csv"

def main():
    if not os.path.exists(MODEL_PATH):
        print("No model found. Run train.py first.")
        return

    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    with open(FEATURES_PATH) as f:
        features = [l.strip() for l in f if l.strip()]

    importance = model.feature_importances_
    df = pd.DataFrame({"feature": features, "importance": importance})
    df = df.sort_values("importance", ascending=False)
    df.to_csv(OUT_PATH, index=False)

    print(f"Model: {MODEL_PATH}")
    print(f"Features: {len(features)}")
    print(f"\nTop 15 features:")
    print(df.head(15).to_string(index=False))

    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            params = json.load(f)
        print(f"\nBest CV accuracy: {params.get('accuracy', 'N/A'):.4f}")
        print(f"Params: {params.get('params', {})}")

    print(f"\nFeature importance saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
