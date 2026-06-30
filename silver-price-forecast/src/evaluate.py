"""
evaluate.py
Generates diagnostic plots for the trained model: predicted vs actual returns,
residuals over time, and feature importance (for tree-based models).
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from features import build_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def plot_predictions_vs_actual(y_true, y_pred, dates, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(dates, y_true, label="Actual return", alpha=0.7)
    axes[0].plot(dates, y_pred, label="Predicted return", alpha=0.7)
    axes[0].set_title("Predicted vs Actual Next-Day Return")
    axes[0].legend()
    axes[0].set_ylabel("Return")

    residuals = y_true - y_pred
    axes[1].plot(dates, residuals, color="firebrick", alpha=0.6)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Residuals Over Time")
    axes[1].set_ylabel("Residual")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_feature_importance(model, feature_cols, save_path, top_n=15):
    if not hasattr(model, "feature_importances_"):
        print("Model has no feature_importances_ attribute (e.g. Ridge). Skipping.")
        return

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot: {save_path}")


def main():
    outputs_dir = PROJECT_ROOT / "outputs"
    data_path = PROJECT_ROOT / "data" / "clean_silver.csv"

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    feats, feature_cols = build_features(df)
    X = feats[feature_cols]
    y = feats["target_return"]

    # Load the best saved model + scaler
    model_files = list(outputs_dir.glob("best_model_*.joblib"))
    if not model_files:
        raise FileNotFoundError("No trained model found. Run train.py first.")
    model = joblib.load(model_files[0])
    scaler = joblib.load(outputs_dir / "scaler.joblib")

    # Evaluate on the final held-out fold (most recent data)
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    dates_test = feats.index[test_idx]

    X_test_s = scaler.transform(X_test)
    preds = model.predict(X_test_s)

    plot_predictions_vs_actual(
        y_test.values, preds, dates_test,
        outputs_dir / "predictions_vs_actual.png"
    )
    plot_feature_importance(
        model, feature_cols, outputs_dir / "feature_importance.png"
    )


if __name__ == "__main__":
    main()
