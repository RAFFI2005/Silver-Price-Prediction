"""
train.py
Trains and compares regression models for short-term silver return forecasting.

Uses TimeSeriesSplit cross-validation (never randomly shuffled — that would
leak future information into training, which is the single most common bug
in financial ML projects).

Models compared:
    - Ridge Regression (linear baseline)
    - Random Forest Regressor
    - XGBoost Regressor
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from features import build_features


MODELS = {
    "ridge": Ridge(alpha=1.0),
    "random_forest": RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    ),
    "xgboost": XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1
    ),
}


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of days where predicted return direction matches actual direction.
    More meaningful than R^2 for trading-relevant evaluation."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def evaluate_model(name, model, X, y, n_splits: int = 5) -> dict:
    """Time-series cross-validated evaluation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)

        fold_metrics.append({
            "fold": fold,
            "mae": mean_absolute_error(y_test, preds),
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "r2": r2_score(y_test, preds),
            "directional_accuracy": directional_accuracy(y_test.values, preds),
        })

    avg = {
        "model": name,
        "mae": np.mean([m["mae"] for m in fold_metrics]),
        "rmse": np.mean([m["rmse"] for m in fold_metrics]),
        "r2": np.mean([m["r2"] for m in fold_metrics]),
        "directional_accuracy": np.mean([m["directional_accuracy"] for m in fold_metrics]),
    }
    return avg, fold_metrics


def run_experiment(data_path: str = None,
                    output_dir: str = None) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parent.parent
    if data_path is None:
        data_path = project_root / "data" / "clean_silver.csv"
    if output_dir is None:
        output_dir = project_root / "outputs"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    feats, feature_cols = build_features(df)

    X = feats[feature_cols]
    y = feats["target_return"]

    results = []
    for name, model in MODELS.items():
        print(f"Evaluating {name}...")
        avg, folds = evaluate_model(name, model, X, y)
        results.append(avg)
        print(f"  MAE={avg['mae']:.5f}  RMSE={avg['rmse']:.5f}  "
              f"R2={avg['r2']:.4f}  DirAcc={avg['directional_accuracy']:.3f}")

    results_df = pd.DataFrame(results).sort_values("rmse")
    results_df.to_csv(f"{output_dir}/model_comparison.csv", index=False)

    # Refit best model on full data and save it
    best_name = results_df.iloc[0]["model"]
    best_model = MODELS[best_name]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    best_model.fit(X_scaled, y)

    joblib.dump(best_model, f"{output_dir}/best_model_{best_name}.joblib")
    joblib.dump(scaler, f"{output_dir}/scaler.joblib")
    with open(f"{output_dir}/feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    print(f"\nBest model: {best_name} -> saved to {output_dir}/")
    return results_df


if __name__ == "__main__":
    run_experiment()
