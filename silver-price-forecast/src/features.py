"""
Feature engineering for silver price regression.

Predicts NEXT-DAY RETURN (not raw price). Predicting raw price directly
is misleading for time series — a naive "predict tomorrow = today" model
scores deceptively well (R^2 near 1) because price is highly autocorrelated.
Forecasting returns is the honest version of "predicting short-term movement."
"""

import pandas as pd
import numpy as np


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_features(df: pd.DataFrame, target_horizon: int = 1) -> pd.DataFrame:
    """
    Build technical/statistical features and the regression target from
    cleaned OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned OHLCV data with columns Open, High, Low, Close, Volume.
    target_horizon : int
        Number of trading days ahead to forecast (default 1 = next day).

    Returns
    -------
    pd.DataFrame
        Feature matrix + target column 'target_return', NaN rows dropped.
    """
    out = df.copy()
    close = out["Close"]

    # --- Returns (lagged, the core autoregressive signal) ---
    out["return_1d"] = close.pct_change(1)
    for lag in [2, 3, 5, 10]:
        out[f"return_lag_{lag}"] = out["return_1d"].shift(lag - 1)

    # --- Moving averages & relative position ---
    for w in [5, 10, 20, 50]:
        out[f"sma_{w}"] = close.rolling(w).mean()
        out[f"close_to_sma_{w}"] = close / out[f"sma_{w}"] - 1

    # --- Volatility ---
    out["volatility_10d"] = out["return_1d"].rolling(10).std()
    out["volatility_20d"] = out["return_1d"].rolling(20).std()

    # --- Momentum / oscillators ---
    out["rsi_14"] = _rsi(close, 14)
    out["momentum_10d"] = close / close.shift(10) - 1

    # --- Range / volume based ---
    out["high_low_range"] = (out["High"] - out["Low"]) / close
    out["volume_change"] = out["Volume"].pct_change(1)
    out["volume_zscore_20d"] = (
        out["Volume"] - out["Volume"].rolling(20).mean()
    ) / out["Volume"].rolling(20).std()

    # --- MACD ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    # --- Target: forward return over target_horizon days ---
    out["target_return"] = close.shift(-target_horizon) / close - 1

    feature_cols = [c for c in out.columns if c not in
                    ["Open", "High", "Low", "Close", "Volume", "target_return"]]

    # RSI and z-score calcs can produce inf when a rolling denominator is 0
    # (e.g. flat price/volume periods). Treat inf as missing, then drop.
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan)

    out = out.dropna(subset=feature_cols + ["target_return"])

    return out, feature_cols


    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    df = pd.read_csv(project_root / "data" / "clean_silver.csv", index_col=0, parse_dates=True)
    feats, cols = build_features(df)
    print(f"Feature matrix shape: {feats.shape}")
    print(f"Number of features: {len(cols)}")
    feats.to_csv(project_root / "data" / "features_silver.csv")
