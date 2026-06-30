"""
data_loader.py
Downloads and cleans historical XAG/USD (silver) price data from Yahoo Finance.

Yahoo Finance ticker for spot silver: "SI=F" (COMEX silver futures, most liquid proxy)
Alternative: "XAGUSD=X" (spot FX-style quote, thinner history/volume data)
"""

import pandas as pd
import yfinance as yf
from pathlib import Path


def download_silver_data(
    ticker: str = "SI=F",
    start: str = "2010-01-01",
    end: str | None = None,
    save_path: str | Path | None = "data/raw_silver.csv",
) -> pd.DataFrame:
    """
    Download historical OHLCV data for silver from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol. Default "SI=F" (COMEX silver futures).
    start : str
        Start date in YYYY-MM-DD format.
    end : str or None
        End date in YYYY-MM-DD format. None = today.
    save_path : str or None
        If provided, saves the raw downloaded data to this CSV path.

    Returns
    -------
    pd.DataFrame
        Raw OHLCV data indexed by date.
    """
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. Check ticker symbol or date range."
        )

    # yfinance sometimes returns MultiIndex columns when a single ticker is passed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path)
        print(f"Saved raw data to {save_path} ({len(df)} rows)")

    return df


def clean_silver_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw OHLCV data:
    - Drop rows with missing Close prices
    - Forward-fill small gaps (e.g. holiday mismatches) up to 2 days
    - Drop any remaining NaNs
    - Ensure sorted, deduplicated date index

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned OHLCV dataframe.
    """
    df = df.copy()
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    df = df.dropna(subset=["Close"])
    df[required_cols] = df[required_cols].ffill(limit=2)
    df = df.dropna(subset=required_cols)

    # Remove non-trading rows (zero volume or zero price artifacts)
    df = df[(df["Close"] > 0) & (df["Volume"] >= 0)]

    return df


if __name__ == "__main__":
    raw = download_silver_data(start="2010-01-01")
    clean = clean_silver_data(raw)
    clean.to_csv("data/clean_silver.csv")
    print(f"Clean data shape: {clean.shape}")
    print(clean.tail())
