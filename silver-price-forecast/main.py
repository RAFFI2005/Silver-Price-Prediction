"""
main.py
Runs the full pipeline end-to-end:
1. Download + clean Yahoo Finance silver data
2. Build features
3. Train + cross-validate models
4. Generate evaluation plots

Usage:
    python main.py --start 2010-01-01 --ticker SI=F
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_loader import download_silver_data, clean_silver_data
from train import run_experiment
import evaluate


def main():
    parser = argparse.ArgumentParser(description="Silver price regression pipeline")
    parser.add_argument("--ticker", default="SI=F", help="Yahoo Finance ticker")
    parser.add_argument("--start", default="2010-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("STEP 1: Downloading and cleaning data")
    print("=" * 60)
    raw = download_silver_data(
        ticker=args.ticker, start=args.start, end=args.end,
        save_path=data_dir / "raw_silver.csv"
    )
    clean = clean_silver_data(raw)
    clean.to_csv(data_dir / "clean_silver.csv")
    print(f"Clean data: {clean.shape[0]} rows, {clean.index.min()} to {clean.index.max()}")

    print("\n" + "=" * 60)
    print("STEP 2-3: Building features and training models")
    print("=" * 60)
    results = run_experiment(data_path=data_dir / "clean_silver.csv")
    print(results)

    print("\n" + "=" * 60)
    print("STEP 4: Generating evaluation plots")
    print("=" * 60)
    evaluate.main()

    print("\nDone. See outputs/ for results, model file, and plots.")


if __name__ == "__main__":
    main()
