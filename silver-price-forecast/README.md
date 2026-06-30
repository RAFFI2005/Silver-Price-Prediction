# Silver (XAG/USD) Short-Term Price Movement Forecasting

An independent research project framing short-term silver price movement as a
**regression problem**: predicting the next-day return of silver (via the
`SI=F` COMEX futures proxy on Yahoo Finance) from lagged price action,
technical indicators, and volatility/volume features.

## Project framing

Rather than regressing on raw price (which is misleading — a naive
"tomorrow = today" model scores deceptively well due to autocorrelation),
this project predicts the **next-day percentage return**. Models are
compared on standard regression metrics (MAE, RMSE, R²) as well as
**directional accuracy** (% of days the predicted sign of the move matches
the actual sign), which is the more decision-relevant metric for a
short-term forecasting task.

## Data

- **Source**: [Yahoo Finance](https://finance.yahoo.com/) via the `yfinance` Python library
- **Ticker**: `SI=F` (COMEX silver futures — most liquid, longest-history proxy for XAG/USD)
- **Range**: configurable; default `2010-01-01` to present
- **Fields**: Open, High, Low, Close, Volume (daily)

### Cleaning steps
- Deduplicate and sort by date
- Drop rows with missing close price
- Forward-fill short gaps (≤2 days, e.g. holiday mismatches)
- Drop remaining incomplete rows
- Filter out zero-price/invalid-volume artifacts

## Features

| Category | Features |
|---|---|
| Lagged returns | 1, 2, 3, 5, 10-day lagged returns |
| Trend | SMA(5,10,20,50), price-to-SMA ratio |
| Volatility | 10-day & 20-day rolling std of returns |
| Momentum | RSI(14), 10-day momentum, MACD + signal line |
| Volume | volume % change, 20-day volume z-score |
| Range | intraday high-low range (normalized by close) |

## Models

Three regressors are trained and compared via **time-series cross-validation**
(`TimeSeriesSplit`, 5 folds — never randomly shuffled, to avoid leaking future
data into training, the most common pitfall in financial ML):

1. **Ridge Regression** — linear baseline
2. **Random Forest Regressor**
3. **XGBoost Regressor**

The best model (by RMSE) is refit on the full dataset and saved.

## Project structure

```
silver-price-forecast/
├── data/                    # raw + cleaned data (gitignored, regenerate via pipeline)
├── outputs/                 # trained model, scaler, metrics, plots (gitignored)
├── src/
│   ├── data_loader.py       # Yahoo Finance download + cleaning
│   ├── features.py          # feature engineering + target construction
│   ├── train.py             # model training + time-series CV
│   └── evaluate.py          # diagnostic plots (predicted vs actual, feature importance)
├── main.py                  # runs the full pipeline end-to-end
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/silver-price-forecast.git
cd silver-price-forecast
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the full pipeline (download → clean → feature engineer → train → evaluate):

```bash
python main.py --ticker SI=F --start 2010-01-01
```

Or run each stage independently:

```bash
python src/data_loader.py     # downloads + cleans data -> data/clean_silver.csv
python src/features.py        # builds feature matrix -> data/features_silver.csv
python src/train.py           # trains + cross-validates models -> outputs/
python src/evaluate.py        # generates plots -> outputs/
```

## Results

After running, see `outputs/model_comparison.csv` for cross-validated metrics
per model, and `outputs/predictions_vs_actual.png` /
`outputs/feature_importance.png` for diagnostics.

> Note: short-term financial returns are dominated by noise; expect R² values
> near zero and directional accuracy modestly above 50% for a well-behaved
> model. This is expected and consistent with weak-form market efficiency —
> the value of this project is in the rigor of the pipeline (correct
> time-series validation, leakage-free features, honest target definition)
> rather than in claiming reliable alpha.

## Limitations & future work

- Single-asset, daily-frequency only — no cross-asset (gold, DXY, real yields) features yet
- No transaction-cost or position-sizing backtest layer
- Could extend to multi-horizon forecasting (3-day, 5-day returns)
- Could add macro features (CPI surprises, Fed rate decisions, ETF flows)

## License

MIT
