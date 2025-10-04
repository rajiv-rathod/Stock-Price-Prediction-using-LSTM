# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/rajiv-rathod/Stock-Price-Prediction-using-LSTM.git
cd Stock-Price-Prediction-using-LSTM

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### Run with default settings (Apple stock)

```bash
python stock_prediction.py
```

### Customize in code

```python
from stock_prediction import StockPricePredictor

# Create predictor for a different stock
predictor = StockPricePredictor(ticker='GOOGL', lookback=60)

# Train the model
predictor.train(epochs=50, batch_size=32)

# Make predictions
predictions = predictor.predict()

# Evaluate
predictor.calculate_metrics(predictions)

# Visualize
predictor.visualize_training_history()
predictor.visualize_results(predictions)
```

## Popular Stock Tickers

- **AAPL** - Apple Inc.
- **GOOGL** - Alphabet Inc. (Google)
- **MSFT** - Microsoft Corporation
- **TSLA** - Tesla, Inc.
- **AMZN** - Amazon.com, Inc.
- **META** - Meta Platforms, Inc. (Facebook)
- **NVDA** - NVIDIA Corporation
- **JPM** - JPMorgan Chase & Co.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| ticker | 'AAPL' | Stock ticker symbol |
| lookback | 60 | Days to look back for prediction |
| epochs | 50 | Training epochs |
| batch_size | 32 | Training batch size |
| split_ratio | 0.8 | Train/test split ratio |

## Output Files

- `stock_prediction_results.png` - Prediction visualization
- `training_history.png` - Training loss plot

## Tips

1. **More epochs** = Better training but takes longer
2. **Larger lookback** = More historical context but slower
3. **Different stocks** have different volatility patterns
4. Results are for **educational purposes only**

## Troubleshooting

### Installation Issues

If you encounter installation errors:

```bash
# Try upgrading pip first
pip install --upgrade pip

# Install packages one by one
pip install numpy pandas matplotlib scikit-learn
pip install tensorflow
pip install yfinance
```

### Memory Issues

Reduce batch size or number of epochs:

```python
predictor.train(epochs=25, batch_size=16)
```

### Data Fetching Issues

Check your internet connection. Yahoo Finance requires internet access.

## Next Steps

1. Try different stocks
2. Experiment with hyperparameters
3. Compare predictions across multiple stocks
4. Extend the model with additional features (volume, indicators, etc.)
