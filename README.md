# Stock Price Prediction using LSTM

A deep learning project that uses Long Short-Term Memory (LSTM) neural networks to predict stock prices. This implementation fetches real-time stock data, trains an LSTM model, and provides predictions with visualization.

## Features

- 📈 Real-time stock data fetching using Yahoo Finance API
- 🧠 LSTM neural network with multiple layers and dropout regularization
- 📊 Comprehensive data preprocessing and normalization
- 📉 Train/test data splitting
- 🎯 Performance metrics (MSE, RMSE, MAE, MAPE)
- 📸 Visualization of predictions vs actual prices
- 📝 Training history visualization

## Architecture

The LSTM model consists of:
- 3 LSTM layers with 50 units each
- Dropout layers (20%) for regularization
- Dense output layer for price prediction
- Adam optimizer with MSE loss function

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- yfinance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rajiv-rathod/Stock-Price-Prediction-using-LSTM.git
cd Stock-Price-Prediction-using-LSTM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script to train and predict stock prices:

```bash
python stock_prediction.py
```

By default, the script predicts Apple (AAPL) stock prices. You can modify the ticker symbol in the `main()` function.

### Customization

You can customize the predictor by modifying parameters:

```python
from stock_prediction import StockPricePredictor

# Initialize with custom parameters
predictor = StockPricePredictor(
    ticker='GOOGL',  # Change stock ticker
    lookback=60      # Number of days to look back
)

# Train with custom parameters
predictor.train(epochs=100, batch_size=64)

# Make predictions
predictions = predictor.predict()

# Calculate metrics
predictor.calculate_metrics(predictions)

# Visualize results
predictor.visualize_training_history()
predictor.visualize_results(predictions)
```

## How It Works

1. **Data Fetching**: Downloads historical stock data from Yahoo Finance
2. **Data Preprocessing**: Normalizes prices using MinMaxScaler
3. **Sequence Creation**: Creates sliding window sequences for LSTM input
4. **Model Training**: Trains the LSTM model on historical data
5. **Prediction**: Generates predictions on test data
6. **Evaluation**: Calculates performance metrics and visualizes results

## Output

The script generates:
- `stock_prediction_results.png`: Visualization of actual vs predicted prices
- `training_history.png`: Training and validation loss over epochs
- Console output with performance metrics

## Model Performance

The model is evaluated using multiple metrics:
- **MSE** (Mean Squared Error): Measures average squared difference
- **RMSE** (Root Mean Squared Error): Square root of MSE
- **MAE** (Mean Absolute Error): Average absolute difference
- **MAPE** (Mean Absolute Percentage Error): Percentage error

## Parameters

- `ticker`: Stock ticker symbol (default: 'AAPL')
- `lookback`: Number of previous days used for prediction (default: 60)
- `epochs`: Number of training epochs (default: 50)
- `batch_size`: Training batch size (default: 32)
- `split_ratio`: Train/test split ratio (default: 0.8)

## Example Results

After training, you'll see output like:

```
==================================================
PERFORMANCE METRICS
==================================================
Mean Squared Error (MSE):  XX.XXXX
Root Mean Squared Error (RMSE): XX.XXXX
Mean Absolute Error (MAE):  XX.XXXX
Mean Absolute Percentage Error (MAPE): XX.XX%
==================================================
```

## Notes

- The model uses the last 60 days of stock prices to predict the next day's price
- Training time depends on the amount of data and number of epochs
- Results may vary based on stock volatility and market conditions
- This is for educational purposes only, not financial advice

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Data provided by Yahoo Finance API
- Built with TensorFlow/Keras
- Inspired by time series forecasting research
