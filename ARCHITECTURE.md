# Model Architecture and Design

## Overview

This project implements a Long Short-Term Memory (LSTM) neural network for time series forecasting of stock prices. The architecture is designed to capture temporal dependencies in historical price data.

## Network Architecture

### Layer Structure

```
Input Layer: (lookback_period, 1)
    ↓
LSTM Layer 1: 50 units, return_sequences=True
    ↓
Dropout: 0.2
    ↓
LSTM Layer 2: 50 units, return_sequences=True
    ↓
Dropout: 0.2
    ↓
LSTM Layer 3: 50 units, return_sequences=False
    ↓
Dropout: 0.2
    ↓
Dense Output: 1 unit (predicted price)
```

### Key Components

#### 1. LSTM Layers
- **Units**: 50 per layer
- **Purpose**: Learn temporal patterns and dependencies
- **Return Sequences**: First two layers return sequences for stacking

#### 2. Dropout Layers
- **Rate**: 0.2 (20%)
- **Purpose**: Prevent overfitting by randomly dropping connections
- **Placement**: After each LSTM layer

#### 3. Dense Output Layer
- **Units**: 1
- **Activation**: None (linear)
- **Purpose**: Output the predicted price

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| LSTM Units | 50 | Number of neurons per LSTM layer |
| Dropout Rate | 0.2 | Percentage of connections to drop |
| Optimizer | Adam | Adaptive learning rate optimizer |
| Loss Function | MSE | Mean Squared Error |
| Lookback Period | 60 | Days of historical data to consider |
| Batch Size | 32 | Number of samples per training batch |
| Epochs | 50 | Number of complete passes through data |

## Data Pipeline

### 1. Data Fetching
```python
# Fetches historical stock data from Yahoo Finance
data = yf.download(ticker, start_date, end_date)
```

### 2. Preprocessing
```python
# Extract closing prices
prices = data['Close'].values

# Normalize to [0, 1] range
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)
```

### 3. Sequence Creation
```python
# Create sliding windows of lookback_period days
for i in range(lookback, len(data)):
    X.append(data[i-lookback:i])  # Input: 60 days
    y.append(data[i])              # Target: next day
```

### 4. Train/Test Split
- **Training Set**: 80% of data
- **Testing Set**: 20% of data
- **Validation**: 10% of training data

## Training Process

### 1. Forward Pass
- Input sequence → LSTM layers → Output prediction
- Each LSTM cell maintains hidden state and cell state

### 2. Loss Calculation
- Mean Squared Error between predicted and actual prices
- Formula: `MSE = (1/n) * Σ(y_pred - y_actual)²`

### 3. Backpropagation
- Gradient calculation through time (BPTT)
- Adam optimizer adjusts weights

### 4. Validation
- 10% of training data used for validation
- Early stopping can be implemented if needed

## Prediction Process

1. **Input Preparation**: Take last 60 days of scaled prices
2. **Prediction**: Feed through trained network
3. **Inverse Scaling**: Convert back to actual price range
4. **Output**: Next day's predicted price

## Performance Metrics

### 1. Mean Squared Error (MSE)
- Measures average squared difference
- Formula: `MSE = (1/n) * Σ(y_pred - y_actual)²`
- Lower is better

### 2. Root Mean Squared Error (RMSE)
- Square root of MSE
- Same unit as price (USD)
- Formula: `RMSE = √MSE`

### 3. Mean Absolute Error (MAE)
- Average absolute difference
- Formula: `MAE = (1/n) * Σ|y_pred - y_actual|`
- More interpretable than MSE

### 4. Mean Absolute Percentage Error (MAPE)
- Percentage error metric
- Formula: `MAPE = (1/n) * Σ|(y_actual - y_pred)/y_actual| * 100`
- Good for comparing across different stocks

## Design Decisions

### Why LSTM?
- **Memory**: Can remember long-term dependencies
- **Gating**: Forget gate, input gate, output gate control information flow
- **Sequential**: Perfect for time series data

### Why 3 Layers?
- **Layer 1**: Learns basic patterns
- **Layer 2**: Learns complex relationships
- **Layer 3**: Integrates information for final decision
- Balance between complexity and overfitting

### Why Dropout?
- **Regularization**: Prevents overfitting
- **Ensemble Effect**: Each training batch uses different network
- **Rate 0.2**: Standard starting point

### Why 60-Day Lookback?
- **Quarterly Pattern**: ~3 months of trading data
- **Balance**: Not too short (misses patterns) or too long (computation)
- **Standard**: Commonly used in financial analysis

## Limitations

1. **Market Complexity**: Cannot capture all market factors
2. **External Events**: News, policies, etc. not included
3. **Black Swan Events**: Unexpected crashes hard to predict
4. **Stationarity**: Assumes patterns will continue
5. **Overfitting Risk**: May memorize training data

## Future Improvements

1. **Multi-feature Input**: Add volume, RSI, MACD, etc.
2. **Attention Mechanism**: Focus on important time steps
3. **Ensemble Methods**: Combine multiple models
4. **Bidirectional LSTM**: Look forward and backward
5. **Hyperparameter Tuning**: Grid search for optimal values
6. **Transfer Learning**: Train on multiple stocks
7. **Sentiment Analysis**: Incorporate news and social media

## References

- Hochreiter & Schmidhuber (1997): LSTM original paper
- Yahoo Finance API for data
- Keras/TensorFlow documentation
- Financial time series forecasting literature
