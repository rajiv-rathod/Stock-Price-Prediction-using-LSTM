# 📈 Simple Stock Price Prediction

## What This Does

Upload a CSV file with stock prices → Get AI predictions using LSTM neural network

**That's it. Simple.**

---

## 🚀 Quick Start

1. **Start the server:**
   ```bash
   python simple_app.py
   ```

2. **Open browser:**
   ```
   http://localhost:5000
   ```

3. **Upload CSV and click "Generate Predictions"**

---

## 📋 CSV Format Required

Your CSV must have a column named **"Close"** with stock prices.

### Example CSV:
```csv
Date,Open,High,Low,Close,Volume
2024-01-01,150.00,152.00,149.00,151.00,1000000
2024-01-02,151.00,153.00,150.50,152.50,1100000
2024-01-03,152.50,154.00,152.00,153.50,1200000
...
```

**Note:** Only the "Close" column is required. Other columns are optional.

### Sample Data Included

Use `sample_stock_data.csv` to test the application.

---

## ⚙️ Settings

- **Lookback Period**: How many days to look back (default: 60)
- **Training Epochs**: How long to train the model (default: 50)
- **Predict Days Ahead**: Future predictions to generate (default: 5)

---

## 📊 What You Get

### Metrics:
- **MAPE**: Mean Absolute Percentage Error (lower is better)
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **Data Points**: Total data points in your CSV

### Predictions:
- **Test Set Predictions**: How well the model performed on historical data
- **Future Predictions**: Price predictions for the next N days

---

## 🧠 How It Works

1. **Upload**: CSV file with stock prices
2. **Preprocess**: Normalize data using MinMaxScaler
3. **Train**: 3-layer LSTM neural network
4. **Predict**: Generate predictions on test set + future dates
5. **Display**: Show accuracy metrics and predictions

---

## 🛠️ Technical Details

### Model Architecture:
```
LSTM(50) → Dropout(0.2) →
LSTM(50) → Dropout(0.2) →
LSTM(50) → Dropout(0.2) →
Dense(1)
```

### Training:
- Optimizer: Adam
- Loss: Mean Squared Error
- Train/Test Split: 80/20
- Batch Size: 32

---

## 📦 Requirements

```bash
pip install flask flask-cors tensorflow numpy pandas scikit-learn
```

Or:

```bash
pip install -r requirements.txt
```

---

## 🎯 Use Cases

- **Stock Price Forecasting**
- **Time Series Analysis**
- **ML Learning Project**
- **Quick Predictions**

---

## ⚡ Performance

- Training time: 30-60 seconds (depending on data size and epochs)
- Typical accuracy: 1-5% MAPE on good data
- Works with any time series data (not just stocks)

---

## 🔥 Features

✅ Drag & drop file upload  
✅ Beautiful modern UI  
✅ Real-time training progress  
✅ Interactive results display  
✅ Future predictions  
✅ Accuracy metrics  
✅ No complex setup  
✅ No API keys required  
✅ No database needed  
✅ 100% local processing  

---

## 📝 Notes

- **Minimum data**: Need at least 80 data points (lookback + 20)
- **Best results**: Use 100+ data points with consistent daily data
- **CSV format**: Must have "Close" column
- **Training time**: Increases with more epochs and data points

---

## 🚫 What's NOT Included

This is intentionally simple. Removed:
- ❌ Real-time data fetching
- ❌ Multiple model types
- ❌ Technical indicators
- ❌ Sentiment analysis
- ❌ Advanced features
- ❌ Database storage
- ❌ User authentication
- ❌ Complex configurations

**Why?** To keep it dead simple. Upload CSV → Get predictions.

---

## 🎨 UI Preview

1. **Upload Area**: Drag & drop or click to upload
2. **Settings**: Adjust lookback, epochs, days ahead
3. **Generate Button**: Start training
4. **Loading**: Shows training progress
5. **Results**: Metrics, future predictions, comparison table

---

## 🧪 Testing

1. Use the included `sample_stock_data.csv`
2. Upload it
3. Click "Generate Predictions"
4. Wait 30-60 seconds
5. View results

---

## 💡 Tips

- **More data = better accuracy**
- **Start with 50 epochs**, increase if needed
- **Lookback of 60 days** works well for most stocks
- **Clean data** (no missing values) gives best results

---

## 🔄 Workflow

```
Upload CSV
    ↓
Read Close prices
    ↓
Normalize (MinMaxScaler)
    ↓
Create sequences (lookback window)
    ↓
Split train/test (80/20)
    ↓
Train LSTM model
    ↓
Predict on test set
    ↓
Calculate metrics (MAPE, RMSE, MAE)
    ↓
Predict future days
    ↓
Display results
```

---

## 🎓 For Students

This is a great learning project because:
- **Simple codebase**: Easy to understand
- **Clear workflow**: Each step is visible
- **Real ML**: Uses actual LSTM neural networks
- **Practical**: Solves a real problem
- **Extensible**: Easy to add features

---

## 📞 API Endpoints

### `POST /api/predict`
Upload CSV and get predictions

**Parameters:**
- `file`: CSV file (multipart/form-data)
- `lookback`: Lookback period (default: 60)
- `epochs`: Training epochs (default: 50)
- `days_ahead`: Future predictions (default: 5)

**Response:**
```json
{
  "success": true,
  "metrics": {
    "MAPE": 2.34,
    "RMSE": 3.21,
    "MAE": 2.45,
    "MSE": 10.3
  },
  "predictions": {
    "actual": [...],
    "predicted": [...]
  },
  "future_predictions": [234.5, 235.2, 236.1, ...],
  "training_history": {
    "loss": [...],
    "val_loss": [...]
  }
}
```

### `GET /api/health`
Health check

**Response:**
```json
{
  "status": "healthy",
  "service": "Simple Stock Predictor"
}
```

---

## 🎯 That's It!

No complexity. No confusion. Just:

1. Upload CSV
2. Click button
3. Get predictions

**Simple as that.** 🚀

---

**Last Updated**: October 4, 2025  
**Version**: 1.0 - Simple Edition  
**Status**: ✅ Working
