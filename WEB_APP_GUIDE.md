# Stock Price Prediction Web Application

A modern web application powered by LSTM neural networks for predicting stock prices. Built with Flask, TensorFlow, and a beautiful responsive UI.

## 🌟 Features

- **AI-Powered Predictions**: Uses LSTM (Long Short-Term Memory) neural networks for accurate time-series forecasting
- **Real-Time Data**: Fetches live stock data from Yahoo Finance
- **Interactive Web Interface**: Modern, responsive UI with beautiful visualizations
- **Comprehensive Metrics**: MAPE, RMSE, MAE, and directional accuracy
- **Multi-Day Forecasting**: Predict stock prices up to 30 days ahead
- **Customizable Parameters**: Adjust lookback period, training epochs, and forecast horizon
- **REST API**: Full API access for integration with other applications

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rajiv-rathod/Stock-Price-Prediction-using-LSTM.git
   cd Stock-Price-Prediction-using-LSTM
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:5000`

## 📖 Usage

### Web Interface

1. Enter a stock ticker symbol (e.g., AAPL, GOOGL, MSFT)
2. Adjust parameters:
   - **Lookback Period**: Number of historical days to analyze (10-200)
   - **Training Epochs**: Model training iterations (10-100)
   - **Forecast Days**: Number of days to predict ahead (1-30)
3. Click "Generate Prediction"
4. View results including:
   - Historical vs predicted prices
   - Future price forecasts
   - Performance metrics
   - Interactive charts

### API Endpoints

#### Health Check
```bash
GET /api/health
```

#### Get Stock Information
```bash
GET /api/stock-info/<ticker>
```

#### Generate Prediction
```bash
POST /api/predict
Content-Type: application/json

{
  "ticker": "AAPL",
  "lookback": 60,
  "epochs": 50,
  "days_ahead": 5
}
```

**Response:**
```json
{
  "success": true,
  "stock_info": {
    "name": "Apple Inc.",
    "symbol": "AAPL",
    "sector": "Technology"
  },
  "current_price": 175.43,
  "predicted_price": 178.21,
  "price_change": 2.78,
  "price_change_pct": 1.58,
  "metrics": {
    "mse": 52.42,
    "rmse": 7.24,
    "mae": 5.50,
    "mape": 2.60,
    "directional_accuracy": 65.5
  },
  "future_predictions": {
    "dates": ["2025-10-05", "2025-10-06", ...],
    "prices": [178.21, 179.45, ...]
  },
  "plot": "base64_encoded_image"
}
```

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t stock-predictor .
```

### Run Container
```bash
docker run -p 5000:5000 stock-predictor
```

Access at `http://localhost:5000`

## 📊 Model Architecture

The application uses a deep LSTM neural network with the following architecture:

- **Input Layer**: Time-series sequences (configurable lookback window)
- **LSTM Layer 1**: 50 units with dropout (0.2)
- **LSTM Layer 2**: 50 units with dropout (0.2)
- **LSTM Layer 3**: 50 units with dropout (0.2)
- **Output Layer**: Dense layer for price prediction

### Training Process

1. Fetch historical stock data
2. Normalize data using MinMaxScaler
3. Create sliding window sequences
4. Train LSTM model with validation split
5. Generate predictions and calculate metrics

## 📈 Performance Metrics

- **MAPE** (Mean Absolute Percentage Error): Percentage accuracy measure
- **RMSE** (Root Mean Squared Error): Overall prediction error
- **MAE** (Mean Absolute Error): Average absolute error
- **Directional Accuracy**: Percentage of correct up/down predictions

## 🛠️ Technology Stack

### Backend
- **Flask**: Web framework
- **TensorFlow/Keras**: Deep learning models
- **NumPy & Pandas**: Data processing
- **scikit-learn**: Data preprocessing
- **yfinance**: Stock data fetching

### Frontend
- **HTML5/CSS3**: Structure and styling
- **Bootstrap 5**: Responsive UI framework
- **JavaScript**: Interactive functionality
- **Font Awesome**: Icons

## 📁 Project Structure

```
Stock-Price-Prediction-using-LSTM/
├── app.py                      # Flask web application
├── stock_prediction.py         # Core LSTM model (original script)
├── templates/
│   └── index.html             # Web interface
├── static/
│   ├── style.css              # Custom styles
│   └── script.js              # Frontend JavaScript
├── models_cache/              # Cached models (auto-generated)
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── README.md                  # Documentation
└── WEB_APP_GUIDE.md          # This guide
```

## ⚙️ Configuration

### Environment Variables

You can configure the application using environment variables:

```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
export MODEL_CACHE_DIR=./models_cache
```

### Custom Settings

Edit `app.py` to modify:
- Default model parameters
- API rate limits
- Cache settings
- Logging configuration

## 🔒 Deployment Considerations

### Production Settings

For production deployment:

1. **Disable Debug Mode**
   ```python
   app.run(debug=False, host='0.0.0.0', port=5000)
   ```

2. **Use Production Server**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **Add HTTPS**
   Use a reverse proxy (Nginx, Apache) with SSL certificates

4. **Set Environment Variables**
   ```bash
   export FLASK_ENV=production
   export SECRET_KEY=your-secret-key
   ```

### Cloud Deployment Options

#### Heroku
```bash
heroku create stock-predictor-app
git push heroku main
```

#### AWS EC2
1. Launch EC2 instance
2. Install dependencies
3. Configure security groups
4. Run with Gunicorn

#### Google Cloud Run
```bash
gcloud run deploy --source .
```

#### Azure App Service
```bash
az webapp up --name stock-predictor
```

## 🧪 Testing

### Test API Endpoints

```bash
# Health check
curl http://localhost:5000/api/health

# Stock info
curl http://localhost:5000/api/stock-info/AAPL

# Prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "lookback": 60,
    "epochs": 50,
    "days_ahead": 5
  }'
```

### Run Tests

```bash
python test_validation.py
```

## 📝 API Rate Limits

To prevent abuse, consider implementing rate limiting:

```bash
pip install flask-limiter
```

## ⚠️ Disclaimer

**Important**: This application is for educational and research purposes only. Stock price predictions should not be used as the sole basis for investment decisions. Always consult with financial advisors and conduct thorough research before making investment decisions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Rajiv Rathod**
- GitHub: [@rajiv-rathod](https://github.com/rajiv-rathod)

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- Yahoo Finance for providing free stock data
- Bootstrap team for the UI framework
- The open-source community

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check existing documentation
- Review the API documentation

---

**Star ⭐ this repository if you find it helpful!**
