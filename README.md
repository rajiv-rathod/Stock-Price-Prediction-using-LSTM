# 🚀 Advanced Stock Price Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1.2-green.svg)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Ensemble-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

A professional-grade stock price prediction platform powered by advanced machine learning ensemble methods, featuring a modern dark mode interface and comprehensive technical analysis.

![Platform Preview](https://via.placeholder.com/800x400/0a0e27/00d4ff?text=Professional+Stock+Prediction+Platform)

## ✨ Features

### 🤖 Advanced Machine Learning
- **7-Model Ensemble System**: XGBoost, LightGBM, CatBoost, Random Forest, Gradient Boosting, Elastic Net, SVR
- **50+ Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, Williams %R, and more
- **Intelligent Feature Engineering**: Volatility analysis, momentum indicators, trend detection
- **Weighted Predictions**: Ensemble averaging based on individual model performance (R² scores)

### 📊 Professional Interface
- **Modern Dark Mode Design**: Trading platform aesthetics with professional color scheme
- **Interactive Charts**: Real-time Chart.js visualizations with customizable parameters
- **Responsive Layout**: Optimized for desktop and mobile devices
- **Real-time Progress**: Live training progress and performance metrics

### 📈 Smart Data Processing
- **Universal CSV Support**: Automatically detects and processes any CSV format
- **Intelligent Column Detection**: Finds price columns regardless of naming convention
- **Data Cleaning Pipeline**: Handles missing values, outliers, and data type conversion
- **Robust Error Handling**: Comprehensive validation and user-friendly error messages

### 🎯 Key Capabilities
- **Future Price Forecasting**: Predict stock prices for 1-30 days ahead
- **Model Performance Metrics**: MAPE, RMSE, R² scores for accuracy assessment
- **Ensemble Comparison**: Side-by-side model performance analysis
- **Technical Analysis**: Comprehensive indicator calculations and visualizations

## 🛠️ Technology Stack

### Backend
- **Flask 3.1.2**: Modern Python web framework with CORS support
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Core machine learning algorithms
- **XGBoost 3.0.5**: Gradient boosting framework
- **LightGBM 4.6.0**: Fast gradient boosting
- **CatBoost 1.2.8**: Categorical feature handling
- **TA-Lib**: Technical analysis library

### Frontend
- **Chart.js**: Interactive and responsive charts
- **Modern CSS**: CSS Grid, Flexbox, custom properties
- **Professional UI**: Dark theme with trading platform aesthetics
- **Responsive Design**: Mobile-first approach

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 500MB for dependencies
- **OS**: Windows, macOS, or Linux

### Python Dependencies
```txt
Flask==3.1.2
flask-cors==4.0.0
pandas==2.1.1
numpy==1.24.3
scikit-learn==1.3.0
xgboost==3.0.5
lightgbm==4.6.0
catboost==1.2.8
scipy==1.11.2
ta==0.11.0
yfinance==0.2.28
plotly==5.17.0
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/rajiv-rathod/Stock-Price-Prediction-using-LSTM.git
cd Stock-Price-Prediction-using-LSTM
```

### 2. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually:
pip install Flask flask-cors pandas numpy scikit-learn
pip install xgboost lightgbm catboost scipy ta yfinance plotly
```

### 3. Launch the Application
```bash
python app.py
```

### 4. Access the Platform
Open your web browser and navigate to:
```
http://localhost:5000
```

## 📖 Usage Guide

### Step 1: Prepare Your Data
- **Format**: CSV file with stock price data
- **Required Column**: Any price column (Close, Price, Adj Close, etc.)
- **Optional Columns**: Open, High, Low, Volume, Date
- **Example**:
```csv
Date,Open,High,Low,Close,Volume
2024-01-01,150.00,152.00,149.00,151.00,1000000
2024-01-02,151.00,153.00,150.50,152.50,1100000
```

### Step 2: Upload and Configure
1. **Upload CSV**: Drag & drop or click to select your file
2. **Set Parameters**:
   - **Lookback Period**: Historical data window (10-200 days)
   - **Training Intensity**: Model complexity (10-100 epochs)
   - **Forecast Days**: Prediction horizon (1-30 days)

### Step 3: Generate Predictions
1. Click **"Generate Advanced Predictions"**
2. Wait for ensemble training (30-60 seconds)
3. View comprehensive results and forecasts

### Step 4: Analyze Results
- **Model Performance**: Compare ensemble vs individual models
- **Accuracy Metrics**: MAPE, RMSE, R² scores
- **Future Forecasts**: Interactive charts and price predictions
- **Technical Analysis**: Comprehensive indicator visualization

## 🏗️ Architecture

### Machine Learning Pipeline
```
CSV Data → Data Cleaning → Feature Engineering → Model Training → Ensemble Prediction
    ↓            ↓              ↓                ↓                ↓
Raw Data → Normalized → 50+ Indicators → 7 Models → Weighted Average
```

### Model Ensemble
1. **XGBoost**: Gradient boosting with tree-based learners
2. **LightGBM**: Memory-efficient gradient boosting
3. **CatBoost**: Handles categorical features automatically
4. **Random Forest**: Ensemble of decision trees
5. **Gradient Boosting**: Sequential error correction
6. **Elastic Net**: Regularized linear regression
7. **SVR**: Support Vector Regression

### Feature Engineering
- **Trend Indicators**: SMA, EMA, MACD signals
- **Momentum Oscillators**: RSI, Stochastic, Williams %R
- **Volatility Measures**: Bollinger Bands, ATR
- **Volume Analysis**: OBV, Volume SMA
- **Price Patterns**: Rate of Change, Price Position

## 📊 Performance Metrics

### Accuracy Measures
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based accuracy
- **RMSE (Root Mean Square Error)**: Prediction variance measure
- **R² Score**: Coefficient of determination (model fit quality)
- **MAE (Mean Absolute Error)**: Average prediction error

### Model Comparison
The platform automatically selects the best-performing model based on R² scores and provides ensemble predictions for maximum accuracy.

## 🔧 Configuration

### Model Parameters
Customize model behavior in `app.py`:

```python
# XGBoost Configuration
'XGBoost': xgb.XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

### Feature Engineering
Add custom indicators in `create_advanced_features()`:

```python
# Custom Technical Indicator
data['Custom_Indicator'] = your_custom_calculation(data['Close'])
```

## 🚨 Troubleshooting

### Common Issues

**Issue**: "CSV loading failed"
- **Solution**: Ensure CSV has proper price column (Close, Price, etc.)

**Issue**: "Not enough data points"
- **Solution**: Minimum 80 data points required (lookback + 20)

**Issue**: "Model training failed"
- **Solution**: Check data quality and reduce lookback period

**Issue**: "Memory error during training"
- **Solution**: Reduce dataset size or close other applications

### Performance Optimization
- **Large Datasets**: Use data sampling for faster processing
- **Memory Usage**: Reduce ensemble size or feature count
- **Training Speed**: Lower n_estimators in model configurations

## 📁 Project Structure

```
Stock-Price-Prediction-using-LSTM/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── README.md              # This documentation
├── LICENSE                # MIT license
├── templates/             # HTML templates
│   └── advanced_index.html # Main interface
├── static/               # Static assets
│   ├── style.css        # CSS styling
│   └── script.js        # JavaScript functionality
└── uploads/             # Temporary file storage
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comments for complex algorithms
- Test with various CSV formats
- Update documentation for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Scikit-learn**: Machine learning framework
- **XGBoost Team**: Gradient boosting library
- **Chart.js**: Interactive charting library
- **Flask Community**: Web framework development
- **TA-Lib**: Technical analysis indicators

## 📧 Support

For support, questions, or feature requests:

- **Issues**: [GitHub Issues](https://github.com/rajiv-rathod/Stock-Price-Prediction-using-LSTM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rajiv-rathod/Stock-Price-Prediction-using-LSTM/discussions)

## 🔮 Future Enhancements

- [ ] Real-time data integration with APIs
- [ ] Advanced deep learning models (LSTM, Transformer)
- [ ] Portfolio optimization features
- [ ] Email alerts for predictions
- [ ] Mobile application
- [ ] Cloud deployment options

---

**⚡ Built with passion for accurate stock prediction and professional trading tools.**

*Made by [Rajiv Rathod](https://github.com/rajiv-rathod) | Star ⭐ this repo if you found it helpful!*

## Features

### Core AI/ML
- Ultra-Advanced Models - Transformer + LSTM ensemble with 50+ features
- Deep Learning LSTM - 3-layer neural network for time series forecasting
- Technical Indicators - RSI, MACD, Bollinger Bands, Moving Averages (19 indicators)
- Sentiment Analysis - News and social media integration
- Real-time Data - Live stock data from Yahoo Finance API
- Custom Data Support - Upload your own CSV files (OHLCV format)
- Advanced Metrics - MAPE, RMSE, MAE, and directional accuracy
- Multi-day Forecasting - Predict up to 30 days ahead
- Confidence Scoring - Prediction reliability indicators

### Web Application
- Professional Trading Dashboard - Real-time ticker tape, candlestick charts, and metrics
- Modern UI - Responsive dark-themed design with Bootstrap 5
- Company Search - Real-time autocomplete ticker search
- Interactive Charts - TradingView-style visualization with ApexCharts
- Dual Input Modes - Ticker search or CSV upload
- Quick Select - Popular stocks shortcuts (AAPL, GOOGL, MSFT, etc.)
- Mobile-Friendly - Works on all devices
- Real-Time Updates - Live price ticker and streaming data

### Developer Features
- REST API - Full API access for integration
- Docker Support - Containerized deployment ready
- Comprehensive Docs - 7+ documentation guides
- Error Handling - Graceful failure management
- Logging - Request/error tracking
- CORS Enabled - Cross-origin resource sharing

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB RAM (8GB recommended)
- Internet connection for real-time data

### Installation

```bash
# Clone the repository
git clone https://github.com/rajiv-rathod/Stock-Price-Prediction-using-LSTM.git
cd Stock-Price-Prediction-using-LSTM

# Install dependencies
pip install -r requirements.txt

# Run the web application
python app.py
```

**Access the app at:** http://localhost:5000

### Using Docker

```bash
# Build and run with Docker Compose
docker-compose up

# Or with Docker directly
docker build -t stock-predictor .
docker run -p 5000:5000 stock-predictor
```

## Usage

1. **Upload CSV**
   - Click or drag a CSV file with price data
   - The system automatically detects the price column (Close, Price, etc.)

2. **Configure Parameters**
   - Lookback period (days to analyze)
   - Training epochs
   - Days ahead to predict

3. **Get Predictions**
   - View performance metrics (MAPE, RMSE, MAE)
   - See interactive charts of actual vs predicted prices
   - Check future price forecasts with trend visualization

4. **API Usage**
   - POST to `/api/predict` with CSV file and parameters
   - Returns JSON with predictions, metrics, and chart data
   - ⚡ **Advanced Mode**: Bidirectional LSTM + Technical indicators + Ensemble
   - Standard Mode: Classic LSTM architecture (Fastest training)

4. **Interactive Charts**
   - TradingView-style candlestick charts
   - Real-time price predictions with confidence bands
   - Technical indicator overlays (RSI, MACD, Moving Averages)
   - Volume analysis and distribution charts

5. **Key Metrics Dashboard**
   - Current vs. Predicted price comparison
   - Model accuracy and confidence scores
   - Detailed performance metrics (MAE, RMSE, MAPE)
   - Directional accuracy percentage

### Web Interface

1. **Search for a Stock**
   - Type company name or ticker symbol in the search box
   - Select from autocomplete suggestions
   - Or use quick-select buttons for popular stocks

2. **Configure Parameters**
   - Lookback Period: 10-200 days (historical data window)
   - Training Epochs: 10-100 (model training iterations)
   - Forecast Days: 1-30 (future prediction range)

3. **Generate Prediction**
   - Click "Generate Prediction" button
   - Wait 30-60 seconds for model training
   - View comprehensive results with charts and metrics

4. **Upload Custom Data**
   - Switch to "Upload CSV" tab
   - Select your CSV file (must have Date, Close columns)
   - Configure parameters
   - Generate predictions on your data

### API Usage

```python
import requests

# Get prediction
response = requests.post('http://localhost:5000/api/predict', json={
    'ticker': 'AAPL',
    'lookback': 60,
    'epochs': 50,
    'days_ahead': 5
})

data = response.json()
if data['success']:
    result = data['data']
    print(f"Current Price: ${result['current_price']:.2f}")
    print(f"Predicted Price: ${result['predicted_price']:.2f}")
    print(f"Change: {result['price_change_pct']:.2f}%")
    print(f"MAPE: {result['metrics']['mape']:.2f}%")
```

```python
# Search companies
response = requests.get('http://localhost:5000/api/search_companies?q=apple')
companies = response.json()['data']
for company in companies:
    print(f"{company['symbol']}: {company['name']}")
```

## API Endpoints

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/` | GET | Web interface | - |
| `/api/predict` | POST | Upload CSV and get predictions | Form data with file, lookback, epochs, days_ahead |

### Prediction Request (Form Data)
- `file`: CSV file with price data
- `lookback`: int (10-200, default 60)
- `epochs`: int (10-200, default 50)
- `days_ahead`: int (1-30, default 5)

### Response
```json
{
  "success": true,
  "data_points": 1000,
  "lookback": 60,
  "epochs": 50,
  "metrics": {
    "MSE": 1.23,
    "RMSE": 1.11,
    "MAE": 0.89,
    "MAPE": 2.34
  },
  "predictions": {
    "actual": [100.0, 101.0, ...],
    "predicted": [99.8, 100.9, ...]
  },
  "future_predictions": [102.0, 103.5, ...]
}
```

## 🏗️ Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│    Frontend     │─────▶│   Flask API      │─────▶│  LSTM Model     │
│  (HTML/JS/CSS)  │      │   (Python)       │      │  (TensorFlow)   │
└─────────────────┘      └──────────────────┘      └─────────────────┘
        │                        │                          │
        │                        ▼                          ▼
        │                 ┌──────────────────┐      ┌─────────────────┐
        │                 │  Yahoo Finance   │      │  Training Data  │
        └────────────────▶│      API         │      │  Preprocessing  │
                          └──────────────────┘      └─────────────────┘
```

**Model Architecture:**
- Input Layer: Sequential data (lookback window)
- LSTM Layer 1: 50 units with dropout (0.2)
- LSTM Layer 2: 50 units with dropout (0.2)
- LSTM Layer 3: 50 units with dropout (0.2)
- Output Layer: Dense (1 unit) - Price prediction
- Optimizer: Adam
- Loss: Mean Squared Error (MSE)

## Project Structure

```
Stock-Price-Prediction-using-LSTM/
├── app.py                      # Flask web application
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
├── docker-compose.yml          # Multi-container orchestration
├── LICENSE                     # MIT license
├── README.md                   # This file
├── templates/
│   ├── simple_index.html       # Main web interface
│   └── layout.html             # Base template
├── static/
│   ├── script.js               # Frontend JavaScript
│   └── style.css               # UI styling
└── test_api.py                 # API tests
```

## Performance

### Model Metrics (Typical):
- **MAPE**: 2-5% on test data
- **RMSE**: 5-10 price points
- **MAE**: 4-8 price points
- **Directional Accuracy**: 60-70%
- **Training Time**: 30-60 seconds (CPU)
- **Prediction Speed**: <1 second

### Application Performance:
- **API Response**: <50ms (health check)
- **Company Search**: <200ms
- **Full Prediction**: 30-60 seconds (includes model training)
- **Concurrent Users**: 10+ (Flask development server)

## Deployment

Deploy to your favorite platform:

### Heroku
```bash
heroku login
heroku create your-app-name
git push heroku main
```

### AWS EC2
```bash
# SSH into EC2 instance
git clone <repo-url>
cd Stock-Price-Prediction-using-LSTM
./start_web_app.sh
```

### Google Cloud Run
```bash
gcloud run deploy stock-predictor \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Docker on Any Platform
```bash
docker-compose up -d
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## 🧪 Testing

```bash
# Run API tests
python test_api.py

# Run model validation
python test_validation.py

# Manual testing
curl http://localhost:5000/api/health
```

## Tech Stack

**Backend:**
- Python 3.8+
- Flask - Web framework
- TensorFlow/Keras - Deep learning
- NumPy, Pandas - Data processing
- scikit-learn - Preprocessing & metrics

**Frontend:**
- HTML5/CSS3
- JavaScript
- Bootstrap - UI framework
- Chart.js - Interactive charts

**DevOps:**
- Docker & Docker Compose
- Git/GitHub

### Phase 2: Model Enhancements (⏳ In Progress)
- [ ] Baseline models (ARIMA, RandomForest, XGBoost)
- [ ] Technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Advanced LSTM variants (GRU, Bidirectional, Attention)
- [ ] Ensemble models
- [ ] Hyperparameter tuning
- [ ] Model versioning

### Phase 3: Advanced Features (📋 Planned)
- [ ] Multi-stock portfolio prediction
- [ ] Sentiment analysis from news/social media
- [ ] Backtesting engine
- [ ] Trading strategy simulation
- [ ] WebSocket real-time updates
- [ ] User authentication
- [ ] Saved predictions history
- [ ] Email/SMS alerts
- [ ] Mobile PWA

### Phase 4: Production Readiness (📋 Planned)
- [ ] Unit & integration tests
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Performance optimization
- [ ] Load balancing
- [ ] Monitoring & alerting
- [ ] Database integration
- [ ] Caching layer (Redis)
- [ ] Rate limiting
- [ ] API authentication

See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for the complete roadmap.

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guide for Python code
- Add tests for new features
- Update documentation
- Ensure all tests pass
- Write clear commit messages

## ⚠️ Disclaimer

**Important:** This application is for **educational and research purposes only**. 

⚠️ **NOT FINANCIAL ADVICE**: Stock price predictions should NOT be used as the sole basis for investment decisions. 

Always:
- Consult with qualified financial advisors
- Conduct your own research
- Understand the risks involved
- Never invest more than you can afford to lose
- Be aware that past performance does not guarantee future results

**USE AT YOUR OWN RISK** - The authors and contributors are not liable for any financial losses.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Rajiv Rathod

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.
```

## 👤 Author

**Rajiv Rathod**
- GitHub: [@rajiv-rathod](https://github.com/rajiv-rathod)
- Repository: [Stock-Price-Prediction-using-LSTM](https://github.com/rajiv-rathod/Stock-Price-Prediction-using-LSTM)

## 🙏 Acknowledgments

- **TensorFlow/Keras Team** - Deep learning framework
- **Yahoo Finance** - Free stock market data API
- **Bootstrap Team** - Responsive UI framework
- **Chart.js Team** - Interactive charting library
- **Flask Team** - Python web framework
- **Open Source Community** - Inspiration and support

## Project Statistics

- **Version**: 1.0.0
- **Lines of Code**: 2,500+
- **Languages**: Python, JavaScript, HTML, CSS
- **Files**: 20+
- **Documentation**: 7 guides
- **API Endpoints**: 5
- **Supported Stocks**: 40+ pre-indexed, unlimited via search
- **Test Coverage**: API and model validation tests

## 🌟 Star History

If you find this project helpful, please consider giving it a star! ⭐

Your support helps the project grow and motivates continued development.

## 📞 Support & Community

- **Issues**: [GitHub Issues](https://github.com/rajiv-rathod/Stock-Price-Prediction-using-LSTM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rajiv-rathod/Stock-Price-Prediction-using-LSTM/discussions)
- **Pull Requests**: Always welcome!

## 📈 Example Results

After training on AAPL stock:
```
==================================================
PERFORMANCE METRICS
==================================================
Mean Squared Error (MSE):      52.4244
Root Mean Squared Error (RMSE): 7.2405
Mean Absolute Error (MAE):      5.4977
Mean Absolute Percentage Error (MAPE): 2.60%
==================================================
```

**Prediction Dashboard Features:**
- 📊 Historical price chart with actual data
- 🎯 Prediction overlay with confidence intervals
- 📈 Training history (loss curves)
- 🔮 Multi-day future forecast
- 📉 Performance metrics table
- 💹 Price change indicators

---

**Made with ❤️ for the Open Source Community**

**[⬆ Back to Top](#-stock-price-prediction-using-lstm)**
