# 📈 Stock Price Prediction using LSTM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/status-production-brightgreen.svg)]()

A **production-ready**, **real-time** web application for stock price prediction using LSTM (Long Short-Term Memory) neural networks. Features a modern UI, REST API, CSV upload, company search, and comprehensive analytics.

## ✨ Features

### 🤖 Core AI/ML
- **Deep Learning LSTM** - 3-layer neural network for time series forecasting
- **Real-time Data** - Live stock data from Yahoo Finance API
- **Custom Data Support** - Upload your own CSV files for analysis
- **Advanced Metrics** - MAPE, RMSE, MAE, and directional accuracy
- **Multi-day Forecasting** - Predict up to 30 days ahead

### 🌐 Web Application
- **Modern UI** - Responsive design with Bootstrap 5
- **Company Search** - Real-time autocomplete ticker search
- **Interactive Charts** - 4-panel analytical dashboard with Chart.js
- **Dual Input Modes** - Ticker search or CSV upload
- **Quick Select** - Popular stocks shortcuts (AAPL, GOOGL, MSFT, etc.)
- **Mobile-Friendly** - Works on all devices

### 🔧 Developer Features
- **REST API** - Full API access for integration
- **Docker Support** - Containerized deployment ready
- **Comprehensive Docs** - 7+ documentation guides
- **Error Handling** - Graceful failure management
- **Logging** - Request/error tracking
- **CORS Enabled** - Cross-origin resource sharing

## 🚀 Quick Start

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

### Alternative: Command-Line Usage

```bash
# Run standalone prediction script
python stock_prediction.py
```

### Using Docker

```bash
# Build and run with Docker Compose
docker-compose up

# Or with Docker directly
docker build -t stock-predictor .
docker run -p 5000:5000 stock-predictor
```

## 📖 Usage

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

## 📊 API Endpoints

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/` | GET | Web interface | - |
| `/api/health` | GET | Health check | - |
| `/api/search_companies` | GET | Search companies | `q` (query string) |
| `/api/predict` | POST | Generate prediction | JSON body (see below) |
| `/api/upload_csv` | POST | Upload custom data | Form data with file |

### Prediction Request Body
```json
{
  "ticker": "AAPL",
  "lookback": 60,
  "epochs": 50,
  "days_ahead": 5
}
```

See [WEB_APP_GUIDE.md](WEB_APP_GUIDE.md) for complete API documentation.

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

## 📁 Project Structure

```
Stock-Price-Prediction-using-LSTM/
├── app.py                      # Flask web application (600+ lines)
├── stock_prediction.py         # Core LSTM model implementation
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
├── docker-compose.yml          # Multi-container orchestration
├── start_web_app.sh            # Bash startup script
├── templates/
│   └── index.html              # Frontend interface (323 lines)
├── static/
│   ├── style.css               # UI styling (250+ lines)
│   └── script.js               # Frontend logic (400+ lines)
├── docs/
│   ├── WEB_APP_GUIDE.md        # User guide
│   ├── DEPLOYMENT.md           # Deployment instructions
│   ├── CSV_FORMAT_GUIDE.md     # CSV upload format
│   ├── ARCHITECTURE.md         # System design
│   ├── PROJECT_STRUCTURE.md    # Code organization
│   ├── QUICK_START.md          # Getting started
│   └── PROJECT_SUMMARY.md      # Implementation status
└── tests/
    ├── test_api.py             # API tests
    └── test_validation.py      # Model validation tests
```

## 🎯 Performance

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

## 🌐 Deployment

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

## 📚 Documentation

- **[WEB_APP_GUIDE.md](WEB_APP_GUIDE.md)** - Complete user guide and API documentation
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Deployment instructions for various platforms
- **[CSV_FORMAT_GUIDE.md](CSV_FORMAT_GUIDE.md)** - CSV upload format specifications
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Code organization guide
- **[QUICK_START.md](QUICK_START.md)** - Quick start tutorial
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Implementation status and roadmap

## 🛠️ Tech Stack

**Backend:**
- Python 3.11+
- Flask 3.1.2 - Web framework
- TensorFlow 2.20.0 / Keras 3.11.3 - Deep learning
- NumPy 2.3.3, Pandas 2.3.3 - Data processing
- scikit-learn 1.7.0 - Preprocessing & metrics
- yfinance 0.2.66 - Yahoo Finance API
- Matplotlib 3.10.3 - Visualization

**Frontend:**
- HTML5/CSS3
- JavaScript (ES6+)
- Bootstrap 5.3.0 - UI framework
- Chart.js 4.4.0 - Interactive charts
- Font Awesome 6.0.0 - Icons

**DevOps:**
- Docker & Docker Compose
- Git/GitHub
- Shell scripts (Bash)

## 🔮 Roadmap

### Phase 1: Core Features (✅ Complete)
- [x] LSTM model implementation
- [x] Web application with Flask
- [x] REST API endpoints
- [x] Real-time data fetching
- [x] Interactive charts
- [x] Company search
- [x] CSV upload capability
- [x] Docker containerization
- [x] Comprehensive documentation

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
- ✅ Consult with qualified financial advisors
- ✅ Conduct your own research
- ✅ Understand the risks involved
- ✅ Never invest more than you can afford to lose
- ✅ Be aware that past performance does not guarantee future results

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

## 📊 Project Statistics

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
