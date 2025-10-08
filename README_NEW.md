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