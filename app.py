"""
Flask Web Application for Stock Price Prediction
Provides REST API endpoints and web interface for LSTM-based stock prediction
Includes CSV upload, company search, and real-time predictions
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import os
import json
import logging
from functools import wraps
from werkzeug.utils import secure_filename
from advanced_model import AdvancedStockPredictor
from ultra_advanced_model import UltraAdvancedPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
MODEL_CACHE_DIR = 'models_cache'
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


class StockPredictorAPI:
    """Enhanced Stock Price Predictor for API usage"""
    
    def __init__(self, ticker='AAPL', lookback=60, data_source='yfinance', custom_data=None):
        self.ticker = ticker.upper()
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None
        self.data = None
        self.data_source = data_source
        self.custom_data = custom_data
        
    def fetch_data(self, start_date=None, end_date=None, period='5y'):
        """Fetch stock data from Yahoo Finance or custom CSV"""
        try:
            if self.data_source == 'csv' and self.custom_data is not None:
                # Use custom CSV data
                self.data = self.custom_data
                logger.info(f"Loaded {len(self.data)} records from CSV")
            else:
                # Fetch from Yahoo Finance
                if end_date is None:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                
                if start_date is None:
                    self.data = yf.download(self.ticker, period=period, progress=False)
                else:
                    self.data = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
                
                logger.info(f"Fetched {len(self.data)} records for {self.ticker}")
            
            if self.data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def prepare_data(self, data, split_ratio=0.8):
        """Prepare data for LSTM training"""
        dataset = data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(dataset)
        
        train_size = int(len(scaled_data) * split_ratio)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size - self.lookback:]
        
        return train_data, test_data, dataset
    
    def create_sequences(self, data):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i, 0])
            y.append(data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train(self, epochs=50, batch_size=32):
        """Train the LSTM model"""
        try:
            data = self.fetch_data()
            train_data, test_data, dataset = self.prepare_data(data)
            
            X_train, y_train = self.create_sequences(train_data)
            
            self.model = self.build_model((X_train.shape[1], 1))
            
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=0
            )
            
            self.test_data = test_data
            self.dataset = dataset
            self.train_size = len(train_data)
            
            return True
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise
    
    def predict(self, days_ahead=1):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Predict on test data
        X_test, y_test = self.create_sequences(self.test_data)
        predictions = self.model.predict(X_test, verbose=0)
        predictions = self.scaler.inverse_transform(predictions)
        
        # Future prediction
        last_sequence = self.test_data[-self.lookback:]
        future_predictions = []
        
        for _ in range(days_ahead):
            last_sequence_reshaped = last_sequence.reshape(1, self.lookback, 1)
            next_pred = self.model.predict(last_sequence_reshaped, verbose=0)
            future_predictions.append(next_pred[0, 0])
            last_sequence = np.append(last_sequence[1:], next_pred)
        
        future_predictions = self.scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        )
        
        return predictions, future_predictions
    
    def calculate_metrics(self, predictions):
        """Calculate performance metrics"""
        actual = self.dataset[self.train_size:]
        
        mse = float(np.mean((actual - predictions) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(actual - predictions)))
        mape = float(np.mean(np.abs((actual - predictions) / actual)) * 100)
        
        # Directional accuracy
        actual_direction = np.diff(actual.flatten()) > 0
        pred_direction = np.diff(predictions.flatten()) > 0
        directional_accuracy = float(np.mean(actual_direction == pred_direction) * 100)
        
        return {
            'mse': round(mse, 4),
            'rmse': round(rmse, 4),
            'mae': round(mae, 4),
            'mape': round(mape, 2),
            'directional_accuracy': round(directional_accuracy, 2)
        }
    
    def get_stock_info(self):
        """Get stock information"""
        try:
            stock = yf.Ticker(self.ticker)
            info = stock.info
            return {
                'name': info.get('longName', self.ticker),
                'symbol': self.ticker,
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'currency': info.get('currency', 'USD')
            }
        except Exception as e:
            logger.warning(f"Could not fetch stock info: {str(e)}")
            return {'name': self.ticker, 'symbol': self.ticker}


def create_plot(predictor, predictions, future_predictions):
    """Create visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{predictor.ticker} Stock Price Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Full historical data
    ax1 = axes[0, 0]
    dates = predictor.data.index
    ax1.plot(dates, predictor.dataset, label='Actual Price', color='#2E86AB', linewidth=2)
    train_split_date = dates[predictor.train_size]
    ax1.axvline(x=train_split_date, color='red', linestyle='--', label='Train/Test Split', alpha=0.7)
    ax1.set_title('Historical Stock Price', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Predictions vs Actual
    ax2 = axes[0, 1]
    test_dates = dates[predictor.train_size:]
    actual = predictor.dataset[predictor.train_size:]
    ax2.plot(test_dates, actual, label='Actual Price', color='#2E86AB', linewidth=2)
    ax2.plot(test_dates, predictions, label='Predicted Price', color='#F24236', linewidth=2, alpha=0.8)
    ax2.set_title('Predictions vs Actual (Test Set)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price (USD)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training history
    ax3 = axes[1, 0]
    if predictor.history:
        epochs = range(1, len(predictor.history.history['loss']) + 1)
        ax3.plot(epochs, predictor.history.history['loss'], label='Training Loss', color='#2E86AB')
        ax3.plot(epochs, predictor.history.history['val_loss'], label='Validation Loss', color='#F24236')
        ax3.set_title('Model Training History', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Future predictions
    ax4 = axes[1, 1]
    last_date = dates[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(future_predictions))
    
    # Plot last 30 days of actual data for context
    context_days = min(30, len(dates))
    ax4.plot(dates[-context_days:], predictor.dataset[-context_days:], 
             label='Recent Actual', color='#2E86AB', linewidth=2)
    ax4.plot(future_dates, future_predictions, label='Future Prediction', 
             color='#F24236', linewidth=2, marker='o', markersize=6)
    ax4.axvline(x=last_date, color='green', linestyle='--', label='Today', alpha=0.7)
    ax4.set_title(f'Future Predictions ({len(future_predictions)} days)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Price (USD)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return plot_data


# API Routes
@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Render professional trading dashboard"""
    return render_template('dashboard.html')


@app.route('/api/stock-info', methods=['GET'])
def stock_info():
    """Get current stock information"""
    try:
        ticker = request.args.get('ticker', 'AAPL').upper()
        
        # Fetch current data
        stock = yf.Ticker(ticker)
        hist = stock.history(period='2d')
        
        if hist.empty:
            return jsonify({'success': False, 'error': 'No data found'}), 404
        
        current_price = float(hist['Close'].iloc[-1])
        prev_price = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
        change = current_price - prev_price
        change_percent = (change / prev_price) * 100 if prev_price != 0 else 0
        
        info = stock.info
        
        response = {
            'success': True,
            'ticker': ticker,
            'name': info.get('longName', ticker),
            'current_price': round(current_price, 2),
            'change': round(change, 2),
            'change_percent': round(change_percent, 2),
            'volume': int(hist['Volume'].iloc[-1]),
            'market_cap': info.get('marketCap', 0),
            'sector': info.get('sector', 'N/A'),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Stock info error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for stock prediction
    Request body: {ticker, lookback, epochs, days_ahead} or file upload
    """
    try:
        # Check if it's a file upload or JSON request
        custom_data = None
        data_source = 'yfinance'
        
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Read CSV file
                custom_data = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
                data_source = 'csv'
                
                # Get other parameters from form data
                ticker = request.form.get('ticker', 'CUSTOM').upper()
                lookback = int(request.form.get('lookback', 60))
                epochs = int(request.form.get('epochs', 50))
                days_ahead = int(request.form.get('days_ahead', 5))
            else:
                return jsonify({'success': False, 'error': 'Invalid file type. Please upload a CSV file'}), 400
        else:
            # JSON request
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': 'No data provided'}), 400
            
            ticker = data.get('ticker', 'AAPL').upper()
            lookback = int(data.get('lookback', 60))
            epochs = int(data.get('epochs', 50))
            days_ahead = int(data.get('days_ahead', 5))
        
        # Validate inputs
        if lookback < 10 or lookback > 200:
            return jsonify({'success': False, 'error': 'Lookback period must be between 10 and 200 days'}), 400
        if epochs < 10 or epochs > 100:
            return jsonify({'success': False, 'error': 'Epochs must be between 10 and 100'}), 400
        if days_ahead < 1 or days_ahead > 30:
            return jsonify({'success': False, 'error': 'Days ahead must be between 1 and 30'}), 400
        
        logger.info(f"Processing prediction request for {ticker} (source: {data_source})")
        
        # Create predictor and train
        predictor = StockPredictorAPI(ticker=ticker, lookback=lookback, data_source=data_source, custom_data=custom_data)
        predictor.train(epochs=epochs)
        
        # Make predictions
        predictions, future_predictions = predictor.predict(days_ahead=days_ahead)
        
        # Calculate metrics
        metrics = predictor.calculate_metrics(predictions)
        
        # Get stock info
        if data_source == 'csv':
            stock_info = {'name': ticker, 'symbol': ticker, 'sector': 'Custom Data', 'industry': 'N/A'}
        else:
            stock_info = predictor.get_stock_info()
        
        # Create visualization
        plot_data = create_plot(predictor, predictions, future_predictions)
        
        # Prepare response data
        actual = predictor.dataset[predictor.train_size:].flatten()
        test_dates = predictor.data.index[predictor.train_size:].strftime('%Y-%m-%d').tolist()
        
        future_dates = pd.date_range(
            start=predictor.data.index[-1] + timedelta(days=1),
            periods=days_ahead
        ).strftime('%Y-%m-%d').tolist()
        
        current_price = float(predictor.data['Close'].iloc[-1])
        predicted_price = float(future_predictions[0][0])
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        response = {
            'success': True,
            'ticker': ticker,
            'model_type': 'Standard LSTM',
            'features_used': 1,
            'stock_info': stock_info,
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted_price, 2),
            'price_change': round(price_change, 2),
            'price_change_pct': round(price_change_pct, 2),
            'metrics': metrics,
            'dates': test_dates,
            'actual_prices': [float(x) for x in actual.tolist()],
            'predictions': [float(x) for x in predictions.flatten().tolist()],
            'future_predictions': [float(x) for x in future_predictions.flatten().tolist()],
            'future_dates': future_dates,
            'historical_data': {
                'dates': test_dates,
                'actual': [float(x) for x in actual.tolist()],
                'predicted': [float(x) for x in predictions.flatten().tolist()]
            },
            'plot': plot_data,
            'training_time': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': f'An error occurred: {str(e)}'}), 500


@app.route('/api/search-companies', methods=['GET'])
def search_companies():
    """Search for companies by name or ticker"""
    try:
        query = request.args.get('q', '').upper()
        if not query or len(query) < 1:
            return jsonify({'success': False, 'error': 'Query parameter required'}), 400
        
        # Popular companies database (subset for demo)
        companies = {
            'AAPL': 'Apple Inc.',
            'GOOGL': 'Alphabet Inc.',
            'MSFT': 'Microsoft Corporation',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation',
            'JPM': 'JPMorgan Chase & Co.',
            'V': 'Visa Inc.',
            'WMT': 'Walmart Inc.',
            'JNJ': 'Johnson & Johnson',
            'PG': 'Procter & Gamble Co.',
            'MA': 'Mastercard Inc.',
            'UNH': 'UnitedHealth Group Inc.',
            'HD': 'The Home Depot Inc.',
            'DIS': 'The Walt Disney Company',
            'BAC': 'Bank of America Corporation',
            'ADBE': 'Adobe Inc.',
            'CRM': 'Salesforce Inc.',
            'NFLX': 'Netflix Inc.',
            'PYPL': 'PayPal Holdings Inc.',
            'INTC': 'Intel Corporation',
            'CSCO': 'Cisco Systems Inc.',
            'PEP': 'PepsiCo Inc.',
            'KO': 'The Coca-Cola Company',
            'NKE': 'NIKE Inc.',
            'MRK': 'Merck & Co. Inc.',
            'ABT': 'Abbott Laboratories',
            'COST': 'Costco Wholesale Corporation',
            'AVGO': 'Broadcom Inc.',
            'TXN': 'Texas Instruments Inc.',
            'QCOM': 'QUALCOMM Inc.',
            'AMD': 'Advanced Micro Devices Inc.',
            'ORCL': 'Oracle Corporation',
            'IBM': 'International Business Machines',
            'SBUX': 'Starbucks Corporation',
            'F': 'Ford Motor Company',
            'GM': 'General Motors Company',
            'BA': 'The Boeing Company',
            'CAT': 'Caterpillar Inc.'
        }
        
        # Search in tickers and names
        results = []
        for ticker, name in companies.items():
            if query in ticker or query in name.upper():
                results.append({
                    'ticker': ticker,
                    'name': name,
                    'display': f"{ticker} - {name}"
                })
        
        # Limit results
        results = results[:10]
        
        return jsonify({'success': True, 'results': results}), 200
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stock-info/<ticker>', methods=['GET'])
def get_stock_info(ticker):
    """Get basic stock information"""
    try:
        predictor = StockPredictorAPI(ticker=ticker)
        info = predictor.get_stock_info()
        
        # Get latest price
        data = predictor.fetch_data(period='5d')
        latest_price = float(data['Close'].iloc[-1])
        prev_price = float(data['Close'].iloc[-2])
        
        info['latest_price'] = round(latest_price, 2)
        info['prev_price'] = round(prev_price, 2)
        info['change'] = round(latest_price - prev_price, 2)
        info['change_pct'] = round(((latest_price - prev_price) / prev_price) * 100, 2)
        
        return jsonify({'success': True, 'data': info}), 200
    except Exception as e:
        logger.error(f"Error fetching stock info: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/predict-advanced', methods=['POST'])
def predict_advanced():
    """
    Advanced prediction endpoint using state-of-the-art LSTM with technical indicators
    Request body: {ticker, lookback, epochs, days_ahead, use_ensemble, use_technical_indicators}
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        ticker = data.get('ticker', 'AAPL').upper()
        lookback = int(data.get('lookback', 60))
        epochs = int(data.get('epochs', 100))
        days_ahead = int(data.get('days_ahead', 5))
        use_ensemble = data.get('use_ensemble', False)
        use_technical_indicators = data.get('use_technical_indicators', True)
        
        # Validate inputs
        if lookback < 10 or lookback > 200:
            return jsonify({'success': False, 'error': 'Lookback period must be between 10 and 200 days'}), 400
        if epochs < 20 or epochs > 200:
            return jsonify({'success': False, 'error': 'Epochs must be between 20 and 200'}), 400
        if days_ahead < 1 or days_ahead > 30:
            return jsonify({'success': False, 'error': 'Days ahead must be between 1 and 30'}), 400
        
        logger.info(f"Processing ADVANCED prediction for {ticker} with technical indicators={use_technical_indicators}, ensemble={use_ensemble}")
        
        # Create advanced predictor
        predictor = AdvancedStockPredictor(
            ticker=ticker,
            lookback=lookback,
            use_technical_indicators=use_technical_indicators
        )
        
        # Train model
        predictor.train(epochs=epochs, batch_size=32, use_ensemble=use_ensemble, patience=15)
        
        # Evaluate and get predictions
        metrics, predictions, actuals = predictor.evaluate()
        
        # Future predictions
        future_predictions = predictor.predict_future(days_ahead=days_ahead)
        
        # Get current price
        current_data = yf.download(ticker, period='1d', progress=False)
        current_price = float(current_data['Close'].iloc[-1])
        
        # Calculate changes
        predicted_price = future_predictions[0]
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # Prepare dates
        last_date = predictor.fetch_and_prepare_data().index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days_ahead
        ).strftime('%Y-%m-%d').tolist()
        
        # Training history
        training_history = {
            'loss': [float(x) for x in predictor.history.history['loss']],
            'val_loss': [float(x) for x in predictor.history.history['val_loss']],
            'mae': [float(x) for x in predictor.history.history['mae']],
            'val_mae': [float(x) for x in predictor.history.history['val_mae']]
        }
        
        response = {
            'success': True,
            'ticker': ticker,
            'model_type': 'Ensemble (LSTM+GRU)' if use_ensemble else 'Advanced Bidirectional LSTM',
            'features_used': len(predictor.feature_columns),
            'feature_list': predictor.feature_columns,
            'stock_info': {'name': ticker, 'symbol': ticker},
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted_price, 2),
            'price_change': round(price_change, 2),
            'price_change_pct': round(price_change_pct, 2),
            'metrics': {
                'mape': round(metrics['MAPE'], 2),
                'rmse': round(metrics['RMSE'], 2),
                'mae': round(metrics['MAE'], 2),
                'mse': round(metrics['MSE'], 2),
                'directional_accuracy': round(metrics['directional_accuracy'], 2)
            },
            'dates': [d.strftime('%Y-%m-%d') for d in predictor.dataset.index[-len(actuals):]],
            'actual_prices': [float(x) for x in actuals],
            'predictions': [float(x) for x in predictions],
            'future_predictions': [round(float(x), 2) for x in future_predictions],
            'future_dates': future_dates,
            'training_history': training_history,
            'training_time': 0,
            'training_epochs_completed': len(predictor.history.history['loss']),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Advanced prediction error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict-ultra', methods=['POST'])
def predict_ultra():
    """
    ULTRA-ADVANCED prediction endpoint
    Features:
    - Transformer + LSTM + GRU ensemble
    - Sentiment analysis (news + social media)
    - Macroeconomic indicators
    - Fourier analysis for cycles
    - Wavelet decomposition
    - Advanced statistical features
    - Order flow analysis
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        ticker = data.get('ticker', 'AAPL').upper()
        lookback = int(data.get('lookback', 60))
        epochs = int(data.get('epochs', 100))
        days_ahead = int(data.get('days_ahead', 5))
        use_transformer = data.get('use_transformer', True)
        use_lstm = data.get('use_lstm', True)
        
        # Validate inputs
        if lookback < 10 or lookback > 200:
            return jsonify({'success': False, 'error': 'Lookback period must be between 10 and 200 days'}), 400
        if epochs < 20 or epochs > 200:
            return jsonify({'success': False, 'error': 'Epochs must be between 20 and 200'}), 400
        if days_ahead < 1 or days_ahead > 30:
            return jsonify({'success': False, 'error': 'Days ahead must be between 1 and 30'}), 400
        
        logger.info(f"Processing ULTRA-ADVANCED prediction for {ticker}")
        logger.info(f"Features: Transformer={use_transformer}, LSTM={use_lstm}, Sentiment=Yes, Macro=Yes, Fourier=Yes, Wavelet=Yes")
        
        # Create ultra-advanced predictor
        predictor = UltraAdvancedPredictor(ticker=ticker, lookback=lookback)
        
        # Train ensemble
        predictor.train_ensemble(
            epochs=epochs,
            batch_size=32,
            use_transformer=use_transformer,
            use_lstm=use_lstm
        )
        
        # Evaluate and get predictions
        metrics, predictions, actuals = predictor.evaluate()
        
        # Future predictions
        _, _, future_predictions = predictor.predict_ensemble(days_ahead=days_ahead)
        
        # Get current price
        current_data = yf.download(ticker, period='1d', progress=False)
        current_price = float(current_data['Close'].iloc[-1])
        
        # Calculate changes
        predicted_price = future_predictions[0]
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # Prepare dates
        last_date = current_data.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days_ahead
        ).strftime('%Y-%m-%d').tolist()
        
        # Get training history from first model
        first_model = list(predictor.models.values())[0]
        training_history = {
            'loss': [float(x) for x in first_model.history.history['loss']],
            'val_loss': [float(x) for x in first_model.history.history['val_loss']],
            'mae': [float(x) for x in first_model.history.history['mae']],
            'val_mae': [float(x) for x in first_model.history.history['val_mae']]
        }
        
        # Historical data for charts
        historical_dates = pd.date_range(
            end=last_date,
            periods=len(predictions)
        ).strftime('%Y-%m-%d').tolist()
        
        response = {
            'success': True,
            'ticker': ticker,
            'model_type': f'Ultra-Advanced Ensemble ({len(predictor.models)} models)',
            'models_used': list(predictor.models.keys()),
            'features_used': len(predictor.feature_columns),
            'feature_categories': {
                'technical_indicators': 19,
                'sentiment_features': 6,
                'macro_indicators': 4,
                'fourier_components': 10,
                'wavelet_levels': 4,
                'statistical_features': 9,
                'order_flow': 4,
                'total': len(predictor.feature_columns)
            },
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted_price, 2),
            'price_change': round(price_change, 2),
            'price_change_pct': round(price_change_pct, 2),
            'metrics': {
                'MAPE': round(metrics['MAPE'], 2),
                'RMSE': round(metrics['RMSE'], 2),
                'MAE': round(metrics['MAE'], 2),
                'MSE': round(metrics['MSE'], 2),
                'directional_accuracy': round(metrics['directional_accuracy'], 2)
            },
            'historical_data': {
                'dates': historical_dates,
                'actual': [round(float(x), 2) for x in actuals.tolist()],
                'predicted': [round(float(x), 2) for x in predictions.tolist()]
            },
            'future_predictions': {
                'dates': future_dates,
                'prices': [round(float(x), 2) for x in future_predictions]
            },
            'training_history': training_history,
            'training_epochs_completed': len(training_history['loss']),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Ultra-advanced prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Stock Price Prediction API'
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal error: {str(error)}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("="*60)
    print("Stock Price Prediction Web Application - ADVANCED MODEL")
    print("="*60)
    print("🚀 Features:")
    print("   ✓ Bidirectional LSTM with Attention")
    print("   ✓ Technical Indicators (RSI, MACD, Bollinger Bands, etc.)")
    print("   ✓ Ensemble Models (LSTM + GRU)")
    print("   ✓ Advanced Training (Early Stopping, LR Scheduling)")
    print("   ✓ Multi-feature Analysis")
    print("="*60)
    print("Starting server...")
    print("Access the application at: http://localhost:5000")
    print("API Endpoints:")
    print("   - /api/predict (Standard LSTM)")
    print("   - /api/predict-advanced (Advanced LSTM with indicators)")
    print("   - /api/health")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
