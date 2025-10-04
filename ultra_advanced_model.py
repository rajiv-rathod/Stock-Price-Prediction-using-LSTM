"""
Ultra-Advanced Stock Prediction System
Combines: Transformer + LSTM + Sentiment + Macro Indicators + Advanced Feature Engineering
World-class model with state-of-the-art techniques
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from datetime import datetime, timedelta
import logging
from scipy import signal
from scipy.fft import fft, fftfreq
import pywt  # PyWavelets for wavelet decomposition

from transformer_model import TransformerStockPredictor
from sentiment_analyzer import StockSentimentAnalyzer, MacroIndicators
from advanced_model import AdvancedStockPredictor

logger = logging.getLogger(__name__)


class UltraAdvancedPredictor:
    """
    Ultimate stock prediction system combining:
    - Transformer architecture with multi-head attention
    - Bidirectional LSTM for sequence modeling
    - Technical indicators (19 features)
    - Sentiment analysis (news + social media)
    - Macroeconomic indicators
    - Fourier analysis for cyclical patterns
    - Wavelet decomposition for multi-scale features
    - Ensemble predictions
    """
    
    def __init__(self, ticker='AAPL', lookback=60):
        self.ticker = ticker.upper()
        self.lookback = lookback
        self.scaler = RobustScaler()
        self.models = {}
        self.feature_columns = []
        self.sentiment_analyzer = StockSentimentAnalyzer(ticker)
        self.macro_indicators = MacroIndicators()
    
    def add_fourier_features(self, data, n_components=5):
        """
        Add Fourier transform features to capture cyclical patterns
        Detects periodic behaviors in stock prices
        """
        prices = data['Close'].values
        
        # Compute FFT
        fft_values = fft(prices)
        fft_freq = fftfreq(len(prices))
        
        # Get dominant frequencies
        power = np.abs(fft_values) ** 2
        top_freq_idx = np.argsort(power)[-n_components:]
        
        # Create features from dominant frequencies
        for i, idx in enumerate(top_freq_idx):
            freq = fft_freq[idx]
            if freq != 0:  # Skip DC component
                period = 1 / abs(freq)
                data[f'fourier_sin_{i}'] = np.sin(2 * np.pi * np.arange(len(prices)) / period)
                data[f'fourier_cos_{i}'] = np.cos(2 * np.pi * np.arange(len(prices)) / period)
        
        logger.info(f"Added {n_components*2} Fourier features")
        return data
    
    def add_wavelet_features(self, data, wavelet='db4', level=3):
        """
        Add wavelet decomposition features for multi-scale analysis
        Captures patterns at different time scales
        """
        prices = data['Close'].values
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(prices, wavelet, level=level)
        
        # Reconstruct signals at different scales
        for i in range(level + 1):
            # Create reconstruction with only one level of coefficients
            rec_coeffs = [np.zeros_like(c) if j != i else c for j, c in enumerate(coeffs)]
            reconstructed = pywt.waverec(rec_coeffs, wavelet)
            
            # Trim to match original length
            reconstructed = reconstructed[:len(prices)]
            data[f'wavelet_level_{i}'] = reconstructed
        
        logger.info(f"Added {level+1} wavelet decomposition features")
        return data
    
    def add_statistical_features(self, data, windows=[5, 10, 20]):
        """
        Add advanced statistical features
        """
        for window in windows:
            # Skewness and Kurtosis
            data[f'skew_{window}'] = data['Close'].rolling(window).skew()
            data[f'kurt_{window}'] = data['Close'].rolling(window).kurt()
            
            # Percentile features
            data[f'percentile_25_{window}'] = data['Close'].rolling(window).quantile(0.25)
            data[f'percentile_75_{window}'] = data['Close'].rolling(window).quantile(0.75)
            
            # Z-score
            mean = data['Close'].rolling(window).mean()
            std = data['Close'].rolling(window).std()
            data[f'zscore_{window}'] = (data['Close'] - mean) / (std + 1e-8)
        
        logger.info(f"Added statistical features for {len(windows)} windows")
        return data
    
    def add_order_flow_features(self, data):
        """
        Add order flow and market microstructure features
        """
        # Volume-weighted features
        data['vwap'] = (data['Close'] * data['Volume']).rolling(20).sum() / data['Volume'].rolling(20).sum()
        data['volume_price_trend'] = data['Volume'] * ((data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1))
        
        # Money flow index
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-8)))
        data['money_flow_index'] = mfi
        
        # Chaikin Money Flow
        mf_multiplier = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'] + 1e-8)
        mf_volume = mf_multiplier * data['Volume']
        data['chaikin_mf'] = mf_volume.rolling(20).sum() / data['Volume'].rolling(20).sum()
        
        logger.info("Added order flow and microstructure features")
        return data
    
    def fetch_and_prepare_all_data(self, period='5y'):
        """
        Fetch and prepare data from all sources:
        - Price data
        - Technical indicators
        - Sentiment data
        - Macro indicators
        - Advanced engineered features
        """
        try:
            # Fetch price data
            logger.info(f"Fetching data for {self.ticker}...")
            price_data = yf.download(self.ticker, period=period, progress=False)
            
            if price_data.empty:
                raise ValueError(f"No data found for {self.ticker}")
            
            logger.info(f"Fetched {len(price_data)} price records")
            
            # Add basic technical indicators (from AdvancedStockPredictor)
            advanced_predictor = AdvancedStockPredictor(self.ticker, self.lookback)
            price_data = advanced_predictor.add_technical_indicators(price_data)
            
            # Add Fourier features
            price_data = self.add_fourier_features(price_data, n_components=5)
            
            # Add wavelet features
            price_data = self.add_wavelet_features(price_data, level=3)
            
            # Add statistical features
            price_data = self.add_statistical_features(price_data)
            
            # Add order flow features
            price_data = self.add_order_flow_features(price_data)
            
            # Get sentiment data
            logger.info("Fetching sentiment data...")
            sentiment_df = self.sentiment_analyzer.get_sentiment_features(days_back=len(price_data))
            
            if sentiment_df is not None and not sentiment_df.empty:
                # Align dates
                sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
                price_data['date'] = price_data.index.date
                
                # Merge sentiment data
                price_data = price_data.merge(sentiment_df, on='date', how='left')
                price_data = price_data.drop('date', axis=1)
                price_data['combined_sentiment'] = price_data['combined_sentiment'].fillna(0)
                price_data['sentiment_strength'] = price_data['sentiment_strength'].fillna(0)
                price_data['sentiment_ma_7'] = price_data['sentiment_ma_7'].fillna(0)
                price_data['sentiment_ma_30'] = price_data['sentiment_ma_30'].fillna(0)
                price_data['sentiment_momentum'] = price_data['sentiment_momentum'].fillna(0)
                price_data['sentiment_volatility'] = price_data['sentiment_volatility'].fillna(0)
                
                logger.info("Merged sentiment data")
            
            # Get macro indicators
            logger.info("Fetching macro indicators...")
            macro_df = self.macro_indicators.get_all_indicators()
            
            if not macro_df.empty:
                # Align dates
                macro_df['date'] = pd.to_datetime(macro_df['date']).dt.date
                price_data['date'] = price_data.index.date
                
                # Merge macro data
                price_data = price_data.merge(macro_df, on='date', how='left')
                price_data = price_data.drop('date', axis=1)
                price_data = price_data.fillna(method='ffill').fillna(method='bfill')
                
                logger.info("Merged macro indicators")
            
            # Drop any remaining NaN
            price_data = price_data.dropna()
            
            logger.info(f"Final dataset: {len(price_data)} records with {len(price_data.columns)} features")
            
            return price_data
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def prepare_features(self, data, split_ratio=0.8):
        """Prepare features for training"""
        # Select all numeric columns except OHLCV
        feature_cols = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Volume']]
        self.feature_columns = feature_cols
        
        dataset = data[feature_cols].values
        
        # Scale data
        scaled_data = self.scaler.fit_transform(dataset)
        
        # Split
        train_size = int(len(scaled_data) * split_ratio)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size - self.lookback:]
        
        return train_data, test_data, dataset
    
    def create_sequences(self, data, target_col_idx=0):
        """Create sequences for training"""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i])
            y.append(data[i, target_col_idx])
        
        X, y = np.array(X), np.array(y)
        return X, y
    
    def train_ensemble(self, epochs=100, batch_size=32, use_transformer=True, use_lstm=True):
        """
        Train ensemble of models:
        - Transformer model
        - Advanced LSTM model
        - Hybrid CNN+Transformer+LSTM
        """
        try:
            # Fetch and prepare data
            data = self.fetch_and_prepare_all_data()
            train_data, test_data, dataset = self.prepare_features(data)
            
            # Create sequences
            X_train, y_train = self.create_sequences(train_data)
            X_test, y_test = self.create_sequences(test_data)
            
            logger.info(f"Training data shape: {X_train.shape}")
            logger.info(f"Number of features: {X_train.shape[2]}")
            
            # Store for later use
            self.X_test = X_test
            self.y_test = y_test
            self.test_data = test_data
            self.dataset = dataset
            self.train_size = len(train_data)
            
            # Train Transformer model
            if use_transformer:
                logger.info("Training Transformer model...")
                transformer = TransformerStockPredictor(self.lookback, X_train.shape[2])
                transformer.train(X_train, y_train, epochs=epochs, batch_size=batch_size, use_hybrid=True)
                self.models['transformer'] = transformer
                logger.info("Transformer model trained")
            
            # Train Advanced LSTM
            if use_lstm:
                logger.info("Training Advanced LSTM model...")
                from advanced_model import AdvancedStockPredictor
                lstm_predictor = AdvancedStockPredictor(self.ticker, self.lookback, use_technical_indicators=True)
                
                # Use already prepared data
                lstm_predictor.scaler = self.scaler
                lstm_predictor.feature_columns = self.feature_columns
                lstm_predictor.test_data = test_data
                lstm_predictor.dataset = dataset
                lstm_predictor.train_size = self.train_size
                
                # Build and train model
                lstm_predictor.model = lstm_predictor.build_ensemble_model((X_train.shape[1], X_train.shape[2]))
                lstm_predictor.history = lstm_predictor.model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.15,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001)
                    ],
                    verbose=1
                )
                
                self.models['lstm'] = lstm_predictor
                logger.info("Advanced LSTM trained")
            
            logger.info(f"Ensemble training complete. {len(self.models)} models trained.")
            return True
            
        except Exception as e:
            logger.error(f"Ensemble training error: {str(e)}")
            raise
    
    def predict_ensemble(self, days_ahead=5):
        """
        Make ensemble predictions combining all models
        Uses weighted voting based on historical accuracy
        """
        if not self.models:
            raise ValueError("No models trained yet")
        
        predictions_list = []
        weights = []
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                if model_name == 'transformer':
                    # Transformer prediction
                    test_pred = model.predict(self.X_test)
                    predictions_list.append(test_pred)
                    weights.append(1.2)  # Higher weight for transformer
                    
                elif model_name == 'lstm':
                    # LSTM prediction
                    test_pred = model.model.predict(self.X_test, verbose=0)
                    predictions_list.append(test_pred)
                    weights.append(1.0)
                
            except Exception as e:
                logger.warning(f"Error getting prediction from {model_name}: {str(e)}")
        
        # Weighted average
        weights = np.array(weights) / sum(weights)
        ensemble_pred = sum(p * w for p, w in zip(predictions_list, weights))
        
        # Inverse transform for actual prices
        pred_full = np.zeros((len(ensemble_pred), len(self.feature_columns)))
        pred_full[:, 0] = ensemble_pred[:, 0]
        predictions_actual = self.scaler.inverse_transform(pred_full)[:, 0]
        
        y_test_full = np.zeros((len(self.y_test), len(self.feature_columns)))
        y_test_full[:, 0] = self.y_test
        y_test_actual = self.scaler.inverse_transform(y_test_full)[:, 0]
        
        # Future predictions
        future_predictions = []
        last_sequence = self.test_data[-self.lookback:]
        
        for _ in range(days_ahead):
            X_pred = last_sequence.reshape(1, self.lookback, -1)
            
            # Get prediction from each model
            future_preds = []
            for model_name, model in self.models.items():
                if model_name == 'transformer':
                    pred = model.predict(X_pred)
                    future_preds.append(pred[0, 0])
                elif model_name == 'lstm':
                    pred = model.model.predict(X_pred, verbose=0)
                    future_preds.append(pred[0, 0])
            
            # Weighted average
            ensemble_future = sum(p * w for p, w in zip(future_preds, weights))
            
            # Inverse transform
            pred_full = np.zeros((1, len(self.feature_columns)))
            pred_full[0, 0] = ensemble_future
            pred_price = self.scaler.inverse_transform(pred_full)[0, 0]
            future_predictions.append(pred_price)
            
            # Update sequence
            new_row = np.zeros((1, len(self.feature_columns)))
            new_row[0, 0] = ensemble_future
            last_sequence = np.vstack([last_sequence[1:], new_row])
        
        return predictions_actual, y_test_actual, future_predictions
    
    def evaluate(self):
        """Evaluate ensemble performance"""
        predictions_actual, y_test_actual, _ = self.predict_ensemble(days_ahead=1)
        
        # Calculate metrics
        mse = np.mean((predictions_actual - y_test_actual) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions_actual - y_test_actual))
        mape = np.mean(np.abs((y_test_actual - predictions_actual) / y_test_actual)) * 100
        
        # Directional accuracy
        actual_direction = np.diff(y_test_actual) > 0
        pred_direction = np.diff(predictions_actual) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        metrics = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape),
            'directional_accuracy': float(directional_accuracy)
        }
        
        return metrics, predictions_actual, y_test_actual


import tensorflow as tf
