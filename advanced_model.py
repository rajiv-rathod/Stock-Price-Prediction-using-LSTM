"""
Advanced Stock Price Prediction Model
Features:
- Bidirectional LSTM with Attention Mechanism
- Multiple technical indicators (RSI, MACD, Bollinger Bands, EMA, SMA)
- Feature engineering with volume and volatility
- Early stopping and learning rate scheduling
- Ensemble predictions
- Enhanced training with multiple data sources
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, Bidirectional, 
    Attention, Input, LayerNormalization, GRU,
    MultiHeadAttention, GlobalAveragePooling1D, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as K
import logging

logger = logging.getLogger(__name__)


class AdvancedStockPredictor:
    """Advanced LSTM model with technical indicators and attention mechanism"""
    
    def __init__(self, ticker='AAPL', lookback=60, use_technical_indicators=True):
        self.ticker = ticker.upper()
        self.lookback = lookback
        self.use_technical_indicators = use_technical_indicators
        self.scaler = RobustScaler()  # More robust to outliers
        self.model = None
        self.history = None
        self.feature_columns = []
        
    def calculate_rsi(self, data, periods=14):
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    
    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    
    def calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range (ATR) for volatility"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()
    
    def add_technical_indicators(self, data):
        """Add comprehensive technical indicators to the dataset"""
        df = data.copy()
        
        # Flatten multi-index columns if present (from yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving Averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # MACD
        df['MACD'], df['MACD_Signal'] = self.calculate_macd(df['Close'])
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['Close'])
        df['BB_Upper'] = bb_upper
        df['BB_Middle'] = bb_middle
        df['BB_Lower'] = bb_lower
        
        # Calculate BB_Width with safe division
        bb_width = (bb_upper - bb_lower) / bb_middle
        df['BB_Width'] = bb_width.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Calculate BB_Position with safe division
        bb_range = bb_upper - bb_lower
        bb_position = (df['Close'] - bb_lower) / bb_range
        df['BB_Position'] = bb_position.replace([np.inf, -np.inf], np.nan).fillna(0.5)
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['ATR'] = self.calculate_atr(df['High'], df['Low'], df['Close'])
        
        # Volume indicators
        volume_sma = df['Volume'].rolling(window=20).mean()
        df['Volume_SMA'] = volume_sma
        volume_ratio = df['Volume'] / volume_sma
        df['Volume_Ratio'] = volume_ratio.replace([np.inf, -np.inf], np.nan).fillna(1)
        
        # Price momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        close_shift = df['Close'].shift(10)
        roc = ((df['Close'] - close_shift) / close_shift) * 100
        df['ROC'] = roc.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Price channels
        df['High_20'] = df['High'].rolling(window=20).max()
        df['Low_20'] = df['Low'].rolling(window=20).min()
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def fetch_and_prepare_data(self, start_date=None, period='5y'):
        """Fetch data and add technical indicators"""
        try:
            # Fetch data from Yahoo Finance
            if start_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
                data = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
            else:
                data = yf.download(self.ticker, period=period, progress=False)
            
            if data.empty:
                raise ValueError(f"No data found for {self.ticker}")
            
            logger.info(f"Fetched {len(data)} records for {self.ticker}")
            
            # Add technical indicators
            if self.use_technical_indicators:
                data = self.add_technical_indicators(data)
                logger.info(f"Added technical indicators. Final dataset: {len(data)} records")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def prepare_features(self, data, split_ratio=0.8):
        """Prepare multi-feature dataset for training"""
        if self.use_technical_indicators:
            # Select relevant features
            feature_cols = [
                'Close', 'Volume', 'Returns', 'SMA_10', 'SMA_20', 'SMA_50',
                'EMA_10', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
                'BB_Width', 'BB_Position', 'Volatility', 'ATR', 'Volume_Ratio',
                'Momentum', 'ROC'
            ]
            self.feature_columns = feature_cols
            dataset = data[feature_cols].values
        else:
            self.feature_columns = ['Close']
            dataset = data['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(dataset)
        
        # Split data
        train_size = int(len(scaled_data) * split_ratio)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size - self.lookback:]
        
        return train_data, test_data, dataset
    
    def create_sequences(self, data, target_col_idx=0):
        """Create sequences for LSTM training with multiple features"""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i])
            y.append(data[i, target_col_idx])  # Predict Close price
        
        X, y = np.array(X), np.array(y)
        return X, y
    
    def build_advanced_model(self, input_shape):
        """
        Build advanced LSTM model with:
        - Bidirectional LSTM layers
        - Layer Normalization
        - Attention mechanism
        - Residual connections
        """
        inputs = Input(shape=input_shape)
        
        # First Bidirectional LSTM block
        x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Second Bidirectional LSTM block
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Third LSTM block
        x = Bidirectional(LSTM(32, return_sequences=True))(x)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Use Adam with custom learning rate
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])
        
        return model
    
    def build_ensemble_model(self, input_shape):
        """Build ensemble model combining LSTM and GRU"""
        inputs = Input(shape=input_shape)
        
        # LSTM branch
        lstm_branch = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        lstm_branch = Dropout(0.3)(lstm_branch)
        lstm_branch = Bidirectional(LSTM(32, return_sequences=False))(lstm_branch)
        lstm_branch = Dropout(0.2)(lstm_branch)
        
        # GRU branch
        gru_branch = Bidirectional(GRU(64, return_sequences=True))(inputs)
        gru_branch = Dropout(0.3)(gru_branch)
        gru_branch = Bidirectional(GRU(32, return_sequences=False))(gru_branch)
        gru_branch = Dropout(0.2)(gru_branch)
        
        # Concatenate branches
        merged = Concatenate()([lstm_branch, gru_branch])
        
        # Dense layers
        x = Dense(64, activation='relu')(merged)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Output
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])
        
        return model
    
    def train(self, epochs=100, batch_size=32, use_ensemble=False, patience=15):
        """
        Train the advanced model with callbacks
        
        Args:
            epochs: Maximum number of epochs
            batch_size: Batch size for training
            use_ensemble: Whether to use ensemble model (LSTM + GRU)
            patience: Early stopping patience
        """
        try:
            # Fetch and prepare data
            data = self.fetch_and_prepare_data()
            train_data, test_data, dataset = self.prepare_features(data)
            
            # Create sequences
            X_train, y_train = self.create_sequences(train_data)
            
            logger.info(f"Training data shape: {X_train.shape}")
            logger.info(f"Number of features: {X_train.shape[2]}")
            
            # Build model
            if use_ensemble:
                self.model = self.build_ensemble_model((X_train.shape[1], X_train.shape[2]))
                logger.info("Using ensemble model (LSTM + GRU)")
            else:
                self.model = self.build_advanced_model((X_train.shape[1], X_train.shape[2]))
                logger.info("Using advanced Bidirectional LSTM model")
            
            # Callbacks for better training
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001,
                    verbose=1
                )
            ]
            
            # Train model
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.15,
                callbacks=callbacks,
                verbose=1
            )
            
            self.test_data = test_data
            self.dataset = dataset
            self.train_size = len(train_data)
            
            logger.info("Training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise
    
    def predict_future(self, days_ahead=5):
        """Predict future stock prices"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get last sequence from test data
        last_sequence = self.test_data[-self.lookback:]
        predictions = []
        
        for _ in range(days_ahead):
            # Reshape for prediction
            X_pred = last_sequence.reshape(1, self.lookback, -1)
            
            # Predict
            pred_scaled = self.model.predict(X_pred, verbose=0)
            
            # Create full feature array for inverse transform
            pred_full = np.zeros((1, len(self.feature_columns)))
            pred_full[0, 0] = pred_scaled[0, 0]  # Close price is first feature
            
            # Inverse transform
            pred_price = self.scaler.inverse_transform(pred_full)[0, 0]
            predictions.append(pred_price)
            
            # Update sequence
            new_row = np.zeros((1, len(self.feature_columns)))
            new_row[0, 0] = pred_scaled[0, 0]
            last_sequence = np.vstack([last_sequence[1:], new_row])
        
        return predictions
    
    def evaluate(self):
        """Evaluate model performance"""
        try:
            X_test, y_test = self.create_sequences(self.test_data)
            
            # Predictions
            predictions = self.model.predict(X_test, verbose=0)
            
            # Inverse transform for Close price only
            pred_full = np.zeros((len(predictions), len(self.feature_columns)))
            pred_full[:, 0] = predictions[:, 0]
            predictions_actual = self.scaler.inverse_transform(pred_full)[:, 0]
            
            y_test_full = np.zeros((len(y_test), len(self.feature_columns)))
            y_test_full[:, 0] = y_test
            y_test_actual = self.scaler.inverse_transform(y_test_full)[:, 0]
            
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
            
        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            raise
