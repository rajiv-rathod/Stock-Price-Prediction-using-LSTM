"""
Professional Stock Price Prediction System
Advanced Machine Learning Models with Ensemble Predictions
"""

# Standard library imports
import os
import logging
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core data processing
try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"Error importing core libraries: {e}")
    print("Please install: pip install numpy pandas")
    exit(1)

# Web framework
try:
    from flask import Flask, render_template, request, jsonify
    from flask_cors import CORS
except ImportError as e:
    print(f"Error importing Flask: {e}")
    print("Please install: pip install Flask flask-cors")
    exit(1)

# Machine learning libraries
try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import ElasticNet
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
except ImportError as e:
    print(f"Error importing scikit-learn: {e}")
    print("Please install: pip install scikit-learn")
    exit(1)

# Advanced ML models
try:
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostRegressor
except ImportError as e:
    print(f"Error importing advanced ML libraries: {e}")
    print("Please install: pip install xgboost lightgbm catboost")
    exit(1)

# Technical analysis and scientific computing
try:
    import ta
    from scipy import stats
    from scipy.signal import argrelextrema
    import yfinance as yf
except ImportError as e:
    print(f"Error importing technical analysis libraries: {e}")
    print("Please install: pip install ta scipy yfinance")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class AdvancedStockPredictor:
    """Professional-grade stock prediction system with ensemble methods"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def create_advanced_features(self, df):
        """Create comprehensive technical indicators and features"""
        data = df.copy()
        
        # Ensure numeric data types
        for col in data.columns:
            if col.lower() not in ['date', 'time', 'symbol', 'ticker']:
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                except:
                    pass
        
        # Ensure we have OHLCV data
        if 'Close' not in data.columns:
            # Auto-detect price column
            price_cols = [col for col in data.columns if any(x in col.lower() for x in ['close', 'price', 'value'])]
            if price_cols:
                data['Close'] = pd.to_numeric(data[price_cols[0]], errors='coerce')
            else:
                raise ValueError("Cannot find price column")
        
        # Ensure Close column is numeric and clean
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        data = data.dropna(subset=['Close'])
        
        # Fill missing OHLCV if not present
        if 'Open' not in data.columns:
            data['Open'] = data['Close'].shift(1).fillna(data['Close'])
        if 'High' not in data.columns:
            data['High'] = data['Close'] * 1.02
        if 'Low' not in data.columns:
            data['Low'] = data['Close'] * 0.98
        if 'Volume' not in data.columns:
            data['Volume'] = 1000000
            
        # Ensure all OHLCV columns are numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
            
        # Technical Indicators
        try:
            # Trend Indicators
            data['SMA_5'] = ta.trend.sma_indicator(data['Close'], window=5)
            data['SMA_10'] = ta.trend.sma_indicator(data['Close'], window=10)
            data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
            data['EMA_12'] = ta.trend.ema_indicator(data['Close'], window=12)
            data['EMA_26'] = ta.trend.ema_indicator(data['Close'], window=26)
            
            # MACD
            data['MACD'] = ta.trend.macd_diff(data['Close'])
            data['MACD_signal'] = ta.trend.macd_signal(data['Close'])
            
            # Bollinger Bands
            data['BB_upper'] = ta.volatility.bollinger_hband(data['Close'])
            data['BB_lower'] = ta.volatility.bollinger_lband(data['Close'])
            data['BB_width'] = data['BB_upper'] - data['BB_lower']
            data['BB_position'] = (data['Close'] - data['BB_lower']) / data['BB_width']
            
            # RSI
            data['RSI'] = ta.momentum.rsi(data['Close'])
            data['RSI_SMA'] = ta.trend.sma_indicator(data['RSI'], window=14)
            
            # Stochastic
            data['Stoch_K'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
            data['Stoch_D'] = ta.momentum.stoch_signal(data['High'], data['Low'], data['Close'])
            
            # Williams %R
            data['Williams_R'] = ta.momentum.williams_r(data['High'], data['Low'], data['Close'])
            
            # Volume indicators
            data['Volume_SMA'] = ta.trend.sma_indicator(data['Volume'], window=20)
            data['Volume_ratio'] = data['Volume'] / data['Volume_SMA']
            
            # Price patterns
            data['Price_change'] = data['Close'].pct_change()
            data['Price_change_3'] = data['Close'].pct_change(3)
            data['Price_change_5'] = data['Close'].pct_change(5)
            
            # Volatility
            data['Volatility'] = data['Price_change'].rolling(window=20).std()
            data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
            
            # Support/Resistance levels
            for window in [5, 10, 20]:
                data[f'High_{window}'] = data['High'].rolling(window=window).max()
                data[f'Low_{window}'] = data['Low'].rolling(window=window).min()
                data[f'Range_{window}'] = data[f'High_{window}'] - data[f'Low_{window}']
            
            # Advanced features
            data['Close_to_SMA20'] = data['Close'] / data['SMA_20']
            data['Close_to_SMA50'] = data['Close'] / data['SMA_50']
            data['SMA20_to_SMA50'] = data['SMA_20'] / data['SMA_50']
            
            # Momentum features
            data['ROC_5'] = ta.momentum.roc(data['Close'], window=5)
            data['ROC_10'] = ta.momentum.roc(data['Close'], window=10)
            
        except Exception as e:
            print(f"Warning: Some technical indicators failed: {e}")
            
        # Clean data thoroughly
        try:
            # Replace infinite values
            data = data.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values
            data = data.fillna(method='bfill').fillna(method='ffill').fillna(0)
            
            # Ensure all values are finite
            for col in data.columns:
                if data[col].dtype in ['float64', 'int64']:
                    data[col] = data[col].astype('float64')
                    mask = ~np.isfinite(data[col])
                    if mask.any():
                        data.loc[mask, col] = data[col].median()
                        
        except Exception as e:
            print(f"Warning: Data cleaning failed: {e}")
            # Fallback: simple cleaning
            data = data.fillna(0)
        
        return data
    
    def create_ensemble_models(self):
        """Create multiple advanced models for ensemble prediction"""
        models = {
            'XGBoost': xgb.XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            'CatBoost': CatBoostRegressor(
                iterations=300,
                depth=8,
                learning_rate=0.05,
                random_seed=42,
                verbose=False
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                random_state=42
            ),
            'ElasticNet': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=42
            ),
            'SVR': SVR(
                kernel='rbf',
                C=100,
                gamma='scale'
            )
        }
        return models
        
    def train_ensemble(self, X_train, y_train, X_test, y_test):
        """Train ensemble of models and return predictions"""
        models = self.create_ensemble_models()
        predictions = {}
        scores = {}
        
        for name, model in models.items():
            try:
                print(f"Training {name}...")
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, pred)
                mae = mean_absolute_error(y_test, pred)
                r2 = r2_score(y_test, pred)
                
                predictions[name] = pred
                scores[name] = {'MSE': mse, 'MAE': mae, 'R2': r2}
                self.models[name] = model
                
                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        # Ensemble prediction (weighted average based on R2 scores)
        if predictions:
            weights = {}
            total_r2 = sum([max(0, scores[name]['R2']) for name in predictions.keys()])
            
            if total_r2 > 0:
                for name in predictions.keys():
                    weights[name] = max(0, scores[name]['R2']) / total_r2
            else:
                # Equal weights if all R2 are negative
                weights = {name: 1/len(predictions) for name in predictions.keys()}
            
            # Weighted ensemble prediction
            ensemble_pred = np.zeros_like(list(predictions.values())[0])
            for name, pred in predictions.items():
                ensemble_pred += pred * weights[name]
            
            predictions['Ensemble'] = ensemble_pred
            
            # Calculate ensemble metrics
            ensemble_mse = mean_squared_error(y_test, ensemble_pred)
            ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
            ensemble_r2 = r2_score(y_test, ensemble_pred)
            scores['Ensemble'] = {'MSE': ensemble_mse, 'MAE': ensemble_mae, 'R2': ensemble_r2}
        
        return predictions, scores
    
    def predict_future(self, data, models, scalers, days_ahead=5):
        """Predict future prices using ensemble models"""
        future_predictions = {name: [] for name in models.keys()}
        
        # Get last known features (make a copy to avoid read-only issues)
        last_features = np.array(data.iloc[-1:].values, copy=True)
        
        for day in range(days_ahead):
            day_predictions = {}
            
            for name, model in models.items():
                try:
                    # Scale features
                    if name in scalers:
                        scaled_features = scalers[name].transform(last_features)
                    else:
                        scaled_features = last_features
                    
                    # Predict
                    pred = model.predict(scaled_features)[0]
                    day_predictions[name] = pred
                    future_predictions[name].append(pred)
                    
                except Exception as e:
                    print(f"Error predicting with {name}: {e}")
                    future_predictions[name].append(np.nan)
            
            # Skip feature updating for simplicity - just use the same features
            # In a real scenario, you'd update technical indicators properly
        
        return future_predictions


predictor = AdvancedStockPredictor()


def detect_price_column(df):
    """Enhanced price column detection for ANY CSV format"""
    try:
        # Comprehensive price column patterns (multiple languages)
        price_patterns = [
            # English
            'close', 'closing', 'closing_price', 'price', 'adj_close', 'adjusted_close', 
            'adj close', 'adjusted close', 'value', 'amount', 'closing_value', 'close_price',
            'end_price', 'final_price', 'settlement', 'last', 'last_price',
            # Spanish
            'cierre', 'precio', 'precio_cierre', 'valor_cierre', 'precio_final',
            # French  
            'fermeture', 'prix', 'prix_fermeture', 'valeur_fermeture',
            # Portuguese
            'fechamento', 'preço', 'valor_fechamento', 'preço_fechamento',
            # German
            'schluss', 'preis', 'schlusspreis',
            # Other common patterns
            'cotação', 'quotation', 'rate', 'nivel', 'level'
        ]
        
        # Convert column names to lowercase for comparison
        columns_lower = [col.lower().strip().replace(' ', '_').replace('-', '_') for col in df.columns]
        
        # First priority: exact matches
        for pattern in price_patterns:
            for i, col_lower in enumerate(columns_lower):
                if col_lower == pattern or col_lower.endswith('_' + pattern) or col_lower.startswith(pattern + '_'):
                    return df.columns[i]
        
        # Second priority: partial matches
        for pattern in price_patterns:
            for i, col_lower in enumerate(columns_lower):
                if pattern in col_lower:
                    # Verify it's numeric
                    try:
                        pd.to_numeric(df.iloc[:, i], errors='coerce')
                        if df.iloc[:, i].notna().sum() > len(df) * 0.5:  # At least 50% non-null
                            return df.columns[i]
                    except:
                        continue
        
        # Third priority: find any numeric column that could be a price
        print("No standard price column found. Analyzing numeric columns...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            # Try to convert object columns to numeric
            for col in df.columns:
                try:
                    temp_series = pd.to_numeric(df[col], errors='coerce')
                    if temp_series.notna().sum() > len(df) * 0.5:
                        numeric_cols.append(col)
                except:
                    continue
        
        if not numeric_cols:
            raise ValueError("No numeric columns found that could represent prices")
        
        # Prefer columns with reasonable price-like properties
        for col in numeric_cols:
            try:
                values = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(values) > 0:
                    # Check if values are reasonable for prices (positive, not extreme)
                    if values.min() > 0 and values.max() < values.min() * 10000:
                        print(f"Selected numeric column '{col}' as price column")
                        return col
            except:
                continue
        
        # Last resort: take first numeric column
        if numeric_cols:
            print(f"Using first numeric column '{numeric_cols[0]}' as price column")
            return numeric_cols[0]
        
        raise ValueError(
            f"Could not find suitable price column. Available columns: {df.columns.tolist()}. "
            f"Please ensure your CSV has a column with price data (Close, Price, Value, etc.)"
        )
        
    except Exception as e:
        print(f"Error in price column detection: {e}")
        raise ValueError(f"Failed to detect price column: {str(e)}")


def calculate_rsi(data, periods=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_technical_indicators(prices, lookback=60):
    """Add technical indicators to improve model"""
    df = pd.DataFrame({'Close': prices})
    df['Returns'] = df['Close'].pct_change()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df = df.fillna(method='bfill').fillna(method='ffill')  # Fill NaN
    # For simplicity, use only Close and RSI for now
    features = df[['Close', 'RSI']].values[lookback:]
    return features


def prepare_data(prices, lookback=60):
    """Prepare data for LSTM"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler, scaled_data


@app.route('/')
def index():
    """Main page"""
    return render_template('advanced_index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle CSV upload and prediction"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'Only CSV files allowed'}), 400
        
        # Read CSV with enhanced error handling
        try:
            # Try different encodings if needed
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                return jsonify({'success': False, 'error': 'Could not read CSV file with any encoding'}), 400
                
            logger.info(f"CSV loaded with shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            # Validate minimum requirements
            if df.empty or df.shape[0] < 10:
                return jsonify({'success': False, 'error': 'CSV must have at least 10 rows of data'}), 400
                
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to read CSV: {str(e)}'}), 400

        # Clean and preprocess data with enhanced error handling
        try:
            # Remove any completely empty rows/columns
            original_shape = df.shape
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            if df.empty:
                return jsonify({'success': False, 'error': 'No valid data found after removing empty rows/columns'}), 400
            
            # Clean column names
            df.columns = df.columns.astype(str).str.strip().str.replace(r'[^\w\s]', '', regex=True).str.replace(r'\s+', '_', regex=True)
            
            # Convert columns to proper types with enhanced handling
            for col in df.columns:
                if col.lower() not in ['date', 'time', 'symbol', 'ticker', 'name']:
                    try:
                        # Skip if already numeric
                        if df[col].dtype in ['int64', 'float64']:
                            continue
                            
                        # Enhanced numeric conversion
                        if df[col].dtype == 'object':
                            # Handle various number formats
                            df[col] = df[col].astype(str)
                            # Remove common non-numeric characters
                            df[col] = df[col].str.replace(r'[$€£¥,]', '', regex=True)  # Currency symbols
                            df[col] = df[col].str.replace('%', '')  # Percentage signs
                            df[col] = df[col].str.replace(' ', '')  # Spaces
                            # Handle European decimal format (comma as decimal separator)
                            df[col] = df[col].str.replace(',', '.')
                            
                        # Convert to numeric
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        # Handle obvious price data errors
                        if any(keyword in col.lower() for keyword in ['price', 'close', 'open', 'high', 'low']):
                            # Remove negative prices and extreme outliers
                            median_val = df[col].median()
                            if pd.notna(median_val) and median_val > 0:
                                df.loc[df[col] <= 0, col] = np.nan
                                df.loc[df[col] > median_val * 1000, col] = np.nan
                                
                    except Exception as e:
                        logger.warning(f"Could not convert column {col} to numeric: {e}")
                        # Keep original column if conversion fails
                        continue
            
            # Remove rows where ALL numeric columns are NaN
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df = df.dropna(subset=numeric_cols.tolist(), how='all')
            
            if df.empty:
                return jsonify({'success': False, 'error': 'No valid numeric data found after cleaning'}), 400
            
            # Intelligent missing value handling
            for col in numeric_cols:
                if df[col].isna().any():
                    # Use forward fill for price-like data, median for others
                    if any(keyword in col.lower() for keyword in ['price', 'close', 'open', 'high', 'low']):
                        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                    else:
                        df[col] = df[col].fillna(df[col].median())
            
            logger.info(f"Data cleaned, final shape: {df.shape} (original: {original_shape})")
            logger.info(f"Data types: {df.dtypes.to_dict()}")
            
        except Exception as e:
            return jsonify({
                'success': False, 
                'error': f'Data cleaning failed: {str(e)}',
                'columns_found': df.columns.tolist()
            }), 400
        
        # Detect price column
        try:
            price_col = detect_price_column(df)
            logger.info(f"Detected price column: {price_col}")
        except ValueError as e:
            return jsonify({
                'success': False, 
                'error': str(e),
                'columns_found': df.columns.tolist()
            }), 400
        
        # Get parameters
        lookback = int(request.form.get('lookback', 60))
        epochs = int(request.form.get('epochs', 50))
        days_ahead = int(request.form.get('days_ahead', 5))
        
        # Enhanced model training with error handling
        try:
            # Prepare advanced features
            logger.info("Creating advanced features...")
            df_features = predictor.create_advanced_features(df)
            
            if df_features.empty or len(df_features) < lookback + 20:
                return jsonify({
                    'success': False,
                    'error': f'Insufficient data after feature creation. Need at least {lookback + 20} points, got {len(df_features)}'
                }), 400
            
            # Select feature columns (exclude target and non-numeric columns)
            feature_cols = []
            exclude_cols = ['Close', 'Date', 'ticker', 'symbol', 'name']
            
            for col in df_features.columns:
                if col not in exclude_cols:
                    try:
                        # Verify column is numeric and has sufficient data
                        if df_features[col].dtype in ['float64', 'int64']:
                            # Check for sufficient non-null values
                            non_null_ratio = df_features[col].notna().sum() / len(df_features)
                            if non_null_ratio > 0.5:  # At least 50% non-null
                                feature_cols.append(col)
                    except Exception as e:
                        logger.warning(f"Skipping column {col}: {e}")
                        continue
            
            if len(feature_cols) < 3:
                return jsonify({
                    'success': False,
                    'error': f'Insufficient valid features. Found {len(feature_cols)} features, need at least 3'
                }), 400
            
            logger.info(f"Training ensemble models on {len(df_features)} samples, testing on {len(df_features)//5} samples")
            logger.info(f"Using {len(feature_cols)} features: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")
            
            # Train ensemble model with error handling
            try:
                results = predictor.predict_prices(df_features, feature_cols, days_ahead)
                
                if results is None:
                    return jsonify({
                        'success': False,
                        'error': 'Model training failed - no results generated'
                    }), 500
                
                # Validate results
                required_keys = ['historical_data', 'predictions', 'metrics']
                missing_keys = [key for key in required_keys if key not in results]
                if missing_keys:
                    return jsonify({
                        'success': False,
                        'error': f'Incomplete results - missing: {missing_keys}'
                    }), 500
                
                # Ensure predictions are valid numbers
                if not results['predictions'] or any(not isinstance(p, (int, float)) or np.isnan(p) for p in results['predictions']):
                    return jsonify({
                        'success': False,
                        'error': 'Invalid predictions generated (NaN or non-numeric values)'
                    }), 500
                
                logger.info(f"Prediction complete. MAPE: {results['metrics'].get('MAPE', 'N/A'):.2f}%")
                
                return jsonify({
                    'success': True,
                    'data': results['historical_data'],
                    'predictions': results['predictions'],
                    'metrics': results['metrics'],
                    'features_used': len(feature_cols),
                    'data_points': len(df_features)
                })
                
            except Exception as e:
                logger.error(f"Model training error: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': f'Model training failed: {str(e)}',
                    'data_shape': df_features.shape,
                    'features_available': len(feature_cols)
                }), 500
                
        except Exception as e:
            logger.error(f"Feature preparation error: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Feature preparation failed: {str(e)}',
                'original_columns': df.columns.tolist()
            }), 500

    except Exception as e:
        logger.error(f"General request error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Request failed: {str(e)}'
        }), 500
        
        if len(feature_cols) < 5:
            # Fallback to basic features if technical indicators failed
            feature_cols = ['Open', 'High', 'Low', 'Volume'] if all(col in df_features.columns for col in ['Open', 'High', 'Low', 'Volume']) else [price_col]
        
        X = df_features[feature_cols].values
        y = df_features[price_col].values
        
        # Ensure X and y are numeric arrays
        try:
            X = X.astype(np.float64)
            y = y.astype(np.float64)
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f'Data conversion failed: {str(e)}. Please ensure all data is numeric.'
            }), 400
        
        # Remove any remaining NaN or infinite values
        try:
            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X, y = X[mask], y[mask]
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Data validation failed: {str(e)}'
            }), 400
        
        if len(X) < lookback + 20:
            return jsonify({
                'success': False,
                'error': f'Need at least {lookback + 20} valid data points. Got {len(X)}'
            }), 400
        
        # Split train/test (80/20)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        logger.info(f"Training ensemble models on {len(X_train)} samples, testing on {len(X_test)} samples")
        logger.info(f"Using {len(feature_cols)} features: {feature_cols[:10]}...")  # Show first 10 features
        
        # Train ensemble models
        predictions_dict, scores_dict = predictor.train_ensemble(X_train, y_train, X_test, y_test)
        
        # Use best performing model or ensemble
        best_model = 'Ensemble' if 'Ensemble' in predictions_dict else max(scores_dict.keys(), key=lambda x: scores_dict[x]['R2'])
        predictions = predictions_dict[best_model]
        
        # Calculate metrics for best model
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        r2 = r2_score(y_test, predictions)
        
        # Predict future prices
        logger.info("Generating future predictions...")
        future_predictions_dict = predictor.predict_future(
            df_features[feature_cols].iloc[-50:], 
            predictor.models, 
            {}, 
            days_ahead
        )
        
        # Use best model's future predictions
        future_predictions = future_predictions_dict.get(best_model, [])
        if not future_predictions:
            # Fallback: simple linear extrapolation
            trend = np.mean(np.diff(y_test[-10:]))
            future_predictions = [y_test[-1] + trend * (i+1) for i in range(days_ahead)]
        
        # Prepare response
        response = {
            'success': True,
            'data_points': len(df),
            'features_used': len(feature_cols),
            'lookback': lookback,
            'epochs': epochs,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'best_model': best_model,
            'metrics': {
                'MSE': float(mse),
                'RMSE': float(rmse),
                'MAE': float(mae),
                'MAPE': float(mape),
                'R2': float(r2)
            },
            'model_scores': {name: {k: float(v) for k, v in score.items()} for name, score in scores_dict.items()},
            'predictions': {
                'actual': y_test.tolist(),
                'predicted': predictions.tolist()
            },
            'future_predictions': future_predictions,
            'feature_importance': dict(zip(feature_cols[:10], predictor.feature_importance.get(best_model, [])[:10])) if hasattr(predictor, 'feature_importance') else {},
            'model_type': f'Advanced Ensemble ({best_model})'
        }
        
        logger.info(f"Prediction complete. MAPE: {mape:.2f}%")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health')
def health():
    """Health check"""
    return jsonify({'status': 'healthy', 'service': 'Simple Stock Predictor'})


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 SIMPLE STOCK PRICE PREDICTION")
    print("="*60)
    print("📊 Upload CSV → Get Predictions")
    print("="*60)
    print("Starting server...")
    print("Access: http://localhost:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
