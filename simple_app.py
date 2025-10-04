"""
Simple Flask App for Stock Price Prediction
Upload CSV → Get Predictions
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def create_lstm_model(input_shape):
    """Create a simple LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def prepare_data(data, lookback=60):
    """Prepare data for LSTM"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler


@app.route('/')
def index():
    """Main page"""
    return render_template('simple_index.html')


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
        
        # Read CSV
        df = pd.read_csv(file)
        logger.info(f"CSV loaded with shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Validate CSV has required column
        if 'Close' not in df.columns:
            return jsonify({
                'success': False, 
                'error': 'CSV must have a "Close" column with stock prices',
                'columns_found': df.columns.tolist()
            }), 400
        
        # Get parameters
        lookback = int(request.form.get('lookback', 60))
        epochs = int(request.form.get('epochs', 50))
        days_ahead = int(request.form.get('days_ahead', 5))
        
        # Prepare data
        prices = df['Close'].values
        if len(prices) < lookback + 20:
            return jsonify({
                'success': False,
                'error': f'Need at least {lookback + 20} data points. Got {len(prices)}'
            }), 400
        
        X, y, scaler = prepare_data(prices, lookback)
        
        # Split train/test (80/20)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Create and train model
        model = create_lstm_model((lookback, 1))
        logger.info("Training model...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Make predictions on test set
        predictions = model.predict(X_test, verbose=0)
        predictions = scaler.inverse_transform(predictions)
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        mse = np.mean((predictions - actual) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actual))
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        
        # Predict future
        last_sequence = scaled_data[-lookback:]
        future_predictions = []
        
        for _ in range(days_ahead):
            last_sequence_reshaped = last_sequence.reshape(1, lookback, 1)
            next_pred = model.predict(last_sequence_reshaped, verbose=0)
            future_predictions.append(next_pred[0, 0])
            last_sequence = np.append(last_sequence[1:], next_pred)
        
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        
        # Prepare response
        response = {
            'success': True,
            'data_points': len(prices),
            'lookback': lookback,
            'epochs': epochs,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'metrics': {
                'MSE': float(mse),
                'RMSE': float(rmse),
                'MAE': float(mae),
                'MAPE': float(mape)
            },
            'predictions': {
                'actual': actual.flatten().tolist(),
                'predicted': predictions.flatten().tolist()
            },
            'future_predictions': future_predictions.flatten().tolist(),
            'training_history': {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']]
            }
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
