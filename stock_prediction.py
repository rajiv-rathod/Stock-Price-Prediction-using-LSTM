import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from datetime import datetime, timedelta


class StockPricePredictor:
    """LSTM-based Stock Price Predictor"""
    
    def __init__(self, ticker='AAPL', lookback=60):
        """
        Initialize the Stock Price Predictor
        
        Args:
            ticker (str): Stock ticker symbol
            lookback (int): Number of days to look back for prediction
        """
        self.ticker = ticker
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None
        
    def fetch_data(self, start_date='2015-01-01', end_date=None):
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: Stock data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"Fetching data for {self.ticker} from {start_date} to {end_date}...")
        data = yf.download(self.ticker, start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data found for ticker {self.ticker}")
            
        print(f"Data fetched successfully: {len(data)} records")
        return data
    
    def prepare_data(self, data, split_ratio=0.8):
        """
        Prepare data for LSTM training
        
        Args:
            data (pd.DataFrame): Stock data
            split_ratio (float): Train/test split ratio
            
        Returns:
            tuple: Training and testing data
        """
        # Use Close price for prediction
        dataset = data['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(dataset)
        
        # Split data
        train_size = int(len(scaled_data) * split_ratio)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size - self.lookback:]
        
        return train_data, test_data, dataset
    
    def create_sequences(self, data):
        """
        Create sequences for LSTM training
        
        Args:
            data (np.array): Scaled data
            
        Returns:
            tuple: X and y arrays
        """
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i, 0])
            y.append(data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def build_model(self, input_shape):
        """
        Build LSTM model
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            Sequential: Compiled LSTM model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        
        # Second LSTM layer
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        
        # Third LSTM layer
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
    
    def train(self, epochs=50, batch_size=32):
        """
        Train the LSTM model
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        print("\nPreparing data...")
        data = self.fetch_data()
        train_data, test_data, dataset = self.prepare_data(data)
        
        print("Creating sequences...")
        X_train, y_train = self.create_sequences(train_data)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"\nBuilding model...")
        self.model = self.build_model((X_train.shape[1], 1))
        
        print(f"\nTraining model for {epochs} epochs...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        
        # Store data for prediction
        self.test_data = test_data
        self.dataset = dataset
        self.train_size = len(train_data)
        
        print("\nTraining completed!")
        
    def predict(self):
        """
        Make predictions on test data
        
        Returns:
            np.array: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        print("\nMaking predictions...")
        X_test, y_test = self.create_sequences(self.test_data)
        
        predictions = self.model.predict(X_test)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions
    
    def visualize_results(self, predictions):
        """
        Visualize predictions vs actual prices
        
        Args:
            predictions (np.array): Model predictions
        """
        # Get actual test data
        train_size = self.train_size
        actual = self.dataset[train_size:]
        
        # Plot
        plt.figure(figsize=(14, 5))
        
        # Plot 1: Full data
        plt.subplot(1, 2, 1)
        plt.plot(self.dataset, label='Actual Price', color='blue', alpha=0.6)
        plt.axvline(x=train_size, color='red', linestyle='--', label='Train/Test Split')
        plt.title(f'{self.ticker} Stock Price - Full Data')
        plt.xlabel('Days')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Predictions vs Actual
        plt.subplot(1, 2, 2)
        plt.plot(actual, label='Actual Price', color='blue')
        plt.plot(predictions, label='Predicted Price', color='red')
        plt.title(f'{self.ticker} Stock Price Prediction')
        plt.xlabel('Days')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('stock_prediction_results.png', dpi=300, bbox_inches='tight')
        print("\nResults saved to 'stock_prediction_results.png'")
        plt.show()
    
    def visualize_training_history(self):
        """Visualize training history"""
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
            
        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("Training history saved to 'training_history.png'")
        plt.show()
    
    def calculate_metrics(self, predictions):
        """
        Calculate performance metrics
        
        Args:
            predictions (np.array): Model predictions
        """
        train_size = self.train_size
        actual = self.dataset[train_size:]
        
        # Calculate metrics
        mse = np.mean((actual - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual - predictions))
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        print(f"Mean Squared Error (MSE):  {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE):  {mae:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print("="*50)


def main():
    """Main function to run the stock prediction"""
    # Initialize predictor
    print("="*60)
    print("Stock Price Prediction using LSTM")
    print("="*60)
    
    predictor = StockPricePredictor(ticker='AAPL', lookback=60)
    
    # Train the model
    predictor.train(epochs=50, batch_size=32)
    
    # Make predictions
    predictions = predictor.predict()
    
    # Calculate metrics
    predictor.calculate_metrics(predictions)
    
    # Visualize results
    predictor.visualize_training_history()
    predictor.visualize_results(predictions)
    
    print("\n✓ Stock price prediction completed successfully!")


if __name__ == "__main__":
    main()
