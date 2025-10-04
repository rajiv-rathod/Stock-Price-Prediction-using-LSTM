"""
Transformer-Based Stock Price Prediction Model
State-of-the-art architecture with multi-head attention for time series forecasting
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, 
    MultiHeadAttention, GlobalAveragePooling1D,
    Add, Concatenate, Conv1D, MaxPooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging

logger = logging.getLogger(__name__)


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block with multi-head attention and feed-forward network"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        # Multi-head attention
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding for transformer input"""
    
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
    
    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=np.arange(position)[:, np.newaxis],
            i=np.arange(d_model)[np.newaxis, :],
            d_model=d_model
        )
        
        # Apply sin to even indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class TransformerStockPredictor:
    """
    Advanced Transformer-based stock price predictor
    Uses multi-head attention mechanism for capturing complex temporal patterns
    """
    
    def __init__(self, lookback=60, num_features=19):
        self.lookback = lookback
        self.num_features = num_features
        self.model = None
        self.history = None
    
    def build_transformer_model(self, input_shape):
        """
        Build transformer model with:
        - Positional encoding
        - Multiple transformer blocks
        - CNN feature extraction
        - Dense prediction layers
        """
        inputs = Input(shape=input_shape)
        
        # Positional encoding
        x = PositionalEncoding(position=input_shape[0], d_model=input_shape[1])(inputs)
        
        # Transformer blocks
        x = TransformerBlock(embed_dim=input_shape[1], num_heads=4, ff_dim=128, dropout_rate=0.1)(x)
        x = TransformerBlock(embed_dim=input_shape[1], num_heads=4, ff_dim=128, dropout_rate=0.1)(x)
        x = TransformerBlock(embed_dim=input_shape[1], num_heads=4, ff_dim=64, dropout_rate=0.1)(x)
        
        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.1)(x)
        
        # Output
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])
        
        return model
    
    def build_hybrid_model(self, input_shape):
        """
        Hybrid architecture: CNN + Transformer + LSTM
        Best of all worlds for time series forecasting
        """
        inputs = Input(shape=input_shape)
        
        # CNN branch for local pattern extraction
        cnn = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(cnn)
        cnn = GlobalAveragePooling1D()(cnn)
        
        # Transformer branch for global attention
        transformer = PositionalEncoding(position=input_shape[0], d_model=input_shape[1])(inputs)
        transformer = TransformerBlock(embed_dim=input_shape[1], num_heads=4, ff_dim=128, dropout_rate=0.1)(transformer)
        transformer = TransformerBlock(embed_dim=input_shape[1], num_heads=4, ff_dim=64, dropout_rate=0.1)(transformer)
        transformer = GlobalAveragePooling1D()(transformer)
        
        # LSTM branch for sequential patterns
        from tensorflow.keras.layers import LSTM, Bidirectional
        lstm = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        lstm = Dropout(0.2)(lstm)
        lstm = Bidirectional(LSTM(32, return_sequences=False))(lstm)
        lstm = Dropout(0.2)(lstm)
        
        # Merge all branches
        merged = Concatenate()([cnn, transformer, lstm])
        
        # Dense layers
        x = Dense(128, activation='relu')(merged)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.1)(x)
        
        # Output
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])
        
        return model
    
    def train(self, X_train, y_train, epochs=100, batch_size=32, use_hybrid=True):
        """
        Train the transformer model
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            epochs: Maximum epochs
            batch_size: Batch size
            use_hybrid: Use hybrid CNN+Transformer+LSTM architecture
        """
        try:
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            if use_hybrid:
                self.model = self.build_hybrid_model(input_shape)
                logger.info("Using Hybrid CNN+Transformer+LSTM model")
            else:
                self.model = self.build_transformer_model(input_shape)
                logger.info("Using Pure Transformer model")
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=0.00001,
                    verbose=1
                )
            ]
            
            # Train
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.15,
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info("Transformer training completed")
            return True
            
        except Exception as e:
            logger.error(f"Transformer training error: {str(e)}")
            raise
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        return {
            'loss': results[0],
            'mae': results[1],
            'mse': results[2]
        }
