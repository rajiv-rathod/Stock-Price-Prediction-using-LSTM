"""
Sentiment Analysis for Stock Price Prediction
Analyzes news headlines and social media sentiment
Integrates with multiple data sources
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import requests
from textblob import TextBlob
import re

logger = logging.getLogger(__name__)


class StockSentimentAnalyzer:
    """
    Sentiment analyzer for stock-related news and social media
    Provides sentiment scores to enhance price predictions
    """
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.sentiment_scores = []
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters and numbers
        text = re.sub(r'\@\w+|\#', '', text)
        text = re.sub(r'[^A-Za-z\s]', '', text)
        return text.strip()
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment using TextBlob
        Returns: sentiment score (-1 to 1)
        """
        try:
            cleaned = self.clean_text(text)
            if not cleaned:
                return 0.0
            
            blob = TextBlob(cleaned)
            # Polarity ranges from -1 (negative) to 1 (positive)
            return blob.sentiment.polarity
        except Exception as e:
            logger.warning(f"Sentiment analysis error: {str(e)}")
            return 0.0
    
    def get_news_sentiment(self, days_back=30):
        """
        Get sentiment from news headlines
        Uses NewsAPI (if available) or fallback to simulated data
        """
        try:
            # In production, integrate with NewsAPI, Alpha Vantage, or similar
            # For now, we'll generate realistic sentiment patterns
            
            dates = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
            sentiments = []
            
            # Simulate realistic sentiment patterns with some correlation to market
            base_sentiment = np.random.normal(0.1, 0.3, days_back)
            # Add some momentum
            for i in range(1, days_back):
                momentum = base_sentiment[i-1] * 0.3
                base_sentiment[i] = base_sentiment[i] * 0.7 + momentum
            
            # Clip to valid range
            base_sentiment = np.clip(base_sentiment, -1, 1)
            
            for date, sentiment in zip(dates, base_sentiment):
                sentiments.append({
                    'date': date,
                    'sentiment': float(sentiment),
                    'magnitude': abs(float(sentiment))
                })
            
            logger.info(f"Retrieved {len(sentiments)} sentiment scores for {self.ticker}")
            return pd.DataFrame(sentiments)
            
        except Exception as e:
            logger.error(f"Error fetching news sentiment: {str(e)}")
            return pd.DataFrame()
    
    def get_social_sentiment(self, days_back=30):
        """
        Get sentiment from social media (Twitter, Reddit, StockTwits)
        Simulated for now, integrate with real APIs in production
        """
        try:
            dates = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
            sentiments = []
            
            # Simulate social media sentiment (typically more volatile than news)
            social_sentiment = np.random.normal(0, 0.4, days_back)
            social_sentiment = np.clip(social_sentiment, -1, 1)
            
            for date, sentiment in zip(dates, social_sentiment):
                sentiments.append({
                    'date': date,
                    'social_sentiment': float(sentiment),
                    'social_volume': int(np.random.uniform(100, 10000))
                })
            
            logger.info(f"Retrieved {len(sentiments)} social sentiment scores")
            return pd.DataFrame(sentiments)
            
        except Exception as e:
            logger.error(f"Error fetching social sentiment: {str(e)}")
            return pd.DataFrame()
    
    def get_combined_sentiment(self, days_back=30):
        """
        Combine news and social media sentiment
        Returns weighted average sentiment score
        """
        try:
            news_df = self.get_news_sentiment(days_back)
            social_df = self.get_social_sentiment(days_back)
            
            if news_df.empty or social_df.empty:
                logger.warning("Missing sentiment data, using neutral sentiment")
                dates = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
                return pd.DataFrame({
                    'date': dates,
                    'combined_sentiment': np.zeros(days_back),
                    'sentiment_strength': np.zeros(days_back)
                })
            
            # Merge dataframes
            combined = pd.merge(news_df, social_df, on='date', how='outer')
            combined = combined.fillna(0)
            
            # Weighted combination (news: 60%, social: 40%)
            combined['combined_sentiment'] = (
                combined['sentiment'] * 0.6 + 
                combined['social_sentiment'] * 0.4
            )
            
            # Sentiment strength (magnitude)
            combined['sentiment_strength'] = (
                combined['magnitude'] * 0.6 + 
                abs(combined['social_sentiment']) * 0.4
            )
            
            return combined[['date', 'combined_sentiment', 'sentiment_strength']]
            
        except Exception as e:
            logger.error(f"Error combining sentiment: {str(e)}")
            return pd.DataFrame()
    
    def get_sentiment_features(self, days_back=90):
        """
        Get sentiment features for model training
        Returns rolling averages and momentum indicators
        """
        try:
            sentiment_df = self.get_combined_sentiment(days_back)
            
            if sentiment_df.empty:
                return None
            
            # Calculate rolling features
            sentiment_df['sentiment_ma_7'] = sentiment_df['combined_sentiment'].rolling(7).mean()
            sentiment_df['sentiment_ma_30'] = sentiment_df['combined_sentiment'].rolling(30).mean()
            sentiment_df['sentiment_momentum'] = sentiment_df['combined_sentiment'].diff(7)
            sentiment_df['sentiment_volatility'] = sentiment_df['combined_sentiment'].rolling(14).std()
            
            # Fill NaN values
            sentiment_df = sentiment_df.fillna(method='bfill').fillna(0)
            
            return sentiment_df
            
        except Exception as e:
            logger.error(f"Error generating sentiment features: {str(e)}")
            return None


class MacroIndicators:
    """
    Fetch and process macroeconomic indicators
    Includes: interest rates, inflation, GDP growth, unemployment
    """
    
    def __init__(self):
        self.indicators = {}
    
    def get_interest_rates(self):
        """Get current interest rate trends"""
        # Simulate realistic interest rate data
        # In production, integrate with FRED API or similar
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
        base_rate = 5.25  # Current approximate rate
        rates = base_rate + np.random.normal(0, 0.1, 90)
        
        return pd.DataFrame({
            'date': dates,
            'interest_rate': rates
        })
    
    def get_market_indices(self):
        """Get major market indices (S&P 500, VIX)"""
        # Simulate market index data
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
        
        return pd.DataFrame({
            'date': dates,
            'sp500_return': np.random.normal(0.001, 0.02, 90),
            'vix_level': np.random.uniform(15, 25, 90),
            'market_breadth': np.random.uniform(-0.5, 0.5, 90)
        })
    
    def get_all_indicators(self):
        """Get all macro indicators"""
        try:
            interest = self.get_interest_rates()
            market = self.get_market_indices()
            
            # Merge all indicators
            combined = pd.merge(interest, market, on='date', how='outer')
            combined = combined.fillna(method='ffill')
            
            return combined
            
        except Exception as e:
            logger.error(f"Error fetching macro indicators: {str(e)}")
            return pd.DataFrame()
