#!/usr/bin/env python3
"""
API Test Script for Stock Price Prediction Web Application
Tests all API endpoints and verifies functionality
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:5000"

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_success(text):
    """Print success message"""
    print(f"✓ {text}")

def print_error(text):
    """Print error message"""
    print(f"✗ {text}")

def print_info(text):
    """Print info message"""
    print(f"ℹ {text}")

def test_health_check():
    """Test the health check endpoint"""
    print_header("Testing Health Check Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success(f"Health check passed: {data['status']}")
            print_info(f"Service: {data['service']}")
            print_info(f"Timestamp: {data['timestamp']}")
            return True
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_error(f"Connection failed: {str(e)}")
        return False

def test_stock_info(ticker="AAPL"):
    """Test the stock info endpoint"""
    print_header(f"Testing Stock Info Endpoint ({ticker})")
    try:
        response = requests.get(f"{BASE_URL}/api/stock-info/{ticker}", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                info = data['data']
                print_success(f"Stock info retrieved successfully")
                print_info(f"Name: {info.get('name', 'N/A')}")
                print_info(f"Symbol: {info.get('symbol', 'N/A')}")
                print_info(f"Latest Price: ${info.get('latest_price', 0):.2f}")
                print_info(f"Change: ${info.get('change', 0):.2f} ({info.get('change_pct', 0):.2f}%)")
                return True
            else:
                print_error(f"API returned error: {data.get('error', 'Unknown error')}")
                return False
        else:
            print_error(f"Request failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_error(f"Connection failed: {str(e)}")
        return False

def test_prediction(ticker="AAPL", lookback=30, epochs=20, days_ahead=3):
    """Test the prediction endpoint (with reduced parameters for faster testing)"""
    print_header(f"Testing Prediction Endpoint ({ticker})")
    print_info("Using reduced parameters for faster testing:")
    print_info(f"  Lookback: {lookback} days")
    print_info(f"  Epochs: {epochs}")
    print_info(f"  Forecast: {days_ahead} days")
    print_info("This may take 30-60 seconds...")
    
    try:
        payload = {
            "ticker": ticker,
            "lookback": lookback,
            "epochs": epochs,
            "days_ahead": days_ahead
        }
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/predict",
            json=payload,
            timeout=120
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print_success(f"Prediction completed in {elapsed:.1f} seconds")
                print_info(f"Stock: {data['stock_info']['name']}")
                print_info(f"Current Price: ${data['current_price']:.2f}")
                print_info(f"Predicted Price (Day 1): ${data['predicted_price']:.2f}")
                print_info(f"Expected Change: ${data['price_change']:.2f} ({data['price_change_pct']:.2f}%)")
                print_info(f"\nMetrics:")
                print_info(f"  MAPE: {data['metrics']['mape']}%")
                print_info(f"  RMSE: {data['metrics']['rmse']}")
                print_info(f"  MAE: {data['metrics']['mae']}")
                print_info(f"  Directional Accuracy: {data['metrics']['directional_accuracy']}%")
                
                print_info(f"\nFuture Predictions ({days_ahead} days):")
                for date, price in zip(data['future_predictions']['dates'], 
                                      data['future_predictions']['prices']):
                    print_info(f"  {date}: ${price:.2f}")
                
                return True
            else:
                print_error(f"API returned error: {data.get('error', 'Unknown error')}")
                return False
        else:
            error_data = response.json()
            print_error(f"Request failed: {response.status_code}")
            print_error(f"Error: {error_data.get('error', 'Unknown error')}")
            return False
    except requests.exceptions.Timeout:
        print_error("Request timed out (exceeded 120 seconds)")
        return False
    except requests.exceptions.RequestException as e:
        print_error(f"Connection failed: {str(e)}")
        return False
    except json.JSONDecodeError:
        print_error("Invalid JSON response")
        return False

def test_web_interface():
    """Test if the web interface is accessible"""
    print_header("Testing Web Interface")
    try:
        response = requests.get(BASE_URL, timeout=5)
        if response.status_code == 200:
            if "Stock Price Predictor" in response.text or "stock" in response.text.lower():
                print_success("Web interface is accessible")
                print_info(f"URL: {BASE_URL}")
                return True
            else:
                print_error("Web interface loaded but content unexpected")
                return False
        else:
            print_error(f"Web interface returned status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_error(f"Connection failed: {str(e)}")
        return False

def test_invalid_ticker():
    """Test error handling with invalid ticker"""
    print_header("Testing Error Handling (Invalid Ticker)")
    try:
        response = requests.get(f"{BASE_URL}/api/stock-info/INVALID123", timeout=10)
        if response.status_code in [400, 404]:
            print_success("API correctly handled invalid ticker")
            return True
        elif response.status_code == 200:
            print_info("API returned 200 - may have fetched data or handled gracefully")
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_error(f"Connection failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  Stock Price Prediction API Test Suite")
    print("="*60)
    print(f"Testing server at: {BASE_URL}")
    
    results = []
    
    # Test 1: Health Check
    results.append(("Health Check", test_health_check()))
    time.sleep(1)
    
    # Test 2: Web Interface
    results.append(("Web Interface", test_web_interface()))
    time.sleep(1)
    
    # Test 3: Stock Info
    results.append(("Stock Info (AAPL)", test_stock_info("AAPL")))
    time.sleep(1)
    
    # Test 4: Error Handling
    results.append(("Error Handling", test_invalid_ticker()))
    time.sleep(1)
    
    # Test 5: Prediction (optional, takes longer)
    print_info("\nPrediction test is optional (takes 30-60 seconds)")
    user_input = input("Run prediction test? (y/n): ").strip().lower()
    if user_input == 'y':
        results.append(("Prediction (AAPL)", test_prediction("AAPL", 30, 20, 3)))
    else:
        print_info("Skipping prediction test")
        results.append(("Prediction", None))
    
    # Summary
    print_header("Test Summary")
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)
    total = len(results) - skipped
    
    for test_name, result in results:
        if result is True:
            print_success(f"{test_name}: PASSED")
        elif result is False:
            print_error(f"{test_name}: FAILED")
        else:
            print_info(f"{test_name}: SKIPPED")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if failed == 0 and passed > 0:
        print_success("All tests passed! 🎉")
        return 0
    elif failed > 0:
        print_error(f"{failed} test(s) failed")
        return 1
    else:
        print_info("No tests completed")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest suite interrupted by user")
        sys.exit(1)
