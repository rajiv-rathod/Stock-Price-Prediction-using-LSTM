// Main JavaScript for Stock Price Prediction Web App
console.log('Stock Predictor AI - JavaScript Loaded');

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM Content Loaded');
    
    const predictionForm = document.getElementById('predictionForm');
    const csvUploadForm = document.getElementById('csvUploadForm');
    const predictBtn = document.getElementById('predictBtn');
    const csvPredictBtn = document.getElementById('csvPredictBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultsSection = document.getElementById('resultsSection');
    const errorAlert = document.getElementById('errorAlert');
    const tickerInput = document.getElementById('ticker');
    const searchResults = document.getElementById('searchResults');

    let searchTimeout;

    // Company search functionality
    if (tickerInput) {
        tickerInput.addEventListener('input', function(e) {
            const query = e.target.value.trim().toUpperCase();
            this.value = query; // Auto-uppercase
            
            clearTimeout(searchTimeout);
            
            if (query.length < 1) {
                searchResults.style.display = 'none';
                return;
            }
            
            searchTimeout = setTimeout(async () => {
                try {
                    const response = await fetch(`/api/search-companies?q=${encodeURIComponent(query)}`);
                    
                    if (!response.ok) {
                        console.error('Search failed');
                        return;
                    }
                    
                    const data = await response.json();
                    
                    if (data.success && data.results && data.results.length > 0) {
                        displaySearchResults(data.results);
                    } else {
                        searchResults.style.display = 'none';
                    }
                } catch (error) {
                    console.error('Search error:', error);
                }
            }, 300);
        });
    }

    function displaySearchResults(results) {
        searchResults.innerHTML = '';
        results.forEach(result => {
            const item = document.createElement('a');
            item.href = '#';
            item.className = 'list-group-item list-group-item-action';
            item.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <strong>${result.ticker}</strong>
                        <small class="text-muted ms-2">${result.name}</small>
                    </div>
                </div>
            `;
            item.addEventListener('click', function(e) {
                e.preventDefault();
                tickerInput.value = result.ticker;
                searchResults.style.display = 'none';
            });
            searchResults.appendChild(item);
        });
        searchResults.style.display = 'block';
    }

    // Hide search results when clicking outside
    document.addEventListener('click', function(e) {
        if (tickerInput && searchResults && !tickerInput.contains(e.target) && !searchResults.contains(e.target)) {
            searchResults.style.display = 'none';
        }
    });

    // Quick select buttons
    document.querySelectorAll('.stock-quick-btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.preventDefault();
            const ticker = this.getAttribute('data-ticker');
            if (tickerInput) {
                tickerInput.value = ticker;
            }
            // Switch to ticker tab if on CSV tab
            const tickerTab = document.getElementById('ticker-tab');
            if (tickerTab) {
                tickerTab.click();
            }
        });
    });

    // Ultra Mode toggle handler
    const ultraModeCheckbox = document.getElementById('useUltraMode');
    const standardModelOptions = document.getElementById('standardModelOptions');
    const ultraModeInfo = document.getElementById('ultraModeInfo');
    const standardModeInfo = document.getElementById('standardModeInfo');
    
    if (ultraModeCheckbox) {
        ultraModeCheckbox.addEventListener('change', function() {
            if (this.checked) {
                standardModelOptions.style.display = 'none';
                ultraModeInfo.style.display = 'block';
                standardModeInfo.style.display = 'none';
            } else {
                standardModelOptions.style.display = 'block';
                ultraModeInfo.style.display = 'none';
                standardModeInfo.style.display = 'block';
            }
        });
    }

    // Ticker prediction form submission
    if (predictionForm) {
        predictionForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            console.log('Prediction form submitted');
            await handlePrediction('ticker');
        });
    }

    // CSV upload form submission
    if (csvUploadForm) {
        csvUploadForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            console.log('CSV form submitted');
            await handlePrediction('csv');
        });
    }

    async function handlePrediction(type) {
        console.log('Starting prediction for type:', type);
        
        let requestData, headers, isFormData = false;
        let useAdvancedModel = false;
        
        if (type === 'csv') {
            const fileInput = document.getElementById('csvFile');
            if (!fileInput || !fileInput.files || !fileInput.files[0]) {
                showError('Please select a CSV file');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('ticker', 'CUSTOM');
            formData.append('lookback', document.getElementById('csvLookback').value);
            formData.append('epochs', document.getElementById('csvEpochs').value);
            formData.append('days_ahead', document.getElementById('csvDaysAhead').value);
            
            requestData = formData;
            isFormData = true;
        } else {
            const ticker = document.getElementById('ticker').value.trim().toUpperCase();
            const lookback = parseInt(document.getElementById('lookback').value);
            const epochs = parseInt(document.getElementById('epochs').value);
            const daysAhead = parseInt(document.getElementById('daysAhead').value);
            
            // Check Ultra Mode
            const useUltraMode = document.getElementById('useUltraMode')?.checked ?? false;
            
            // Get advanced model settings
            const useTechnicalIndicators = document.getElementById('useTechnicalIndicators')?.checked ?? true;
            const useEnsemble = document.getElementById('useEnsemble')?.checked ?? false;
            
            if (useUltraMode) {
                useAdvancedModel = 'ultra'; // Use ultra-advanced endpoint
            } else {
                useAdvancedModel = true; // Use advanced model for better results
            }

            // Validate inputs
            if (!ticker) {
                showError('Please enter a valid stock ticker symbol');
                return;
            }

            if (lookback < 10 || lookback > 200) {
                showError('Lookback period must be between 10 and 200 days');
                return;
            }

            if (epochs < 20 || epochs > 200) {
                showError('Training epochs must be between 20 and 200');
                return;
            }

            if (daysAhead < 1 || daysAhead > 30) {
                showError('Forecast days must be between 1 and 30');
                return;
            }

            const modelType = useUltraMode ? 'ULTRA-ADVANCED' : 'ADVANCED';
            console.log(`Sending ${modelType} prediction request:`, {
                ticker, lookback, epochs, daysAhead, 
                useUltraMode, useTechnicalIndicators, useEnsemble
            });

            if (useUltraMode) {
                requestData = JSON.stringify({
                    ticker: ticker,
                    lookback: lookback,
                    epochs: epochs,
                    days_ahead: daysAhead,
                    use_transformer: true,
                    use_lstm: true
                });
            } else {
                requestData = JSON.stringify({
                    ticker: ticker,
                    lookback: lookback,
                    epochs: epochs,
                    days_ahead: daysAhead,
                    use_technical_indicators: useTechnicalIndicators,
                    use_ensemble: useEnsemble
                });
            }
            headers = {'Content-Type': 'application/json'};
        }

        // Hide previous results and errors
        hideError();
        if (resultsSection) {
            resultsSection.style.display = 'none';
        }
        
        // Show loading spinner
        if (loadingSpinner) {
            loadingSpinner.style.display = 'block';
        }
        
        const activeBtn = type === 'csv' ? csvPredictBtn : predictBtn;
        if (activeBtn) {
            activeBtn.disabled = true;
            const btnText = useAdvancedModel === 'ultra' ? 'Training ULTRA Model...' : 'Training Advanced Model...';
            activeBtn.innerHTML = `<i class="fas fa-spinner fa-spin me-2"></i>${btnText}`;
        }

        try {
            console.log('Fetching prediction from API...');
            
            // Select endpoint based on model type
            let endpoint;
            if (useAdvancedModel === 'ultra') {
                endpoint = '/api/predict-ultra';
            } else if (useAdvancedModel) {
                endpoint = '/api/predict-advanced';
            } else {
                endpoint = '/api/predict';
            }
            console.log('Using endpoint:', endpoint);
            
            // Make API request
            const fetchOptions = {
                method: 'POST',
                body: requestData
            };
            
            if (!isFormData) {
                fetchOptions.headers = headers;
            }
            
            const response = await fetch(endpoint, fetchOptions);
            console.log('Response status:', response.status);

            // Check if response is OK
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({error: 'Unknown error occurred'}));
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Prediction data received:', data);

            if (!data.success) {
                throw new Error(data.error || 'Prediction failed');
            }

            // Display results
            displayResults(data);
            
            // Scroll to results
            setTimeout(() => {
                if (resultsSection) {
                    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }, 300);

        } catch (error) {
            console.error('Prediction error:', error);
            showError(error.message || 'Failed to generate prediction. Please try again.');
        } finally {
            // Hide loading spinner
            if (loadingSpinner) {
                loadingSpinner.style.display = 'none';
            }
            
            if (activeBtn) {
                activeBtn.disabled = false;
                activeBtn.innerHTML = type === 'csv' ? 
                    '<i class="fas fa-upload me-2"></i>Upload & Predict' : 
                    '<i class="fas fa-rocket me-2"></i>Generate Advanced Prediction';
            }
        }
    }

    function displayResults(data) {
        console.log('Displaying results...');
        
        // Stock info
        const stockNameEl = document.getElementById('stockName');
        if (stockNameEl) {
            const stockName = data.stock_info?.name || data.ticker || 'Stock';
            const modelType = data.model_type || 'LSTM';
            const featuresUsed = data.features_used || 1;
            stockNameEl.textContent = `${stockName} - ${modelType} (${featuresUsed} features)`;
        }
        
        // Price information
        const currentPriceEl = document.getElementById('currentPrice');
        if (currentPriceEl) {
            currentPriceEl.textContent = `$${data.current_price.toFixed(2)}`;
        }
        
        const predictedPriceEl = document.getElementById('predictedPrice');
        if (predictedPriceEl) {
            predictedPriceEl.textContent = `$${data.predicted_price.toFixed(2)}`;
        }
        
        const priceChangeEl = document.getElementById('priceChange');
        if (priceChangeEl) {
            const change = data.price_change;
            priceChangeEl.textContent = `${change >= 0 ? '+' : ''}$${change.toFixed(2)}`;
            priceChangeEl.className = change >= 0 ? 'mb-0 text-success' : 'mb-0 text-danger';
        }
        
        const priceChangePctEl = document.getElementById('priceChangePct');
        if (priceChangePctEl) {
            const changePct = data.price_change_pct;
            priceChangePctEl.textContent = `${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%`;
            priceChangePctEl.className = changePct >= 0 ? 'mb-0 text-success' : 'mb-0 text-danger';
        }

        // Metrics
        const mapeEl = document.getElementById('mape');
        if (mapeEl) {
            mapeEl.textContent = `${data.metrics.mape}%`;
        }
        
        const rmseEl = document.getElementById('rmse');
        if (rmseEl) {
            rmseEl.textContent = data.metrics.rmse.toFixed(4);
        }
        
        const maeEl = document.getElementById('mae');
        if (maeEl) {
            maeEl.textContent = data.metrics.mae.toFixed(4);
        }
        
        const dirAccEl = document.getElementById('directionalAccuracy');
        if (dirAccEl) {
            dirAccEl.textContent = `${data.metrics.directional_accuracy}%`;
        }

        // Interactive Chart.js charts
        // Price Chart
        const priceChartEl = document.getElementById('predictionChart');
        if (priceChartEl && data.historical_data && data.future_predictions) {
            const ctx = priceChartEl.getContext('2d');
            if (window.priceChartInstance) window.priceChartInstance.destroy();
            window.priceChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [...data.historical_data.dates, ...data.future_predictions.dates],
                    datasets: [
                        {
                            label: 'Actual',
                            data: data.historical_data.actual,
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16,185,129,0.1)',
                            pointRadius: 0,
                            fill: false,
                        },
                        {
                            label: 'Predicted',
                            data: [...data.historical_data.predicted, ...data.future_predictions.prices],
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102,126,234,0.1)',
                            pointRadius: 0,
                            fill: false,
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { position: 'top' },
                        title: { display: true, text: 'Stock Price Prediction' }
                    },
                    scales: {
                        x: { display: true, title: { display: true, text: 'Date' } },
                        y: { display: true, title: { display: true, text: 'Price ($)' } }
                    }
                }
            });
        }

        // Loss Chart
        const lossChartEl = document.getElementById('lossChart');
        if (lossChartEl && data.training_history) {
            const ctx = lossChartEl.getContext('2d');
            if (window.lossChartInstance) window.lossChartInstance.destroy();
            window.lossChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array.from({length: data.training_history.loss.length}, (_, i) => i+1),
                    datasets: [
                        {
                            label: 'Loss',
                            data: data.training_history.loss,
                            borderColor: '#ef4444',
                            backgroundColor: 'rgba(239,68,68,0.1)',
                            fill: true,
                        },
                        {
                            label: 'Val Loss',
                            data: data.training_history.val_loss,
                            borderColor: '#06b6d4',
                            backgroundColor: 'rgba(6,182,212,0.1)',
                            fill: true,
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { position: 'top' },
                        title: { display: true, text: 'Training Loss Curve' }
                    },
                    scales: {
                        x: { display: true, title: { display: true, text: 'Epoch' } },
                        y: { display: true, title: { display: true, text: 'Loss' } }
                    }
                }
            });
        }

        // Accuracy Chart
        const accuracyChartEl = document.getElementById('accuracyChart');
        if (accuracyChartEl && data.training_history) {
            const ctx = accuracyChartEl.getContext('2d');
            if (window.accuracyChartInstance) window.accuracyChartInstance.destroy();
            window.accuracyChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array.from({length: data.training_history.mae.length}, (_, i) => i+1),
                    datasets: [
                        {
                            label: 'MAE',
                            data: data.training_history.mae,
                            borderColor: '#f59e0b',
                            backgroundColor: 'rgba(245,158,11,0.1)',
                            fill: true,
                        },
                        {
                            label: 'Val MAE',
                            data: data.training_history.val_mae,
                            borderColor: '#764ba2',
                            backgroundColor: 'rgba(118,75,162,0.1)',
                            fill: true,
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { position: 'top' },
                        title: { display: true, text: 'Training MAE (Accuracy)' }
                    },
                    scales: {
                        x: { display: true, title: { display: true, text: 'Epoch' } },
                        y: { display: true, title: { display: true, text: 'MAE' } }
                    }
                }
            });
        }

        // Show results section
        if (resultsSection) {
            resultsSection.style.display = 'block';
        }
        
        console.log('Results displayed successfully');
    }

    function showError(message) {
        console.error('Showing error:', message);
        if (errorAlert) {
            errorAlert.style.display = 'block';
            const errorMessageEl = document.getElementById('errorMessage');
            if (errorMessageEl) {
                errorMessageEl.textContent = message;
            }
            setTimeout(() => {
                errorAlert.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }, 100);
        }
    }

    function hideError() {
        if (errorAlert) {
            errorAlert.style.display = 'none';
        }
    }

    // Popular stocks quick selection
    const popularStocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM'];
    
    // Add click handlers for stock examples (if you want to add quick-select buttons)
    window.selectStock = function(ticker) {
        document.getElementById('ticker').value = ticker;
    };

    // Smooth scroll for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Tooltips for form inputs (if Bootstrap 5 tooltip is available)
    try {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
            tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        }
    } catch (error) {
        console.log('Tooltips not initialized:', error);
    }
});

// Health check on page load
window.addEventListener('load', async function() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        console.log('API Health:', data);
    } catch (error) {
        console.error('API health check failed:', error);
    }
});
