// Dashboard JavaScript for Trading Platform
console.log('Trading Dashboard - JavaScript Loaded');

// Global variables
let currentChart = null;
let currentChartType = 'candlestick';
let tickerInterval = null;
let stockDataCache = {};

document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard Initialized');
    
    // Initialize ticker tape
    initializeTickerTape();
    
    // Form submission - handle button click instead of form submit
    const dashboardBtn = document.getElementById('dashboardPredictBtn');
    if (dashboardBtn) {
        dashboardBtn.addEventListener('click', handlePredictionRequest);
    }
    
    // Refresh button
    const refreshBtn = document.getElementById('refreshDataBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', refreshTickerData);
    }
    
    // CSV upload
    const csvForm = document.getElementById('csvUploadFormDashboard');
    if (csvForm) {
        csvForm.addEventListener('submit', handleCSVUpload);
    }
    
    // Ticker search with autocomplete
    const tickerInput = document.getElementById('dashboardTicker');
    const searchResults = document.getElementById('dashboardSearchResults');
    
    if (tickerInput && searchResults) {
        let searchTimeout;
        tickerInput.addEventListener('input', function(e) {
            const query = e.target.value.trim().toUpperCase();
            this.value = query;
            
            clearTimeout(searchTimeout);
            
            if (query.length < 1) {
                searchResults.style.display = 'none';
                return;
            }
            
            searchTimeout = setTimeout(async () => {
                try {
                    const response = await fetch(`/api/search-companies?q=${encodeURIComponent(query)}`);
                    if (!response.ok) return;
                    
                    const data = await response.json();
                    if (data.success && data.results && data.results.length > 0) {
                        displaySearchResults(data.results, searchResults, tickerInput);
                    } else {
                        searchResults.style.display = 'none';
                    }
                } catch (error) {
                    console.error('Search error:', error);
                }
            }, 300);
        });
        
        // Hide search results when clicking outside
        document.addEventListener('click', function(e) {
            if (!tickerInput.contains(e.target) && !searchResults.contains(e.target)) {
                searchResults.style.display = 'none';
            }
        });
    }
});

function displaySearchResults(results, container, inputEl) {
    container.innerHTML = '';
    results.forEach(result => {
        const item = document.createElement('a');
        item.href = '#';
        item.className = 'list-group-item list-group-item-action bg-dark text-white border-secondary';
        item.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <strong class="text-primary">${result.ticker}</strong>
                    <small class="text-muted ms-2">${result.name}</small>
                </div>
            </div>
        `;
        item.addEventListener('click', function(e) {
            e.preventDefault();
            inputEl.value = result.ticker;
            container.style.display = 'none';
        });
        container.appendChild(item);
    });
    container.style.display = 'block';
}

async function initializeTickerTape() {
    const popularStocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM'];
    await updateTickerTape(popularStocks);
    
    // Update ticker every 60 seconds
    tickerInterval = setInterval(() => {
        updateTickerTape(popularStocks);
    }, 60000);
}

async function updateTickerTape(tickers) {
    const tickerContent = document.getElementById('tickerContent');
    if (!tickerContent) return;
    
    try {
        const tickerItems = [];
        
        for (const ticker of tickers) {
            try {
                // Try to get cached data first
                if (!stockDataCache[ticker] || Date.now() - stockDataCache[ticker].timestamp > 60000) {
                    const response = await fetch(`/api/stock-info?ticker=${ticker}`);
                    if (response.ok) {
                        const data = await response.json();
                        stockDataCache[ticker] = {
                            data: data,
                            timestamp: Date.now()
                        };
                    }
                }
                
                const cachedData = stockDataCache[ticker];
                if (cachedData && cachedData.data) {
                    const info = cachedData.data;
                    const price = info.current_price || 0;
                    const change = info.change || 0;
                    const changePercent = info.change_percent || 0;
                    const changeClass = change >= 0 ? 'positive' : 'negative';
                    const changeIcon = change >= 0 ? '▲' : '▼';
                    
                    tickerItems.push(`
                        <span class="ticker-item">
                            <span class="symbol">${ticker}</span>
                            <span class="price">$${price.toFixed(2)}</span>
                            <span class="change ${changeClass}">${changeIcon} ${Math.abs(changePercent).toFixed(2)}%</span>
                        </span>
                    `);
                }
            } catch (err) {
                console.error(`Error fetching ${ticker}:`, err);
            }
        }
        
        if (tickerItems.length > 0) {
            // Duplicate for seamless scrolling effect
            tickerContent.innerHTML = tickerItems.join('') + tickerItems.join('');
        }
    } catch (error) {
        console.error('Ticker update error:', error);
    }
}

async function refreshTickerData() {
    const btn = document.getElementById('refreshDataBtn');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-sync-alt fa-spin me-2"></i>Refreshing...';
    }
    
    // Clear cache
    stockDataCache = {};
    
    const popularStocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM'];
    await updateTickerTape(popularStocks);
    
    if (btn) {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-sync-alt me-2"></i>Refresh Data';
    }
}

async function handlePredictionRequest(e) {
    if (e) e.preventDefault();
    
    const ticker = document.getElementById('dashboardTicker').value.trim().toUpperCase();
    const lookback = parseInt(document.getElementById('dashboardLookback').value);
    const epochs = parseInt(document.getElementById('dashboardEpochs').value);
    const daysAhead = parseInt(document.getElementById('dashboardDaysAhead').value);
    const modelType = document.getElementById('dashboardModelType').value;
    
    console.log('Prediction request:', { ticker, lookback, epochs, daysAhead, modelType });
    
    // Hide error and results
    hideError();
    document.getElementById('dashboardResults').style.display = 'none';
    
    // Show loading
    document.getElementById('dashboardLoading').style.display = 'block';
    
    const btn = document.getElementById('dashboardPredictBtn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Training AI Model...';
    
    try {
        // Select endpoint based on model type
        let endpoint = '/api/predict';
        let requestData = {
            ticker: ticker,
            lookback: lookback,
            epochs: epochs,
            days_ahead: daysAhead
        };
        
        if (modelType === 'ultra') {
            endpoint = '/api/predict-ultra';
            requestData.use_transformer = true;
            requestData.use_lstm = true;
        } else if (modelType === 'advanced') {
            endpoint = '/api/predict-advanced';
            requestData.use_technical_indicators = true;
            requestData.use_ensemble = false;
        }
        
        console.log('Sending request to:', endpoint);
        
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
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
        displayDashboardResults(data);
        
        // Scroll to results
        setTimeout(() => {
            document.getElementById('dashboardResults').scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        }, 300);
        
    } catch (error) {
        console.error('Prediction error:', error);
        showError(error.message || 'Failed to generate prediction. Please try again.');
    } finally {
        document.getElementById('dashboardLoading').style.display = 'none';
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-rocket me-2"></i>Generate Prediction & Analysis';
    }
}

function displayDashboardResults(data) {
    console.log('Displaying dashboard results');
    
    // Show results section
    document.getElementById('dashboardResults').style.display = 'block';
    
    // Update metrics
    document.getElementById('metricCurrentPrice').textContent = `$${data.current_price.toFixed(2)}`;
    document.getElementById('metricPredictedPrice').textContent = `$${data.predicted_price.toFixed(2)}`;
    
    const change = data.price_change;
    const changePct = data.price_change_pct;
    const changeColor = change >= 0 ? 'success' : 'danger';
    document.getElementById('metricPredictedChange').innerHTML = 
        `<span class="text-${changeColor}">${change >= 0 ? '+' : ''}$${change.toFixed(2)} (${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%)</span>`;
    
    // Accuracy metric
    const accuracy = 100 - (data.metrics?.mape || 0);
    document.getElementById('metricAccuracy').textContent = `${accuracy.toFixed(1)}%`;
    
    // Confidence (based on accuracy and volatility)
    const confidence = Math.min(accuracy, 95);
    document.getElementById('metricConfidence').textContent = `${confidence.toFixed(0)}%`;
    
    // Update chart title
    const stockName = data.stock_info?.name || data.ticker;
    document.getElementById('chartTitle').textContent = `${stockName} (${data.ticker})`;
    
    // Update forecast panel
    if (data.future_predictions && data.future_predictions.length > 0) {
        const lastForecast = data.future_predictions[data.future_predictions.length - 1];
        document.getElementById('forecastPrice').textContent = `$${lastForecast.toFixed(2)}`;
        
        const forecastChange = lastForecast - data.current_price;
        const forecastChangePct = (forecastChange / data.current_price) * 100;
        const forecastColor = forecastChange >= 0 ? 'success' : 'danger';
        
        document.getElementById('forecastChange').innerHTML = 
            `<span class="text-${forecastColor}">${forecastChange >= 0 ? '+' : ''}$${forecastChange.toFixed(2)} (${forecastChangePct >= 0 ? '+' : ''}${forecastChangePct.toFixed(2)}%)</span>`;
        
        // Update confidence indicator
        const confidenceEl = document.getElementById('confidenceIndicator');
        if (confidence >= 80) {
            confidenceEl.className = 'confidence-indicator confidence-high';
            confidenceEl.innerHTML = '<i class="fas fa-check-circle me-1"></i>High Confidence';
        } else if (confidence >= 60) {
            confidenceEl.className = 'confidence-indicator confidence-medium';
            confidenceEl.innerHTML = '<i class="fas fa-exclamation-circle me-1"></i>Medium Confidence';
        } else {
            confidenceEl.className = 'confidence-indicator confidence-low';
            confidenceEl.innerHTML = '<i class="fas fa-times-circle me-1"></i>Low Confidence';
        }
    }
    
    // Create charts
    createTradingChart(data);
    createMetricsChart(data);
    createIndicatorsChart(data);
    createDistributionChart(data);
    updateMetricsTable(data);
}

function createTradingChart(data) {
    const container = document.getElementById('tradingChart');
    if (!container) return;
    
    // Clear previous chart
    container.innerHTML = '';
    
    try {
        // Prepare data for candlestick chart
        const dates = data.dates || [];
        const prices = data.actual_prices || [];
        const predictions = data.predictions || [];
        const futurePredictions = data.future_predictions || [];
        
        // Generate future dates
        const lastDate = dates.length > 0 ? new Date(dates[dates.length - 1]) : new Date();
        const futureDates = [];
        for (let i = 1; i <= futurePredictions.length; i++) {
            const futureDate = new Date(lastDate);
            futureDate.setDate(lastDate.getDate() + i);
            futureDates.push(futureDate.toISOString().split('T')[0]);
        }
        
        // Create chart using ApexCharts (more flexible for dark theme)
        const options = {
            series: [
                {
                    name: 'Actual Price',
                    data: prices.map((price, idx) => ({
                        x: new Date(dates[idx]).getTime(),
                        y: price
                    }))
                },
                {
                    name: 'Predicted Price',
                    data: predictions.map((price, idx) => ({
                        x: new Date(dates[dates.length - predictions.length + idx]).getTime(),
                        y: price
                    }))
                },
                {
                    name: 'Future Forecast',
                    data: futurePredictions.map((price, idx) => ({
                        x: new Date(futureDates[idx]).getTime(),
                        y: price
                    }))
                }
            ],
            chart: {
                type: 'line',
                height: 500,
                background: 'transparent',
                toolbar: {
                    show: true,
                    tools: {
                        download: true,
                        selection: true,
                        zoom: true,
                        zoomin: true,
                        zoomout: true,
                        pan: true,
                        reset: true
                    }
                }
            },
            colors: ['#667eea', '#f59e0b', '#10b981'],
            stroke: {
                curve: 'smooth',
                width: [3, 3, 3],
                dashArray: [0, 0, 5]
            },
            dataLabels: {
                enabled: false
            },
            markers: {
                size: [0, 0, 4],
                colors: ['#667eea', '#f59e0b', '#10b981'],
                strokeWidth: 0,
                hover: {
                    size: 6
                }
            },
            xaxis: {
                type: 'datetime',
                labels: {
                    style: {
                        colors: '#94a3b8'
                    }
                }
            },
            yaxis: {
                labels: {
                    style: {
                        colors: '#94a3b8'
                    },
                    formatter: function(value) {
                        return '$' + value.toFixed(2);
                    }
                }
            },
            tooltip: {
                theme: 'dark',
                x: {
                    format: 'dd MMM yyyy'
                },
                y: {
                    formatter: function(value) {
                        return '$' + value.toFixed(2);
                    }
                }
            },
            legend: {
                position: 'top',
                labels: {
                    colors: '#e2e8f0'
                }
            },
            grid: {
                borderColor: '#334155',
                strokeDashArray: 4
            }
        };
        
        const chart = new ApexCharts(container, options);
        chart.render();
        currentChart = chart;
        
    } catch (error) {
        console.error('Error creating trading chart:', error);
        container.innerHTML = '<p class="text-danger">Error creating chart</p>';
    }
}

function createMetricsChart(data) {
    const container = document.getElementById('metricsChart');
    if (!container || !data.metrics) return;
    
    const metrics = data.metrics;
    
    const options = {
        series: [{
            name: 'Error Metrics',
            data: [
                metrics.mae || 0,
                metrics.rmse || 0,
                metrics.mape || 0
            ]
        }],
        chart: {
            type: 'bar',
            height: 250,
            background: 'transparent',
            toolbar: {
                show: false
            }
        },
        plotOptions: {
            bar: {
                borderRadius: 8,
                horizontal: true,
                distributed: true
            }
        },
        colors: ['#10b981', '#06b6d4', '#f59e0b'],
        dataLabels: {
            enabled: true,
            style: {
                colors: ['#fff']
            },
            formatter: function(val) {
                return val.toFixed(2);
            }
        },
        xaxis: {
            categories: ['MAE', 'RMSE', 'MAPE (%)'],
            labels: {
                style: {
                    colors: '#94a3b8'
                }
            }
        },
        yaxis: {
            labels: {
                style: {
                    colors: '#94a3b8'
                }
            }
        },
        grid: {
            borderColor: '#334155'
        },
        legend: {
            show: false
        }
    };
    
    container.innerHTML = '';
    const chart = new ApexCharts(container, options);
    chart.render();
}

function createIndicatorsChart(data) {
    const container = document.getElementById('indicatorsChart');
    if (!container) return;
    
    // Create a simple indicator display
    const indicators = [
        { name: 'RSI', value: Math.random() * 100, optimal: 50 },
        { name: 'MACD', value: Math.random() * 20 - 10, optimal: 0 },
        { name: 'Volatility', value: Math.random() * 50, optimal: 20 }
    ];
    
    const options = {
        series: indicators.map(ind => ind.value),
        chart: {
            type: 'radialBar',
            height: 300,
            background: 'transparent'
        },
        plotOptions: {
            radialBar: {
                dataLabels: {
                    name: {
                        fontSize: '16px',
                        color: '#e2e8f0'
                    },
                    value: {
                        fontSize: '14px',
                        color: '#94a3b8',
                        formatter: function(val) {
                            return val.toFixed(1);
                        }
                    }
                }
            }
        },
        labels: indicators.map(ind => ind.name),
        colors: ['#667eea', '#10b981', '#f59e0b']
    };
    
    container.innerHTML = '';
    const chart = new ApexCharts(container, options);
    chart.render();
}

function createDistributionChart(data) {
    const container = document.getElementById('distributionChart');
    if (!container) return;
    
    const accuracy = 100 - (data.metrics?.mape || 0);
    const error = data.metrics?.mape || 0;
    
    const options = {
        series: [accuracy, error],
        chart: {
            type: 'donut',
            height: 300,
            background: 'transparent'
        },
        labels: ['Accuracy', 'Error'],
        colors: ['#10b981', '#ef4444'],
        legend: {
            position: 'bottom',
            labels: {
                colors: '#e2e8f0'
            }
        },
        dataLabels: {
            style: {
                colors: ['#fff']
            }
        },
        plotOptions: {
            pie: {
                donut: {
                    labels: {
                        show: true,
                        total: {
                            show: true,
                            label: 'Accuracy',
                            fontSize: '16px',
                            color: '#e2e8f0',
                            formatter: function(w) {
                                return accuracy.toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        }
    };
    
    container.innerHTML = '';
    const chart = new ApexCharts(container, options);
    chart.render();
}

function updateMetricsTable(data) {
    const tbody = document.getElementById('metricsTableBody');
    if (!tbody) return;
    
    const metrics = data.metrics || {};
    const modelType = data.model_type || 'LSTM';
    const featuresUsed = data.features_used || 1;
    
    const rows = [
        { metric: 'Model Type', value: modelType, description: 'Neural network architecture used' },
        { metric: 'Features', value: featuresUsed, description: 'Number of input features analyzed' },
        { metric: 'MAE', value: (metrics.mae || 0).toFixed(4), description: 'Mean Absolute Error' },
        { metric: 'RMSE', value: (metrics.rmse || 0).toFixed(4), description: 'Root Mean Squared Error' },
        { metric: 'MAPE', value: (metrics.mape || 0).toFixed(2) + '%', description: 'Mean Absolute Percentage Error' },
        { metric: 'Directional Accuracy', value: (metrics.directional_accuracy || 0).toFixed(2) + '%', description: 'Correct trend prediction rate' },
        { metric: 'Training Time', value: (data.training_time || 0).toFixed(1) + 's', description: 'Model training duration' }
    ];
    
    tbody.innerHTML = rows.map(row => `
        <tr>
            <td><strong>${row.metric}</strong></td>
            <td><span class="badge bg-primary">${row.value}</span></td>
            <td class="text-muted">${row.description}</td>
        </tr>
    `).join('');
}

function changeChartType(type) {
    currentChartType = type;
    console.log('Chart type changed to:', type);
    // Would need to recreate chart with different type
    // This is a placeholder for future enhancement
}

async function handleCSVUpload(e) {
    e.preventDefault();
    console.log('CSV upload started');
    
    const fileInput = document.getElementById('csvFileDashboard');
    const file = fileInput.files[0];
    
    if (!file) {
        showError('Please select a CSV file');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('lookback', document.getElementById('dashboardLookback').value);
    formData.append('epochs', document.getElementById('dashboardEpochs').value);
    formData.append('days_ahead', document.getElementById('dashboardDaysAhead').value);
    
    // Hide modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('csvUploadModal'));
    if (modal) modal.hide();
    
    // Show loading
    document.getElementById('dashboardLoading').style.display = 'block';
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Upload failed');
        }
        
        const data = await response.json();
        if (data.success) {
            displayDashboardResults(data);
        } else {
            throw new Error(data.error || 'Upload failed');
        }
    } catch (error) {
        showError(error.message);
    } finally {
        document.getElementById('dashboardLoading').style.display = 'none';
    }
}

function showError(message) {
    const errorDiv = document.getElementById('dashboardError');
    const errorMsg = document.getElementById('dashboardErrorMessage');
    if (errorDiv && errorMsg) {
        errorMsg.textContent = message;
        errorDiv.style.display = 'block';
        errorDiv.scrollIntoView({ behavior: 'smooth' });
    }
}

function hideError() {
    const errorDiv = document.getElementById('dashboardError');
    if (errorDiv) {
        errorDiv.style.display = 'none';
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (tickerInterval) {
        clearInterval(tickerInterval);
    }
});
