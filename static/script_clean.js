// Clean, working JavaScript for Stock Price Prediction
console.log('Stock Predictor - Clean JavaScript Loaded');

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM Content Loaded');
    
    // Get elements safely
    const csvUploadForm = document.getElementById('csvUploadForm');
    const csvPredictBtn = document.getElementById('csvPredictBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultsSection = document.getElementById('resultsSection');
    const errorAlert = document.getElementById('errorAlert');
    const errorMessage = document.getElementById('errorMessage');

    // CSV upload form handler
    if (csvUploadForm) {
        csvUploadForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            await handleCSVPrediction();
        });
    }

    async function handleCSVPrediction() {
        const fileInput = document.getElementById('csvFile');
        if (!fileInput || !fileInput.files || !fileInput.files[0]) {
            showError('Please select a CSV file');
            return;
        }

        const lookback = parseInt(document.getElementById('lookback')?.value) || 60;
        const epochs = parseInt(document.getElementById('epochs')?.value) || 50;
        const daysAhead = parseInt(document.getElementById('daysAhead')?.value) || 5;

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('lookback', lookback);
        formData.append('epochs', epochs);
        formData.append('days_ahead', daysAhead);

        showLoading(true);
        hideError();
        hideResults();

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                let errorData;
                try {
                    errorData = await response.json();
                } catch (e) {
                    errorData = { error: `HTTP ${response.status}` };
                }
                throw new Error(errorData.error || 'Server error');
            }

            const data = await response.json();
            
            if (!data || !data.success) {
                throw new Error(data?.error || 'Prediction failed');
            }

            displayResults(data);
            showResults();

        } catch (error) {
            console.error('Error:', error);
            showError(error.message || 'Prediction failed');
        } finally {
            showLoading(false);
        }
    }

    function displayResults(data) {
        if (!data || typeof data !== 'object') {
            showError('Invalid data received');
            return;
        }

        try {
            // Display metrics
            const metrics = data.metrics || {};
            
            const mapeEl = document.getElementById('mape');
            if (mapeEl) {
                const mape = parseFloat(metrics.MAPE || metrics.mape || 0);
                mapeEl.textContent = `${mape.toFixed(2)}%`;
            }
            
            const rmseEl = document.getElementById('rmse');
            if (rmseEl) {
                const rmse = parseFloat(metrics.RMSE || metrics.rmse || 0);
                rmseEl.textContent = rmse.toFixed(4);
            }
            
            const maeEl = document.getElementById('mae');
            if (maeEl) {
                const mae = parseFloat(metrics.MAE || metrics.mae || 0);
                maeEl.textContent = mae.toFixed(4);
            }
            
            const r2El = document.getElementById('r2Score');
            if (r2El) {
                const r2 = parseFloat(metrics.R2 || metrics.r2 || 0);
                r2El.textContent = r2.toFixed(3);
            }

            // Model info
            const modelTypeEl = document.getElementById('modelType');
            if (modelTypeEl) {
                const modelType = data.model_type || 'Advanced ML';
                const featuresUsed = parseInt(data.features_used || 0);
                const dataPoints = parseInt(data.data_points || 0);
                modelTypeEl.textContent = `${modelType} (${featuresUsed} features, ${dataPoints} points)`;
            }

            // Display predictions
            const predictions = Array.isArray(data.predictions) ? data.predictions : [];
            const predictionsContainer = document.getElementById('predictionsContainer');
            if (predictionsContainer && predictions.length > 0) {
                let html = '<h5>Future Predictions:</h5><div class="row">';
                
                predictions.forEach((pred, index) => {
                    const price = parseFloat(pred || 0);
                    html += `
                        <div class="col-md-6 mb-2">
                            <div class="card">
                                <div class="card-body text-center">
                                    <h6>Day ${index + 1}</h6>
                                    <h4 class="text-primary">$${price.toFixed(2)}</h4>
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                html += '</div>';
                predictionsContainer.innerHTML = html;
            }

            // Create chart
            createChart(data);
            
        } catch (error) {
            console.error('Display error:', error);
            showError('Error displaying results');
        }
    }

    function createChart(data) {
        const chartCanvas = document.getElementById('predictionChart');
        if (!chartCanvas || typeof Chart === 'undefined') return;

        try {
            const ctx = chartCanvas.getContext('2d');
            
            if (window.stockChart) {
                window.stockChart.destroy();
            }

            const historicalData = Array.isArray(data.data) ? data.data : 
                                 Array.isArray(data.historical_data) ? data.historical_data : [];
            const predictions = Array.isArray(data.predictions) ? data.predictions : [];
            
            const historicalLabels = historicalData.map((_, i) => `Point ${i + 1}`);
            const predictionLabels = predictions.map((_, i) => `Future ${i + 1}`);
            
            window.stockChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [...historicalLabels, ...predictionLabels],
                    datasets: [
                        {
                            label: 'Historical Data',
                            data: [...historicalData, ...new Array(predictions.length).fill(null)],
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16,185,129,0.1)',
                            pointRadius: 2,
                            fill: false
                        },
                        {
                            label: 'Predictions',
                            data: [...new Array(historicalData.length).fill(null), ...predictions],
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59,130,246,0.1)',
                            pointRadius: 3,
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Stock Price Prediction'
                        },
                        legend: {
                            position: 'top'
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Time Points'
                            }
                        },
                        y: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Price ($)'
                            }
                        }
                    }
                }
            });
            
        } catch (error) {
            console.error('Chart error:', error);
        }
    }

    function showLoading(show) {
        if (loadingSpinner) {
            loadingSpinner.style.display = show ? 'block' : 'none';
        }
        
        if (csvPredictBtn) {
            csvPredictBtn.disabled = show;
            csvPredictBtn.innerHTML = show ? 
                '<i class="fas fa-spinner fa-spin me-2"></i>Processing...' :
                '<i class="fas fa-upload me-2"></i>Upload & Predict';
        }
    }

    function showError(message) {
        console.error('Error:', message);
        if (errorAlert && errorMessage) {
            errorMessage.textContent = message || 'An error occurred';
            errorAlert.style.display = 'block';
            setTimeout(hideError, 8000);
        }
    }

    function hideError() {
        if (errorAlert) {
            errorAlert.style.display = 'none';
        }
    }

    function showResults() {
        if (resultsSection) {
            resultsSection.style.display = 'block';
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    function hideResults() {
        if (resultsSection) {
            resultsSection.style.display = 'none';
        }
    }

    console.log('Stock Predictor JavaScript initialized successfully');
});