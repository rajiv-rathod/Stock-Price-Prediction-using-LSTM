# 🚀 Stock Price Prediction Web Application - Deployment Guide

## 📋 Table of Contents
1. [Quick Start](#quick-start)
2. [Local Development](#local-development)
3. [Production Deployment](#production-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [API Documentation](#api-documentation)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Quick Start

### Option 1: Simple Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Access at: **http://localhost:5000**

### Option 2: Using the Start Script
```bash
# Make script executable (first time only)
chmod +x start_web_app.sh

# Run the application
./start_web_app.sh
```

---

## 💻 Local Development

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)
- Internet connection (for fetching stock data)

### Setup Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file for environment variables:

```env
FLASK_ENV=development
FLASK_DEBUG=1
MODEL_CACHE_DIR=./models_cache
SECRET_KEY=your-secret-key-here
```

### Running the Application

```bash
python app.py
```

The application will be available at:
- **Web Interface**: http://localhost:5000
- **API Health**: http://localhost:5000/api/health

---

## 🏭 Production Deployment

### Using Gunicorn (Recommended)

1. **Install Gunicorn**
   ```bash
   pip install gunicorn
   ```

2. **Run with Gunicorn**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app
   ```

   Options explained:
   - `-w 4`: 4 worker processes
   - `-b 0.0.0.0:5000`: Bind to all interfaces on port 5000
   - `--timeout 120`: 120 seconds timeout (for model training)

### Using Systemd (Linux)

Create `/etc/systemd/system/stock-predictor.service`:

```ini
[Unit]
Description=Stock Price Prediction Web Service
After=network.target

[Service]
User=www-data
WorkingDirectory=/var/www/stock-predictor
Environment="PATH=/var/www/stock-predictor/venv/bin"
ExecStart=/var/www/stock-predictor/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable stock-predictor
sudo systemctl start stock-predictor
sudo systemctl status stock-predictor
```

### Nginx Reverse Proxy

Create `/etc/nginx/sites-available/stock-predictor`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 120s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
    }

    location /static {
        alias /var/www/stock-predictor/static;
        expires 30d;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/stock-predictor /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t stock-predictor:latest .

# Run the container
docker run -d \
  --name stock-predictor \
  -p 5000:5000 \
  --restart unless-stopped \
  stock-predictor:latest
```

### Using Docker Compose

```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

### Docker Commands

```bash
# View running containers
docker ps

# View logs
docker logs -f stock-predictor

# Execute commands in container
docker exec -it stock-predictor bash

# Stop container
docker stop stock-predictor

# Start container
docker start stock-predictor

# Remove container
docker rm stock-predictor
```

---

## ☁️ Cloud Deployment

### Heroku

1. **Install Heroku CLI**
   ```bash
   curl https://cli-assets.heroku.com/install.sh | sh
   ```

2. **Create Procfile**
   ```
   web: gunicorn app:app
   ```

3. **Deploy**
   ```bash
   heroku login
   heroku create stock-predictor-app
   git push heroku main
   heroku open
   ```

### AWS EC2

1. **Launch EC2 Instance**
   - Ubuntu 22.04 LTS
   - t2.medium or larger
   - Security Group: Allow ports 22 (SSH), 80 (HTTP), 443 (HTTPS)

2. **Connect and Setup**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install Python and dependencies
   sudo apt install python3-pip python3-venv nginx -y
   
   # Clone repository
   git clone https://github.com/rajiv-rathod/Stock-Price-Prediction-using-LSTM.git
   cd Stock-Price-Prediction-using-LSTM
   
   # Setup virtual environment
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install gunicorn
   
   # Configure Nginx (see above)
   # Setup systemd service (see above)
   ```

### Google Cloud Run

1. **Create cloudbuild.yaml**
   ```yaml
   steps:
     - name: 'gcr.io/cloud-builders/docker'
       args: ['build', '-t', 'gcr.io/$PROJECT_ID/stock-predictor', '.']
     - name: 'gcr.io/cloud-builders/docker'
       args: ['push', 'gcr.io/$PROJECT_ID/stock-predictor']
   ```

2. **Deploy**
   ```bash
   gcloud builds submit --config cloudbuild.yaml
   gcloud run deploy stock-predictor \
     --image gcr.io/$PROJECT_ID/stock-predictor \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

### Azure App Service

```bash
# Login to Azure
az login

# Create resource group
az group create --name StockPredictorRG --location eastus

# Create App Service plan
az appservice plan create \
  --name StockPredictorPlan \
  --resource-group StockPredictorRG \
  --sku B1 \
  --is-linux

# Create web app
az webapp create \
  --resource-group StockPredictorRG \
  --plan StockPredictorPlan \
  --name stock-predictor-app \
  --runtime "PYTHON:3.11"

# Deploy code
az webapp up --name stock-predictor-app
```

### DigitalOcean

1. **Create Droplet**
   - Ubuntu 22.04
   - 2GB RAM minimum

2. **Setup Application**
   ```bash
   # Connect to droplet
   ssh root@your-droplet-ip
   
   # Follow AWS EC2 setup steps
   ```

---

## 📚 API Documentation

### Endpoints

#### 1. Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-04T13:49:35.123456",
  "service": "Stock Price Prediction API"
}
```

#### 2. Get Stock Information
```http
GET /api/stock-info/<ticker>
```

**Example:**
```bash
curl http://localhost:5000/api/stock-info/AAPL
```

**Response:**
```json
{
  "success": true,
  "data": {
    "name": "Apple Inc.",
    "symbol": "AAPL",
    "latest_price": 175.43,
    "prev_price": 174.21,
    "change": 1.22,
    "change_pct": 0.70
  }
}
```

#### 3. Generate Prediction
```http
POST /api/predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "ticker": "AAPL",
  "lookback": 60,
  "epochs": 50,
  "days_ahead": 5
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "lookback": 60,
    "epochs": 50,
    "days_ahead": 5
  }'
```

**Response:**
```json
{
  "success": true,
  "stock_info": {
    "name": "Apple Inc.",
    "symbol": "AAPL"
  },
  "current_price": 175.43,
  "predicted_price": 178.21,
  "price_change": 2.78,
  "price_change_pct": 1.58,
  "metrics": {
    "mse": 52.42,
    "rmse": 7.24,
    "mae": 5.50,
    "mape": 2.60,
    "directional_accuracy": 65.5
  },
  "future_predictions": {
    "dates": ["2025-10-05", "2025-10-06", "2025-10-07"],
    "prices": [178.21, 179.45, 180.12]
  },
  "plot": "base64_encoded_image_data",
  "timestamp": "2025-10-04T13:50:00.000000"
}
```

### Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| ticker | string | - | AAPL | Stock ticker symbol |
| lookback | integer | 10-200 | 60 | Historical days to analyze |
| epochs | integer | 10-100 | 50 | Training iterations |
| days_ahead | integer | 1-30 | 5 | Days to forecast |

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Find process using port 5000
lsof -i :5000

# Kill the process
kill -9 <PID>
```

#### 2. Memory Issues
- Reduce `epochs` parameter
- Reduce `lookback` parameter
- Use swap space:
  ```bash
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  ```

#### 3. TensorFlow CPU Warnings
These are informational and can be ignored. The app runs on CPU without issues.

#### 4. Yahoo Finance Connection Issues
- Check internet connection
- Verify ticker symbol is valid
- Try again later (API rate limits)

#### 5. Model Training Timeout
- Increase timeout in Gunicorn: `--timeout 180`
- Reduce epochs or lookback period
- Use a more powerful server

### Logs

**View Application Logs:**
```bash
# Docker
docker logs -f stock-predictor

# Systemd
sudo journalctl -u stock-predictor -f

# Direct
tail -f app.log
```

### Performance Optimization

1. **Enable Response Caching**
2. **Use a CDN for static files**
3. **Implement rate limiting**
4. **Pre-train popular models**
5. **Use GPU if available**

---

## 📊 Monitoring

### Basic Monitoring

```bash
# CPU and Memory usage
htop

# Network connections
netstat -tupln | grep :5000

# Application health
watch -n 5 'curl -s http://localhost:5000/api/health'
```

### Advanced Monitoring

Consider using:
- **Prometheus + Grafana**: Metrics and dashboards
- **ELK Stack**: Log aggregation
- **Sentry**: Error tracking
- **New Relic/DataDog**: APM

---

## 🔒 Security Considerations

1. **Use HTTPS in production**
2. **Set strong SECRET_KEY**
3. **Implement rate limiting**
4. **Validate all inputs**
5. **Keep dependencies updated**
6. **Use environment variables for secrets**
7. **Enable firewall rules**
8. **Regular security audits**

---

## 📈 Scaling

### Horizontal Scaling
- Deploy multiple instances behind a load balancer
- Use Redis for session management
- Implement caching layer

### Vertical Scaling
- Increase CPU/RAM resources
- Use GPU for faster training
- Optimize model architecture

---

## 📞 Support

For issues, questions, or contributions:
- **GitHub Issues**: https://github.com/rajiv-rathod/Stock-Price-Prediction-using-LSTM/issues
- **Documentation**: See WEB_APP_GUIDE.md
- **Email**: [Your Email]

---

**Made with ❤️ by Rajiv Rathod**
