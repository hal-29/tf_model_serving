set -e

echo "=== Starting ML Deployment System ==="

echo "Creating required directories..."
mkdir -p models/mnist
mkdir -p monitoring

if [ ! -d "models/mnist/1" ]; then
    echo "No trained model found. Training initial model..."
    python -m src.pipeline
else
    echo "Existing model found at models/mnist/1/"
fi

echo "Initializing model configuration..."
python -m scripts.init_model_config

# Start Docker containers
echo "Starting Docker containers..."
docker-compose up -d

echo "=== System started successfully ==="
echo "Services available at:"
echo "- API: http://localhost:8080"
echo "- TF Serving: http://localhost:8501"
echo "- Prometheus: http://localhost:9090"