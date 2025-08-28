from prometheus_client import Counter, Gauge, Histogram
import time

# Prediction metrics
PREDICTION_COUNTER = Counter(
    'model_predictions_total', 
    'Total predictions', 
    ['model_name', 'version', 'status']
)

PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_name', 'version']
)

ACCURACY_GAUGE = Gauge(
    'model_accuracy',
    'Model accuracy',
    ['model_name', 'version']
)

MODEL_VERSION_GAUGE = Gauge(
    'model_version',
    'Current model version',
    ['model_name']
)

# Training metrics
TRAINING_COUNTER = Counter(
    'model_training_total',
    'Total training runs',
    ['model_name', 'status']
)

TRAINING_DURATION = Histogram(
    'model_training_duration_seconds',
    'Training duration in seconds',
    ['model_name']
)

def record_prediction_metrics(version, latency_ms, success):
    """Record prediction metrics"""
    status = "success" if success else "failure"
    PREDICTION_COUNTER.labels(
        model_name="mnist", 
        version=version, 
        status=status
    ).inc()
    
    if success:
        PREDICTION_LATENCY.labels(
            model_name="mnist", 
            version=version
        ).observe(latency_ms / 1000)  # Convert to seconds

def record_training_metrics(duration, success, accuracy=None, version=None):
    """Record training metrics"""
    status = "success" if success else "failure"
    TRAINING_COUNTER.labels(model_name="mnist", status=status).inc()
    
    if success:
        TRAINING_DURATION.labels(model_name="mnist").observe(duration)
        if accuracy is not None and version is not None:
            ACCURACY_GAUGE.labels(
                model_name="mnist", 
                version=version
            ).set(accuracy)
            MODEL_VERSION_GAUGE.labels(model_name="mnist").set(version)