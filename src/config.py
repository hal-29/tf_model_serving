import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "mnist")
MODEL_BASE_PATH = os.getenv("MODEL_BASE_PATH", "/app/models")
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.97"))
API_KEY = os.getenv("API_KEY", "default-insecure-key")
TF_SERVING_URL = os.getenv("TF_SERVING_URL", "http://tf-serving:8501/v1/models/mnist")
CURRENT_VERSION_FILE = os.path.join(MODEL_BASE_PATH, MODEL_NAME, "current_version.txt")
MODEL_CONFIG_FILE = os.path.join(MODEL_BASE_PATH, MODEL_NAME, "models.config")
