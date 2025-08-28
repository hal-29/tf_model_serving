import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.version_manager import VersionManager

def initialize_model_config():
    """Initialize the model configuration file"""
    print("Initializing model configuration...")
    
    version_manager = VersionManager()
    
    os.makedirs(os.path.join("models", "mnist"), exist_ok=True)
    
    versions = version_manager.get_available_versions()
    
    if not versions:
        print("No model versions found. Please run the training pipeline first.")
        print("Run: python -m src.pipeline")
        return False
    
    latest_version = max(versions)
    success = version_manager.set_current_version(latest_version)
    
    if success:
        print(f"Model configuration initialized with version {latest_version}")
        return True
    else:
        print("Failed to initialize model configuration")
        return False

if __name__ == "__main__":
    success = initialize_model_config()
    sys.exit(0 if success else 1)
