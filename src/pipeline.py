import os
import shutil
import time
import logging
from src import config
from src import train
from src.version_manager import VersionManager
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

version_manager = VersionManager()

def get_latest_version():
    versions = version_manager.get_available_versions()
    return max(versions) if versions else 0

def run_pipeline():
    start_time = time.time()
    logger.info("--- Starting Model Update Pipeline ---")
    
    try:
        latest_version = get_latest_version()
        logger.info(f"Current model version: {latest_version}")

        candidate_model, accuracy = train.train_and_evaluate()
        
        logger.info(f"Candidate model accuracy: {accuracy:.4f}")
        logger.info(f"Checking accuracy against threshold ({config.ACCURACY_THRESHOLD})...")

        if accuracy >= config.ACCURACY_THRESHOLD:
            new_version = latest_version + 1
            logger.info(f"PASSED: Accuracy ({accuracy:.4f}) is above threshold.")
            logger.info(f"Promoting and saving new model as version {new_version}.")

            export_path = os.path.join(config.MODEL_BASE_PATH, config.MODEL_NAME, str(new_version))
            if os.path.exists(export_path):
                shutil.rmtree(export_path)
            os.makedirs(export_path, exist_ok=True)
            
            candidate_model.save(export_path, save_format='tf')
            
            version_manager.set_current_version(new_version)
            
            logger.info(f"Model successfully saved to {export_path}!")
            
        else:
            logger.info(f"FAILED: Accuracy ({accuracy:.4f}) is below threshold.")
            logger.info(f"Keeping previous version ({latest_version}) and discarding candidate model.")
            
        logger.info("--- Pipeline Run Finished ---")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")

if __name__ == "__main__":
    model_dir = os.path.join(config.MODEL_BASE_PATH, config.MODEL_NAME)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    run_pipeline()
