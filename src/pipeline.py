import os
import shutil
from src import config
from src import train

def get_latest_version():
   model_path = os.path.join(config.MODEL_BASE_PATH, config.MODEL_NAME)
   if not os.path.exists(model_path):
      return 0
   
   versions = [
      int(d) for d in os.listdir(model_path) 
      if os.path.isdir(os.path.join(model_path, d)) and d.isdigit()
   ]
   
   return max(versions) if versions else 0

def run_pipeline():
   """
   Executes the full retrain-evaluate-deploy pipeline.
   """
   print("--- Starting Model Update Pipeline ---")

   latest_version = get_latest_version()
   print(f"Current model version: {latest_version}")

   candidate_model, accuracy = train.train_and_evaluate()

   print(f"\nChecking accuracy against threshold ({config.ACCURACY_THRESHOLD})...")
   
   if accuracy >= config.ACCURACY_THRESHOLD:
      new_version = latest_version + 1
      print(f"PASSED: Accuracy ({accuracy:.4f}) is above threshold.")
      print(f"Promoting and saving new model as version {new_version}.")
      
      export_path = os.path.join(config.MODEL_BASE_PATH, config.MODEL_NAME, str(new_version))
      os.makedirs(os.path.dirname(export_path), exist_ok=True)
      candidate_model.save(export_path)
      
      print(f"Model successfully saved!")
   else:
      print(f"FAILED: Accuracy ({accuracy:.4f}) is below threshold.")
      print(f"Keeping previous version ({latest_version}) and discarding candidate model.")

   print("\n--- Pipeline Run Finished ---")


if __name__ == "__main__":
   if not os.path.exists(config.MODEL_BASE_PATH):
      os.makedirs(config.MODEL_BASE_PATH)
      
   run_pipeline()
