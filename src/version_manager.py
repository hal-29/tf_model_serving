import os
from .config import CURRENT_VERSION_FILE, MODEL_CONFIG_FILE, MODEL_NAME, MODEL_BASE_PATH

class VersionManager:
    def __init__(self):
        self.ensure_directories_exist()
        
    def ensure_directories_exist(self):
        os.makedirs(os.path.join(MODEL_BASE_PATH, MODEL_NAME), exist_ok=True)
        
    def get_current_version(self):
        try:
            if os.path.exists(CURRENT_VERSION_FILE):
                with open(CURRENT_VERSION_FILE, 'r') as f:
                    return int(f.read().strip())
        except (ValueError, IOError):
            pass
            
        versions = self.get_available_versions()
        if versions:
            latest = max(versions)
            self.set_current_version(latest)
            return latest
        return None
        
    def set_current_version(self, version):
        try:
            with open(CURRENT_VERSION_FILE, 'w') as f:
                f.write(str(version))
            
            self.update_model_config()
            return True
        except IOError:
            return False
            
    def rollback_version(self, version):
        available_versions = self.get_available_versions()
        if version in available_versions:
            return self.set_current_version(version)
        return False
        
    def get_available_versions(self):
        model_path = os.path.join(MODEL_BASE_PATH, MODEL_NAME)
        if not os.path.exists(model_path):
            return []
            
        versions = []
        for item in os.listdir(model_path):
            if item.isdigit() and os.path.isdir(os.path.join(model_path, item)):
                model_dir = os.path.join(model_path, item)
                if os.path.exists(os.path.join(model_dir, "saved_model.pb")):
                    versions.append(int(item))
                    
        return sorted(versions)
        
    def update_model_config(self):
        """
        Generates the models.config file to instruct TensorFlow Serving to load
        ALL valid model versions found in the directory.
        """
        if not self.get_available_versions():
            print("No versions found, skipping config update.")
            return

        config_content = f"""model_config_list {{
                           config {{
                              name: "{MODEL_NAME}",
                              base_path: "/models/{MODEL_NAME}",
                              model_platform: "tensorflow",
                              model_version_policy {{
                                 all {{}}
                              }}
                           }}
                           }}"""
        
        try:
            with open(MODEL_CONFIG_FILE, 'w') as f:
                f.write(config_content)
        except IOError as e:
            print(f"Failed to update model config: {e}")