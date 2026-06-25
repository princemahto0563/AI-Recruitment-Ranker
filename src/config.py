import yaml
import os

class RankerConfig:
    def __init__(self, config_path="config/config.yaml"):
        if not os.path.exists(config_path):
            # Try to resolve relative path if run from a subdirectory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, config_path)
            
        with open(config_path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)
            
        # Extract sections
        self.model_name = self._data["model"]["name"]
        self.model_local_path = self._data["model"].get("local_path", None)
        
        self.weights = self._data["scoring_weights"]
        self.location = self._data["location"]
        self.behavior = self._data["behavior_multipliers"]
        self.paths = self._data["paths"]

    def get(self, key, default=None):
        return self._data.get(key, default)
