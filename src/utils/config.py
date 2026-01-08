import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration loader for the NER model."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def __getattr__(self, name: str) -> Any:
        """Allow dot notation access to config values."""
        if name in self.config:
            value = self.config[name]
            if isinstance(value, dict):
                return type('Config', (), value)()
            return value
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update(self, key: str, value: Any) -> None:
        """Update configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: str = None) -> None:
        """Save configuration to YAML file."""
        save_path = Path(path) if path else self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
