import torch
import json
import os
from typing import Dict, Any, Optional

class Config:
    class data:
        data_root = 'data/flickr8k/Flicker8k_Dataset'
        captions_file = 'data/flickr8k/captions.txt'

    class train:
        batch_size = 32
        num_epochs = 20
        learning_rate = 0.0003
        freeze_encoder_epochs = 5  # Generalized from freeze_cnn_epochs
        checkpoint_dir = 'checkpoints'
        resume = True
        checkpoint_path = None
        save_step = 2 
        use_mixed_precision = False
        early_stopping_patience = 3

    class evaluate:
        beam_search = False
        beam_size = 3
        visualize = True
        visualize_dir = 'visualizations'
        num_examples = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def save_to_json(file_path: str = "logs/config.json") -> None:
        """Save all config parameters to a JSON file."""
        config_dict: Dict[str, Any] = {}

        for key in dir(Config):
            if not key.startswith("__") and key not in ["save_to_json", "load_from_json"]:
                attr = getattr(Config, key)
                if isinstance(attr, type):  # Nested class
                    config_dict[key] = {k: v for k, v in vars(attr).items() if not k.startswith("__")}
                else:
                    config_dict[key] = attr
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save to JSON
        with open(file_path, "w") as json_file:
            json.dump(config_dict, json_file, indent=4)

        print(f"Config saved to {file_path}")
    
    @staticmethod
    def load_from_json(file_path: str = "config.json") -> None:
        """Load config parameters from a JSON file and update the Config class."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file '{file_path}' not found.")

        with open(file_path, "r") as json_file:
            config_dict = json.load(json_file)

        for key, value in config_dict.items():
            if hasattr(Config, key):  # Check if key exists in Config
                attr = getattr(Config, key)
                if isinstance(attr, type):  # If it's a nested class, update its attributes
                    for sub_key, sub_value in value.items():
                        setattr(attr, sub_key, sub_value)
                else:
                    setattr(Config, key, value)

        print(f"Config loaded from {file_path}")

