import torch
import json
import os
from typing import Dict, Any, Optional

class Config:
    class data:
        # Path to the root directory containing image files
        data_root = 'data/flickr8k/Flicker8k_Dataset' # Common structure: dataset/Images
        # Path to the captions file (CSV format with 'image' and 'caption' columns)
        captions_file = 'data/flickr8k/captions.txt'
        # Minimum word frequency to be included in the vocabulary
        vocab_freq_threshold = 1

    class train:
        # Batch size for training and validation
        batch_size = 32
        # Total number of training epochs
        num_epochs = 20
        # Initial learning rate for the Adam optimizer
        learning_rate = 0.0003
        # Number of initial epochs to keep the encoder frozen (applies to CNN base in CNNtoRNN)
        freeze_encoder_epochs = 5
        # Directory to save model checkpoints
        checkpoint_dir = 'checkpoints'
        # Whether to attempt resuming training from the latest checkpoint if available
        resume = True
        # Specific checkpoint path to resume from (overrides searching for latest)
        checkpoint_path = None
        # Frequency (in epochs) to save a checkpoint
        save_step = 2
        # Whether to use Automatic Mixed Precision (AMP) for training (requires CUDA)
        use_mixed_precision = True # Changed default to True, common practice
        # Number of epochs with no improvement in validation loss before stopping training
        early_stopping_patience = 5 # Increased patience
        # Fraction of data to use for validation split
        validation_split = 0.1
        # Number of workers for DataLoader
        num_workers = 2 # Adjusted based on typical resources

    class evaluate:
        # Whether to use beam search for caption generation during evaluation
        beam_search = True # Changed default to True for better quality captions
        # Beam size if beam_search is True
        beam_size = 5 # Increased beam size
        # Whether to generate and save visualizations of example predictions
        visualize = True
        # Directory to save evaluation results (BLEU scores, visualizations)
        visualize_dir = 'eval_results' # Renamed for clarity
        # Number of examples to visualize if visualize is True
        num_examples = 10
        # Batch size for evaluation (often 1, especially for beam search)
        batch_size = 16 # Can be > 1 for greedy eval if memory allows

    # Automatically set device to CUDA if available, otherwise CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def save_to_json(file_path: str) -> None:
        """Save all config parameters to a JSON file."""
        config_dict: Dict[str, Any] = {}

        # Iterate through Config attributes
        for key in dir(Config):
            # Skip special methods and the save/load methods themselves
            if not key.startswith("__") and key not in ["save_to_json", "load_from_json"]:
                attr = getattr(Config, key)
                # Handle nested classes (like data, train, evaluate)
                if isinstance(attr, type):
                    config_dict[key] = {k: v for k, v in vars(attr).items() if not k.startswith("__")}
                # Handle direct attributes (like device)
                else:
                    config_dict[key] = attr

        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Save to JSON file with indentation
            with open(file_path, "w") as json_file:
                json.dump(config_dict, json_file, indent=4)
            print(f"Config saved to {file_path}")
        except Exception as e:
            print(f"Error saving config to {file_path}: {e}")

    @staticmethod
    def load_from_json(file_path: str) -> None:
        """Load config parameters from a JSON file and update the Config class."""
        if not os.path.exists(file_path):
            print(f"Warning: Config file '{file_path}' not found. Using default configuration.")
            return # Don't raise error, just use defaults

        try:
            with open(file_path, "r") as json_file:
                config_dict = json.load(json_file)

            # Update Config attributes
            for key, value in config_dict.items():
                if hasattr(Config, key):
                    attr = getattr(Config, key)
                    # Update nested class attributes
                    if isinstance(attr, type):
                        for sub_key, sub_value in value.items():
                            if hasattr(attr, sub_key):
                                setattr(attr, sub_key, sub_value)
                            else:
                                print(f"Warning: Key '{sub_key}' not found in nested Config class '{key}'.")
                    # Update direct attributes
                    else:
                         # Special handling for device? No, allow override.
                         setattr(Config, key, value)
                else:
                     print(f"Warning: Key '{key}' from config file not found in Config class.")

            # Re-evaluate device after loading, in case 'device' was specified in the file
            Config.device = config_dict.get('device', Config.device)
            Config.device = 'cuda' if Config.device == 'cuda' and torch.cuda.is_available() else 'cpu'
            print(f"Config loaded from {file_path}. Current device: {Config.device}")

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from config file {file_path}: {e}")
        except Exception as e:
            print(f"Error loading config from {file_path}: {e}")