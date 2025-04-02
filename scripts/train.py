import os
import sys
import torch
import argparse
from typing import Dict, Any, Optional, Type, List, Tuple
import importlib # Keep importlib if dynamically loading models is ever needed

# Add project root to path to allow imports like 'utils', 'model'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.trainer import Trainer
from utils.config import Config
from model.base_model import BaseModel
from model.cnntornn import CNNtoRNN
from model.transformer import VisionEncoderDecoder

def get_model_config(model_name: str) -> Tuple[Type[BaseModel], Dict[str, Any]]:
    """
    Get the model class and default keyword arguments based on the model name.
    These defaults are used if a specific config file doesn't override them.

    Args:
        model_name: Name of the model (e.g., 'transformer', 'cnntornn').

    Returns:
        Tuple of (model_class, model_kwargs).
    """
    model_name_lower = model_name.lower()
    if model_name_lower == "transformer":
        # Default parameters for Transformer (can be overridden by config file)
        return VisionEncoderDecoder, {
            'image_size': 224,
            'channels_in': 3,
            'patch_size': 16,
            'hidden_size': 256, # Increased default hidden size
            'num_layers': (1, 1),  # (encoder_layers, decoder_layers) - Increased default layers
            'num_heads': 4,
            # vocab_size will be added by Trainer
        }
    elif model_name_lower == "cnntornn":
        # Default parameters for CNNtoRNN (can be overridden by config file)
        return CNNtoRNN, {
            'embed_size': 256,
            'hidden_size': 512, # Increased default hidden size
            'num_layers': 1, # Changed default to 1 layer LSTM
            'dropout_rate': 0.5,
            'trainCNN': False,  # Initially False, Trainer handles unfreezing
            # vocab_size will be added by Trainer
        }
    else:
        raise ValueError(f"Unknown model name: {model_name_lower}. Choose 'transformer' or 'cnntornn'.")


def main():
    """
    Main function to train image captioning models (CNNtoRNN and/or Transformer).
    """
    parser = argparse.ArgumentParser(description="Train Image Captioning Models")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON configuration file. Overrides default settings."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        default=["transformer", "cnntornn"], # Default to training only CNNtoRNN
        help="Specify which model(s) to train: 'transformer', 'cnntornn'."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Attempt to resume training from the latest checkpoint for the specified model(s)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint file to resume from (overrides --resume default behavior)."
    )
    args = parser.parse_args()

    # --- Configuration Loading ---
    # Load base config from file if provided
    if args.config and os.path.exists(args.config):
        print(f"Loading base configuration from: {args.config}")
        Config.load_from_json(args.config)
    else:
        if args.config:
             print(f"Warning: Specified config file '{args.config}' not found. Using defaults.")
        else:
             print("No config file specified. Using default configuration.")

    # Update config based on command-line args (e.g., resume)
    if args.resume or args.checkpoint:
        Config.train.resume = True # Enable resume flag in config
        if args.checkpoint:
             Config.train.checkpoint_path = args.checkpoint # Set specific path

    # Create necessary directories based on Config
    os.makedirs(Config.train.checkpoint_dir, exist_ok=True)
    # Logs directory will be created by SummaryWriter or save_config

    # --- Model Training Loop ---
    for model_name in args.models:
        print(f"\n{'='*50}")
        print(f"Preparing to train: {model_name.upper()}")
        print(f"{'='*50}\n")

        # --- Get Model Class and Default Kwargs ---
        try:
            model_class, model_kwargs = get_model_config(model_name)
        except ValueError as e:
            print(f"Error: {e}")
            continue # Skip to the next model if name is invalid

        # --- Initialize Trainer ---
        # Trainer uses values from the (potentially updated) Config object
        trainer = Trainer(model_name=model_name) # Pass model_name

        # --- Save Configuration ---
        # Save the final configuration used for this specific model training run
        config_save_path = os.path.join("logs", model_name, "config_used.json")
        Config.save_to_json(config_save_path)

        # --- Load Data ---
        # Data loading is done once per training session if needed
        # If you train multiple models sequentially, you might optimize this
        if trainer.train_loader is None:
            try:
                trainer.load_data()
            except FileNotFoundError as e:
                 print(f"Error loading data: {e}")
                 print("Please ensure dataset paths in config.py or your config file are correct.")
                 continue # Skip this model if data cannot be loaded
            except Exception as e:
                 print(f"An unexpected error occurred during data loading: {e}")
                 continue

        # --- Initialize Model ---
        # Pass default kwargs; Trainer adds vocab_size
        try:
            trainer.initialize_model(model_class, **model_kwargs)
        except Exception as e:
            print(f"Error initializing model {model_name}: {e}")
            continue

        # Print model info
        print(f"Model Class: {trainer.model.__class__.__name__}")
        param_count = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {param_count:,}")
        print(f"Uses MHA Decoder: {trainer.model.has_mha_decoder}")


        # --- Load Checkpoint if Resuming ---
        start_epoch = 0
        best_val_loss = float('inf')
        if Config.train.resume:
            print("Attempting to resume training...")
            # Pass model_class and kwargs in case model needs re-initialization during loading
            start_epoch, best_val_loss = trainer.load_checkpoint(
                checkpoint_path=Config.train.checkpoint_path, # Use specific path from config if set
                model_class=model_class,
                **model_kwargs
            )
        else:
            print("Starting training from scratch.")


        # --- Start Training ---
        try:
            print(f"Starting training {model_name.upper()} from epoch {start_epoch}...")
            trainer.train(start_epoch=start_epoch, initial_best_val_loss=best_val_loss)
            print(f"\n{'='*50}")
            print(f"Training finished for: {model_name.upper()}")
            print(f"{'='*50}\n")
        except Exception as e:
            print(f"\nAn error occurred during training for {model_name}: {e}")
            # Consider logging the error traceback
            import traceback
            traceback.print_exc()
            print(f"Skipping further training for {model_name}.")
            continue # Move to the next model

if __name__ == "__main__":
    main()