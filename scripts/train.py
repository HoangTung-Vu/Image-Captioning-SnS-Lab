import os
import sys
import torch
import argparse
from typing import Dict, Any, Optional, Type, List, Tuple
import importlib

# Add parent directory to path to allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.trainer import Trainer
from utils.config import Config
from model.base_model import BaseModel
from model.cnntornn import CNNtoRNN
from model.transformer import VisionEncoderDecoder

def main():
    """
    Main function to train image captioning models
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train image captioning models")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--models", type=str, nargs='+', default=["transformer", "cnntornn"], 
                        help="Models to train: transformer, cnntornn, or both")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Train each specified model
    for model_name in args.models:
        print(f"\n{'='*50}")
        print(f"Training {model_name.upper()} model")
        print(f"{'='*50}\n")
        
        # Load model-specific config if available
        model_config_path = args.config
        if model_config_path is None:
            # Try to find model-specific config
            default_config_path = f"configs/{model_name.lower()}_config.json"
            if os.path.exists(default_config_path):
                model_config_path = default_config_path
                print(f"Using model-specific config: {model_config_path}")
        
        # Load configuration
        if model_config_path and os.path.exists(model_config_path):
            Config.load_from_json(model_config_path)
            print(f"Loaded configuration from {model_config_path}")
        else:
            print("Using default configuration")
        
        # Save configuration for reproducibility
        os.makedirs(f"logs/{model_name}", exist_ok=True)
        Config.save_to_json(f"logs/{model_name}/config.json")
        
        # Initialize trainer
        trainer = Trainer(
            data_root=Config.data.data_root,
            captions_file=Config.data.captions_file,
            learning_rate=Config.train.learning_rate,
            batch_size=Config.train.batch_size,
            num_epochs=Config.train.num_epochs,
            save_step=Config.train.save_step,
            checkpoint_dir=os.path.join(Config.train.checkpoint_dir, model_name),
            device=torch.device(Config.device),
            freeze_encoder_epochs=Config.train.freeze_encoder_epochs,
            use_mixed_precision=Config.train.use_mixed_precision,
            early_stopping_patience=Config.train.early_stopping_patience
        )
        
        # Load data
        print("Loading data...")
        train_loader, val_loader = trainer.load_data()
        
        # Get model class and kwargs
        model_class, model_kwargs = get_model_config(model_name)
        
        # Initialize model
        print(f"Initializing {model_name} model...")
        model = trainer.initialize_model(model_class, **model_kwargs)
        
        # Print model information
        print(f"Model: {model.__class__.__name__}")
        print(f"Has MHA Decoder: {model.has_mha_decoder}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Load checkpoint if resuming
        start_epoch = 0
        if args.resume or Config.train.resume:
            checkpoint_path = None
            if hasattr(Config.train, 'checkpoint_path') and Config.train.checkpoint_path:
                checkpoint_path = Config.train.checkpoint_path
            
            start_epoch = trainer.load_checkpoint(
                checkpoint_path=checkpoint_path,
                model_class=model_class,
                **model_kwargs
            )
        
        # Train model
        print(f"Starting training from epoch {start_epoch}...")
        trainer.train(start_epoch=start_epoch)
        
        print(f"Training complete for {model_name} model!")

def get_model_config(model_name: str) -> Tuple[Type[BaseModel], Dict[str, Any]]:
    """
    Get the model class and configuration based on the model name
    
    Args:
        model_name: Name of the model
        
    Returns:
        Tuple of (model_class, model_kwargs)
    """
    # Model-specific parameters
    if model_name.lower() == "transformer":
        return VisionEncoderDecoder, {
            'image_size': 224,
            'channels_in': 3,
            'patch_size': 16,
            'hidden_size': 512,
            'num_layers': (6, 6),  # (encoder_layers, decoder_layers)
            'num_heads': 8,
        }
    elif model_name.lower() == "cnntornn":
        return CNNtoRNN, {
            'embed_size': 256,
            'hidden_size': 512,
            'num_layers': 2,
            'dropout_rate': 0.5,
            'trainCNN': False,  # Will be set to True after freeze_encoder_epochs
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")

if __name__ == "__main__":
    main()

