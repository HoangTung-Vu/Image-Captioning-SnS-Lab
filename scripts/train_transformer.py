import os
import sys
import torch
import argparse

# Add parent directory to path to allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.cnntornn import CNNtoRNN
from utils.trainer import Trainer
from utils.config import Config

def main():
    """
    Main function to train the image captioning model
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train an image captioning model")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        Config.load_from_json(args.config)
    
    # Create directories if they don't exist
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Save configuration for reproducibility
    Config.save_to_json("logs/config.json")
    
    # Initialize trainer
    trainer = Trainer(
        data_root=Config.data.data_root,
        captions_file=Config.data.captions_file,
        learning_rate=Config.train.learning_rate,
        batch_size=Config.train.batch_size,
        num_epochs=Config.train.num_epochs,
        save_step=Config.train.save_step,
        checkpoint_dir=Config.train.checkpoint_dir,
        device=torch.device(Config.device),
        freeze_encoder_epochs=Config.train.freeze_encoder_epochs,
        use_mixed_precision=Config.train.use_mixed_precision,
        early_stopping_patience=Config.train.early_stopping_patience
    )
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = trainer.load_data()
    
    # Get model class based on model argument
    model_class = CNNtoRNN  # Default model
    
    # You can add more model types here as they are implemented
    # if args.model == "your_custom_model":
    #     from model.your_custom_model import YourCustomModel
    #     model_class = YourCustomModel
    
    # Initialize model with model-specific parameters
    # These parameters should be defined in the model class itself
    print("Initializing model...")
    model_kwargs = {
        'embed_size': 256,
        'hidden_size': 512,
        'num_layers': 1,
        'trainCNN': False,
        'dropout_rate': 0.5
    }
    
    model = trainer.initialize_model(model_class, **model_kwargs)
    

    start_epoch = 0
    if args.resume or Config.train.resume:
        if Config.train.checkpoint_path:
            start_epoch = trainer.load_checkpoint(
                checkpoint_path=Config.train.checkpoint_path,
                model_class=model_class,
                **model_kwargs
            )
        else:
            # Try to load latest checkpoint
            start_epoch = trainer.load_checkpoint(
                model_class=model_class,
                **model_kwargs
            )
    
    # Train model
    print(f"Starting training from epoch {start_epoch}...")
    trainer.train(start_epoch=start_epoch)
    
    print("Training complete!")

if __name__ == "__main__":
    main()

