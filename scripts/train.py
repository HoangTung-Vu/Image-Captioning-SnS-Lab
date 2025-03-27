import os
import sys
import torch
import argparse
import torchvision.transforms as transforms

# Add parent directory to path to allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.cnntornn import CNNtoRNN
from utils.trainer import Trainer
from utils.config import Config
from utils.dataloader import get_loader

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
        embed_size=Config.model.embed_size,
        hidden_size=Config.model.hidden_size,
        num_layers=Config.model.num_layers,
        learning_rate=Config.train.learning_rate,
        batch_size=Config.train.batch_size,
        num_epochs=Config.train.num_epochs,
        save_step=Config.train.save_step,
        checkpoint_dir="checkpoints",  # Use absolute path from project root
        device=torch.device(Config.device),
        freeze_cnn_epochs=Config.train.freeze_cnn_epochs,
        use_mixed_precision=Config.train.use_mixed_precision,
        early_stopping_patience=Config.train.early_stopping_patience
    )
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = trainer.load_data()
    
    # Get vocabulary size from dataset
    vocab_size = len(trainer.dataset.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Initialize model
    print("Initializing model...")
    model = trainer.initialize_model(
        model_class=CNNtoRNN,
        embed_size=Config.model.embed_size,
        hidden_size=Config.model.hidden_size,
        vocab_size=vocab_size,
        num_layers=Config.model.num_layers,
        dropout_rate=Config.model.dropout
    )
    
    # Load checkpoint if resuming training
    start_epoch = 0
    if args.resume or Config.train.resume:
        if Config.train.checkpoint_path:
            start_epoch = trainer.load_checkpoint(
                checkpoint_path=Config.train.checkpoint_path,
                model_class=CNNtoRNN,
                embed_size=Config.model.embed_size,
                hidden_size=Config.model.hidden_size,
                vocab_size=vocab_size,
                num_layers=Config.model.num_layers,
                dropout_rate=Config.model.dropout
            )
        else:
            # Try to load latest checkpoint
            start_epoch = trainer.load_checkpoint(
                model_class=CNNtoRNN,
                embed_size=Config.model.embed_size,
                hidden_size=Config.model.hidden_size,
                vocab_size=vocab_size,
                num_layers=Config.model.num_layers,
                dropout_rate=Config.model.dropout
            )
    
    # Train model
    print(f"Starting training from epoch {start_epoch}...")
    trainer.train(start_epoch=start_epoch)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
    