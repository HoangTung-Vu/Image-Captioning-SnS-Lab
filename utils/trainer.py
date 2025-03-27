import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Type
from utils.dataloader import get_loader
from model.base_model import BaseModel

class Trainer:
    def __init__(
        self, 
        data_root: str = 'data/flickr8k/Flicker8k_Dataset',
        captions_file: str = 'data/flickr8k/captions.txt',
        learning_rate: float = 3e-4,
        batch_size: int = 32,
        num_epochs: int = 10,
        save_step: int = 1,
        checkpoint_dir: str = 'checkpoints',
        device: Optional[torch.device] = None,
        freeze_encoder_epochs: int = 3,
        use_mixed_precision: bool = True,
        early_stopping_patience: int = 5
    ):
        """
        Initialize the Trainer with hyperparameters
        
        Args:
            data_root: Path to the dataset images
            captions_file: Path to the captions file
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            save_step: Frequency of saving checkpoints (in epochs)
            checkpoint_dir: Directory to save checkpoints
            device: Device to run the model on (cuda or cpu)
            freeze_encoder_epochs: Number of epochs to freeze encoder
            use_mixed_precision: Whether to use mixed precision training
            early_stopping_patience: Number of epochs to wait for improvement before stopping
        """
        # Hyperparameters
        self.data_root = data_root
        self.captions_file = captions_file
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_step = save_step
        self.checkpoint_dir = checkpoint_dir
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.early_stopping_patience = early_stopping_patience
        
        # Set device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        # Initialize tensorboard writer
        self.writer = SummaryWriter('runs/image_captioning')
        
        # Initialize model, data loader, etc. to None
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.dataset = None
        self.vocab_size = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
        # Initialize mixed precision scaler
        self.scaler = torch.amp.GradScaler() if self.use_mixed_precision else None
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    def load_data(self, val_split: float = 0.1) -> Tuple[Any, Any]:
        """
        Load and prepare the dataset
        
        Args:
            val_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Load data
        loader, self.dataset = get_loader(
            root_folder=self.data_root,
            annotation_file=self.captions_file,
            transform=self.transform,
            batch_size=self.batch_size,
            # Disable pin_memory to avoid CUDA errors
            pin_memory=False,
            num_workers=2  # Reduce number of workers to avoid memory issues
        )
        
        # Set vocabulary size
        self.vocab_size = len(self.dataset.vocab)
        
        # Split into train and validation
        dataset_size = len(loader.dataset)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        
        # Create train and validation datasets
        train_dataset, val_dataset = torch.utils.data.random_split(
            loader.dataset, [train_size, val_size]
        )
        
        # Create train and validation loaders
        pad_idx = self.dataset.vocab.stoi["<PAD>"]
        
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,  # Reduce number of workers
            pin_memory=False,  # Disable pin_memory
            collate_fn=loader.collate_fn
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,  # Reduce number of workers
            pin_memory=False,  # Disable pin_memory
            collate_fn=loader.collate_fn
        )
        
        return self.train_loader, self.val_loader
    
    def initialize_model(self, model_class: Type[BaseModel], **model_kwargs) -> BaseModel:
        """
        Initialize the model, loss function, and optimizer
        
        Args:
            model_class: The model class to instantiate
            model_kwargs: Additional keyword arguments for the model
            
        Returns:
            Initialized model
        """
        # Add vocab_size to model parameters if not provided
        if 'vocab_size' not in model_kwargs and self.vocab_size is not None:
            model_kwargs['vocab_size'] = self.vocab_size
            
        # Initialize model
        self.model = model_class(**model_kwargs).to(self.device)
        
        # Initialize loss function and optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.dataset.vocab.stoi["<PAD>"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2, 
            verbose=True
        )
        
        return self.model
    
    def load_checkpoint(
        self, 
        checkpoint_path: Optional[str] = None, 
        model_class: Optional[Type[BaseModel]] = None, 
        **model_kwargs
    ) -> int:
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to the checkpoint file
            model_class: The model class to instantiate if model is not initialized
            model_kwargs: Additional keyword arguments for the model
            
        Returns:
            Starting epoch number
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth.tar')
            
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Initialize model if not already initialized
            if self.model is None and model_class is not None:
                # Add vocab_size to model parameters if not provided
                if 'vocab_size' not in model_kwargs and 'vocab' in checkpoint:
                    model_kwargs['vocab_size'] = len(checkpoint['vocab'])
                    
                # Initialize model
                self.model = model_class(**model_kwargs).to(self.device)
                
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Initialize optimizer if not already initialized
            if self.optimizer is None:
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
                
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler if it exists in checkpoint
            if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            # Load scaler if it exists in checkpoint and we're using mixed precision
            if 'scaler_state_dict' in checkpoint and self.scaler is not None:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # Load vocabulary if it exists in checkpoint
            if 'vocab' in checkpoint and self.dataset is not None:
                self.dataset.vocab = checkpoint['vocab']
                self.vocab_size = len(checkpoint['vocab'])
            
            start_epoch = checkpoint['epoch'] + 1
            print(f"Loaded checkpoint from epoch {start_epoch-1}")
            return start_epoch
        else:
            print("No checkpoint found, starting from scratch")
            return 0
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False) -> None:
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch number
            loss: Current loss value
            is_best: Whether this is the best model so far
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch+1}.pth.tar')
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'vocab': self.dataset.vocab,
        }
        
        # Add scheduler and scaler if they exist
        if self.scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = self.scheduler.state_dict()
            
        if self.scaler is not None:
            checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Also save as latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth.tar')
        torch.save(checkpoint_data, latest_path)
        
        # Save as best model if specified
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth.tar')
            torch.save(checkpoint_data, best_path)
            print(f"New best model saved at {best_path}")
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average loss for the epoch
        """
        # Set model to train mode
        self.model.train()
        
        # Initialize metrics
        epoch_loss = 0
        num_batches = len(self.train_loader)
        
        # Iterate over batches
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")
        for idx, (imgs, captions) in enumerate(progress_bar):
            try:
                # Move to device safely
                imgs = imgs.to(self.device, non_blocking=False)
                captions = captions.to(self.device, non_blocking=False)
                
                # Forward pass with mixed precision if enabled
                self.optimizer.zero_grad()
                
                if self.use_mixed_precision and self.scaler is not None:
                    with torch.amp.autocast(device_type=self.device.type):
                        outputs = self.model(imgs, captions[:-1])  # Remove <EOS> token for input
                        loss = self.criterion(
                            outputs.reshape(-1, self.vocab_size),
                            captions.reshape(-1)
                        )
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard forward and backward pass
                    outputs = self.model(imgs, captions[:-1])
                    loss = self.criterion(
                        outputs.reshape(-1, self.vocab_size),
                        captions.reshape(-1)
                    )
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix(loss=loss.item())
                
                # Log to tensorboard (every 100 batches)
                if idx % 100 == 0:
                    self.writer.add_scalar('Training/BatchLoss', loss.item(), epoch * num_batches + idx)
                    
            except RuntimeError as e:
                print(f"Error in batch {idx}: {e}")
                # Skip this batch and continue with the next one
                continue
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / num_batches
        
        # Log to tensorboard
        self.writer.add_scalar('Training/EpochLoss', avg_loss, epoch)
        
        return avg_loss
    
    def validate(self, epoch: int) -> float:
        """
        Validate the model
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average validation loss
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        val_loss = 0
        num_batches = len(self.val_loader)
        
        # Iterate over batches
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}")
            for idx, (imgs, captions) in enumerate(progress_bar):
                try:
                    # Move to device safely
                    imgs = imgs.to(self.device, non_blocking=False)
                    captions = captions.to(self.device, non_blocking=False)
                    
                    # Forward pass
                    outputs = self.model(imgs, captions[:-1])
                    
                    # Calculate loss
                    loss = self.criterion(
                        outputs.reshape(-1, self.vocab_size),
                        captions.reshape(-1)
                    )
                    
                    # Update metrics
                    val_loss += loss.item()
                    
                    # Update progress bar
                    progress_bar.set_postfix(loss=loss.item())
                    
                except RuntimeError as e:
                    print(f"Error in validation batch {idx}: {e}")
                    # Skip this batch and continue with the next one
                    continue
        
        # Calculate average loss for the epoch
        avg_val_loss = val_loss / max(1, num_batches)  # Avoid division by zero
        
        # Log to tensorboard
        self.writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
        
        return avg_val_loss
    
    def train(self, start_epoch: int = 0, unfreeze_encoder: bool = True) -> None:
        """
        Train the model
        
        Args:
            start_epoch: Starting epoch number
            unfreeze_encoder: Whether to unfreeze the encoder after freeze_encoder_epochs
        """
        print("Starting training...")
        
        # Initialize best validation loss and patience counter
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(start_epoch, self.num_epochs):
            print(f"Epoch [{epoch+1}/{self.num_epochs}]")
            
            # Unfreeze encoder after specified number of epochs if model has encoder attribute
            if unfreeze_encoder and epoch == self.freeze_encoder_epochs and hasattr(self.model, 'encoder'):
                print("Unfreezing encoder layers...")
                # Different models might have different ways to control encoder training
                # For CNNtoRNN, it's the train_CNN attribute
                if hasattr(self.model.encoder, 'train_CNN'):
                    self.model.encoder.train_CNN = True
                # Reinitialize optimizer with all parameters
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate * 0.1)  # Lower learning rate for fine-tuning
            
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Print epoch results
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Check if this is the best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save model checkpoint
            if (epoch + 1) % self.save_step == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
            
            # Evaluate model on a few examples
            if (epoch + 1) % self.save_step == 0:
                self.evaluate_model()
                
            # Early stopping
            if patience_counter >= self.early_stopping_patience:
                print(f"Early stopping after {patience_counter} epochs without improvement")
                break
        
        self.writer.close()
        print("Training complete!")
    
    def evaluate_model(self, num_examples: int = 5) -> None:
        """
        Evaluate the model on a few examples and print the results
        
        Args:
            num_examples: Number of examples to evaluate
        """
        self.model.eval()
        
        # Get a few random images from the dataset
        indices = np.random.choice(len(self.dataset), num_examples, replace=False)
        
        print("\nEvaluation examples:")
        for idx in indices:
            try:
                img, caption = self.dataset[idx]
                img = img.unsqueeze(0).to(self.device, non_blocking=False)
                
                # Generate caption
                with torch.no_grad():
                    # Try both methods if they exist
                    if hasattr(self.model, 'caption_image_greedy'):
                        greedy_caption = self.model.caption_image_greedy(img, self.dataset.vocab)
                        greedy_caption = ' '.join([word for word in greedy_caption 
                                                if word not in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]])
                    else:
                        greedy_caption = "Greedy caption method not available"
                    
                    if hasattr(self.model, 'caption_image_beam_search'):
                        beam_caption = self.model.caption_image_beam_search(img, self.dataset.vocab, beam_size=3)
                        beam_caption = ' '.join([word for word in beam_caption 
                                                if word not in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]])
                    else:
                        beam_caption = "Beam search method not available"
                
                # Convert ground truth caption to words
                true_caption = [self.dataset.vocab.itos[idx.item()] for idx in caption]
                
                # Remove special tokens and join words
                true_caption = ' '.join([word for word in true_caption 
                                        if word not in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]])
                
                print(f"Image {idx}:")
                print(f"Ground truth: {true_caption}")
                print(f"Greedy prediction: {greedy_caption}")
                print(f"Beam search prediction: {beam_caption}")
                print()
                
            except Exception as e:
                print(f"Error evaluating example {idx}: {e}")
                continue

