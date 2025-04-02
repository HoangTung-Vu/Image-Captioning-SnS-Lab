import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
# import matplotlib.pyplot as plt # Removed, plotting done in evaluate
import numpy as np
from tqdm import tqdm
import os
import time # For timing epochs
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Type
from torch.utils.data import DataLoader
from utils.dataloader import get_loader, FlickrDataset
from utils.config import Config # Import Config
from model.base_model import BaseModel

class Trainer:
    def __init__(
        self,
        model_name: str, # Added model_name for logging/saving
        data_root: str = Config.data.data_root,
        captions_file: str = Config.data.captions_file,
        vocab_freq_threshold: int = Config.data.vocab_freq_threshold,
        learning_rate: float = Config.train.learning_rate,
        batch_size: int = Config.train.batch_size,
        num_epochs: int = Config.train.num_epochs,
        save_step: int = Config.train.save_step,
        checkpoint_dir: str = Config.train.checkpoint_dir,
        device: Optional[torch.device] = None,
        freeze_encoder_epochs: int = Config.train.freeze_encoder_epochs,
        use_mixed_precision: bool = Config.train.use_mixed_precision,
        early_stopping_patience: int = Config.train.early_stopping_patience,
        validation_split: float = Config.train.validation_split,
        num_workers: int = Config.train.num_workers
    ):
        """
        Initialize the Trainer using parameters from Config.

        Args:
            model_name: Name of the model being trained (e.g., 'cnntornn', 'transformer').
            data_root: Path to the dataset images.
            captions_file: Path to the captions file.
            vocab_freq_threshold: Minimum word frequency for vocabulary.
            learning_rate: Learning rate for optimizer.
            batch_size: Batch size for training/validation.
            num_epochs: Number of training epochs.
            save_step: Frequency of saving checkpoints (in epochs).
            checkpoint_dir: Base directory to save checkpoints.
            device: Device to run the model on (cuda or cpu).
            freeze_encoder_epochs: Number of epochs to freeze encoder.
            use_mixed_precision: Whether to use mixed precision training.
            early_stopping_patience: Patience for early stopping based on validation loss.
            validation_split: Fraction of data for validation.
            num_workers: Number of DataLoader workers.
        """
        self.model_name = model_name
        self.data_root = data_root
        self.captions_file = captions_file
        self.vocab_freq_threshold = vocab_freq_threshold
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_step = save_step
        # Model-specific checkpoint directory
        self.checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.num_workers = num_workers

        # Set device
        self.device = device or torch.device(Config.device) # Use Config device
        print(f"Using device: {self.device}")

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize components to None
        self.model: Optional[BaseModel] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.dataset: Optional[FlickrDataset] = None
        self.vocab_size: Optional[int] = None
        self.pad_idx: Optional[int] = None
        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.writer: Optional[SummaryWriter] = None

        # Initialize mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        if self.use_mixed_precision:
            print("Mixed precision training enabled.")

        # Image transform (standard Imagenet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5), # Data augmentation
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # More augmentation
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Load and prepare the dataset, splitting into train and validation sets.

        Returns:
            Tuple of (train_loader, val_loader)
        """
        print("Loading data...")
        # Load the full dataset using get_loader
        full_loader, self.dataset = get_loader(
            root_folder=self.data_root,
            annotation_file=self.captions_file,
            transform=self.transform,
            batch_size=self.batch_size, # Use instance batch size
            num_workers=self.num_workers,
            shuffle=True, # Shuffle full dataset before splitting
            pin_memory=False, # Pin memory handled by get_loader based on CUDA
            freq_threshold=self.vocab_freq_threshold,
            img_cache_size=100 # Example cache size, adjust as needed
        )

        # Set vocabulary size and padding index
        self.vocab_size = len(self.dataset.vocab)
        self.pad_idx = self.dataset.vocab.stoi["<PAD>"]
        print(f"Vocabulary size: {self.vocab_size}")

        # Split dataset into train and validation
        dataset_size = len(self.dataset)
        val_size = int(self.validation_split * dataset_size)
        train_size = dataset_size - val_size
        print(f"Dataset size: {dataset_size}, Train size: {train_size}, Validation size: {val_size}")

        # Use PyTorch's random_split for splitting
        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42) # for reproducible splits
        )

        # Create DataLoaders for train and validation sets
        # We need the collate_fn from the original loader/dataset
        collate_fn = full_loader.collate_fn

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True, # Shuffle training data each epoch
            num_workers=self.num_workers,
            pin_memory=full_loader.pin_memory,
            collate_fn=collate_fn
        )

        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False, # No need to shuffle validation data
            num_workers=self.num_workers,
            pin_memory=full_loader.pin_memory,
            collate_fn=collate_fn
        )
        print("Data loading and splitting complete.")
        return self.train_loader, self.val_loader

    def initialize_model(self, model_class: Type[BaseModel], **model_kwargs) -> BaseModel:
        """
        Initialize the model, loss function, optimizer, and TensorBoard writer.

        Args:
            model_class: The model class to instantiate (e.g., CNNtoRNN, VisionEncoderDecoder).
            model_kwargs: Keyword arguments specific to the model class constructor.

        Returns:
            Initialized model instance.
        """
        print(f"Initializing model: {model_class.__name__}")
        # Ensure vocab_size is available
        if self.vocab_size is None:
            raise ValueError("Vocabulary size not set. Call load_data() first.")
        if 'vocab_size' not in model_kwargs:
            model_kwargs['vocab_size'] = self.vocab_size

        # Initialize model and move to device
        self.model = model_class(**model_kwargs).to(self.device)

        # Initialize TensorBoard writer
        log_dir = os.path.join('runs', self.model_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")

        # Initialize loss function (ignore padding)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx) # Use standard reduction='mean'

        # Initialize optimizer (Adam is a common choice)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Initialize learning rate scheduler (ReduceLROnPlateau is good for validation loss)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',      # Reduce LR when validation loss stops decreasing
            factor=0.2,      # Factor to reduce LR by (lr * factor)
            patience=2,      # Number of epochs with no improvement to wait
            verbose=True     # Print message when LR is reduced
        )
        print("Model, Loss, Optimizer, Scheduler, and TensorBoard initialized.")
        return self.model

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        model_class: Optional[Type[BaseModel]] = None, # Needed if model not yet initialized
        **model_kwargs # Needed if model not yet initialized
    ) -> int:
        """
        Load model, optimizer, scheduler, and scaler state from a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file. If None, tries latest and best.
            model_class: The model class (required if self.model is None).
            model_kwargs: Model arguments (required if self.model is None).

        Returns:
            Starting epoch number (epoch saved + 1). Returns 0 if no checkpoint loaded.
        """
        load_path = checkpoint_path # Use provided path first

        # If no specific path, try 'latest_checkpoint.pth.tar'
        if load_path is None:
            latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth.tar')
            if os.path.exists(latest_path):
                load_path = latest_path
                print(f"No specific checkpoint path provided. Attempting to load latest: {load_path}")
            else:
                # If latest not found, try 'best_model.pth.tar'
                best_path = os.path.join(self.checkpoint_dir, 'best_model.pth.tar')
                if os.path.exists(best_path):
                    load_path = best_path
                    print(f"Latest checkpoint not found. Attempting to load best: {load_path}")


        if load_path and os.path.exists(load_path):
            print(f"Loading checkpoint from: {load_path}")
            try:
                checkpoint = torch.load(load_path, map_location=self.device)

                # --- Initialize components if they don't exist ---
                # Initialize model if necessary
                if self.model is None:
                    if model_class is None:
                        raise ValueError("model_class must be provided to load_checkpoint if model is not initialized.")
                    if self.dataset is None and 'vocab' not in checkpoint:
                         raise ValueError("Dataset must be loaded or vocab must be in checkpoint to initialize model.")
                    if 'vocab_size' not in model_kwargs:
                        # Try to get vocab_size from checkpoint if dataset not loaded
                        if 'vocab' in checkpoint and hasattr(checkpoint['vocab'], '__len__'):
                            model_kwargs['vocab_size'] = len(checkpoint['vocab'])
                        elif self.vocab_size:
                             model_kwargs['vocab_size'] = self.vocab_size
                        else:
                             raise ValueError("Cannot determine vocab_size for model initialization.")
                    self.initialize_model(model_class, **model_kwargs) # This also initializes optimizer, scheduler

                # --- Load states ---
                self.model.load_state_dict(checkpoint['model_state_dict'])

                # Load optimizer state (only if optimizer exists)
                if self.optimizer and 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                     print("Warning: Optimizer state not found in checkpoint or optimizer not initialized.")

                # Load scheduler state (only if scheduler exists)
                if self.scheduler and 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                else:
                    print("Warning: Scheduler state not found in checkpoint or scheduler not initialized.")


                # Load scaler state (only if scaler exists and using mixed precision)
                if self.scaler and 'scaler_state_dict' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                elif self.use_mixed_precision:
                     print("Warning: Scaler state not found in checkpoint or scaler not initialized.")


                # Load vocabulary (important for consistency)
                if 'vocab' in checkpoint:
                    if self.dataset is not None:
                        self.dataset.vocab = checkpoint['vocab']
                        self.vocab_size = len(self.dataset.vocab)
                        self.pad_idx = self.dataset.vocab.stoi["<PAD>"]
                        print("Loaded vocabulary from checkpoint.")
                    else:
                        print("Warning: Dataset not loaded, cannot overwrite vocabulary from checkpoint.")
                else:
                    print("Warning: Vocabulary not found in checkpoint.")


                start_epoch = checkpoint.get('epoch', -1) + 1 # Get epoch number, default to -1 if not found
                best_val_loss = checkpoint.get('best_val_loss', float('inf')) # Load best loss if saved
                print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}.")
                # Return start_epoch and best_val_loss for resuming training state
                return start_epoch, best_val_loss

            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting from scratch.")
                return 0, float('inf')
        else:
            print("No checkpoint found or specified. Starting training from scratch.")
            return 0, float('inf') # Start from epoch 0, best loss is infinity


    def save_checkpoint(self, epoch: int, val_loss: float, best_val_loss: float, is_best: bool) -> None:
        """
        Save model checkpoint, including model state, optimizer, scheduler, vocab, etc.

        Args:
            epoch: Current epoch number (0-based).
            val_loss: Validation loss for the current epoch.
            best_val_loss: Best validation loss achieved so far.
            is_best: Boolean indicating if this is the best model based on val_loss.
        """
        if self.model is None or self.optimizer is None or self.dataset is None:
            print("Warning: Cannot save checkpoint. Model, optimizer, or dataset not initialized.")
            return

        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss, # Save the best loss achieved
            'vocab': self.dataset.vocab, # Save vocabulary for consistent loading
            'model_name': self.model_name, # Save model name/type
            'model_class': self.model.__class__.__name__ # Save class name
        }

        # Add scheduler and scaler states if they exist
        if self.scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler is not None:
            checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()

        # Define file paths
        # Save checkpoint for the current epoch (optional, can take space)
        # epoch_checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch+1}.pth.tar')
        # torch.save(checkpoint_data, epoch_checkpoint_path)
        # print(f"Epoch checkpoint saved to {epoch_checkpoint_path}")

        # Always save as the latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth.tar')
        torch.save(checkpoint_data, latest_path)
        print(f"Latest checkpoint saved to {latest_path}")

        # Save as the best model if performance improved
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth.tar')
            torch.save(checkpoint_data, best_path)
            print(f"New best model saved to {best_path} (Val Loss: {val_loss:.4f})")


    def _prepare_batch(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare a batch for model input and loss calculation.
        Handles moving data to device and creating input/target sequences.

        Args:
            batch: Tuple of (images, captions) from the DataLoader.
                   images: [batch_size, C, H, W]
                   captions: [seq_len, batch_size] (includes <SOS> and <EOS>)

        Returns:
            Tuple containing:
            - imgs: Images on the correct device.
            - input_captions: Captions prepared for model input (e.g., excluding <EOS>). [input_seq_len, batch_size]
            - target_captions: Captions prepared for loss calculation (e.g., excluding <SOS>). [target_seq_len, batch_size]
            - target_captions_flat: Flattened target captions for loss. [target_seq_len * batch_size]
            - padding_mask: Boolean mask for attention (Transformer). [batch_size, input_seq_len]. True indicates padding.
        """
        imgs, captions = batch
        imgs = imgs.to(self.device, non_blocking=True)
        captions = captions.to(self.device, non_blocking=True)
        # captions shape: [seq_len, batch_size]

        # Input captions for the model (exclude <EOS> token)
        # input_captions shape: [seq_len - 1, batch_size]
        input_captions = captions[:-1, :]

        # Target captions for loss calculation (exclude <SOS> token)
        # target_captions shape: [seq_len - 1, batch_size]
        target_captions = captions[1:, :]

        # Flatten targets for CrossEntropyLoss
        # target_captions_flat shape: [(seq_len - 1) * batch_size]
        target_captions_flat = target_captions.reshape(-1)

        # Create padding mask for Transformer models (MHA expects True for padding)
        padding_mask = None
        if self.model and hasattr(self.model, 'has_mha_decoder') and self.model.has_mha_decoder:
            # Mask should correspond to input_captions
            # input_captions shape: [input_seq_len, batch_size]
            # padding_mask shape needs to be [batch_size, input_seq_len] for MHA key_padding_mask
            padding_mask = (input_captions == self.pad_idx).transpose(0, 1) # Transpose to [batch_size, seq_len]

        return imgs, input_captions, target_captions, target_captions_flat, padding_mask


    def train_epoch(self, epoch: int) -> float:
        """
        Train the model for one epoch.

        Args:
            epoch: Current epoch number (0-based).

        Returns:
            Average training loss for the epoch.
        """
        self.model.train() # Set model to training mode

        # Unfreeze encoder after specified epochs (only affects CNNtoRNN's Encoder)
        if epoch == self.freeze_encoder_epochs:
             if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'train_CNN'):
                 print(f"Epoch {epoch+1}: Unfreezing CNN encoder layers...")
                 self.model.encoder.train_CNN = True
                 # Option 1: Re-initialize optimizer (simpler, might lose momentum)
                 # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate * 0.1) # Use lower LR for fine-tuning
                 # Option 2: Add new parameters to existing optimizer group with lower LR
                 print("Adjusting optimizer for fine-tuning...")
                 finetune_lr = self.learning_rate * 0.1
                 base_params = [p for n, p in self.model.named_parameters() if p.requires_grad and not n.startswith('encoder.resnet')]
                 encoder_params = [p for n, p in self.model.named_parameters() if p.requires_grad and n.startswith('encoder.resnet')]
                 self.optimizer = optim.Adam([
                      {'params': base_params},
                      {'params': encoder_params, 'lr': finetune_lr}
                 ], lr=self.learning_rate)
                 print(f"Optimizer adjusted. Encoder LR: {finetune_lr}, Base LR: {self.learning_rate}")
             else:
                 print(f"Epoch {epoch+1}: Model does not have a 'train_CNN' attribute in its encoder. No unfreezing applied.")


        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        progress_bar = tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}/{self.num_epochs}", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            try:
                imgs, input_captions, _, target_captions_flat, padding_mask = self._prepare_batch(batch)

                self.optimizer.zero_grad()

                # Forward pass with Automatic Mixed Precision (AMP) if enabled
                if self.use_mixed_precision and self.scaler:
                    with torch.cuda.amp.autocast():
                        # outputs shape: [input_seq_len, batch_size, vocab_size]
                        outputs = self.model(imgs, input_captions, padding_mask=padding_mask)
                        # Reshape outputs for loss calculation: [(input_seq_len * batch_size), vocab_size]
                        loss = self.criterion(outputs.reshape(-1, self.vocab_size), target_captions_flat)

                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    # Update scaler for next iteration
                    self.scaler.update()
                else:
                    # Standard forward pass
                    outputs = self.model(imgs, input_captions, padding_mask=padding_mask)
                    loss = self.criterion(outputs.reshape(-1, self.vocab_size), target_captions_flat)

                    # Standard backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                progress_bar.set_postfix(loss=f"{batch_loss:.4f}")

                # Log batch loss to TensorBoard (optional, can be noisy)
                # if batch_idx % 100 == 0:
                #     step = epoch * num_batches + batch_idx
                #     self.writer.add_scalar('Loss/Train_Batch', batch_loss, step)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nWARNING: CUDA out of memory in batch {batch_idx}. Skipping batch.")
                    torch.cuda.empty_cache() # Clear cache
                    # Consider reducing batch size if this happens frequently
                else:
                    print(f"\nError during training batch {batch_idx}: {e}")
                    # Decide whether to continue or stop based on the error
                    # continue
                    raise e # Re-raise other runtime errors
            except Exception as e:
                 print(f"\nUnexpected error during training batch {batch_idx}: {e}")
                 raise e # Re-raise unexpected errors

        avg_epoch_loss = epoch_loss / max(1, num_batches) # Avoid division by zero
        return avg_epoch_loss


    def validate(self, epoch: int) -> float:
        """
        Validate the model on the validation set.

        Args:
            epoch: Current epoch number (0-based).

        Returns:
            Average validation loss for the epoch.
        """
        if self.val_loader is None:
            print("Warning: Validation loader not available. Skipping validation.")
            return float('inf') # Return infinity if no validation is possible

        self.model.eval() # Set model to evaluation mode
        val_loss = 0.0
        num_batches = len(self.val_loader)
        progress_bar = tqdm(self.val_loader, desc=f"Valid Epoch {epoch+1}/{self.num_epochs}", leave=False)

        with torch.no_grad(): # Disable gradient calculations
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    imgs, input_captions, _, target_captions_flat, padding_mask = self._prepare_batch(batch)

                    # Forward pass (no need for AMP context manager in no_grad)
                    outputs = self.model(imgs, input_captions, padding_mask=padding_mask)
                    loss = self.criterion(outputs.reshape(-1, self.vocab_size), target_captions_flat)

                    batch_loss = loss.item()
                    val_loss += batch_loss
                    progress_bar.set_postfix(loss=f"{batch_loss:.4f}")

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\nWARNING: CUDA out of memory during validation batch {batch_idx}. Skipping batch.")
                        torch.cuda.empty_cache()
                    else:
                        print(f"\nError during validation batch {batch_idx}: {e}")
                        # Usually, we don't want to stop validation, just report the issue
                        # Continue if possible, or return a high loss value
                except Exception as e:
                     print(f"\nUnexpected error during validation batch {batch_idx}: {e}")
                     # Continue if possible

        avg_val_loss = val_loss / max(1, num_batches)
        return avg_val_loss


    def train(self, start_epoch: int = 0, initial_best_val_loss: float = float('inf')) -> None:
        """
        Run the main training loop.

        Args:
            start_epoch: The epoch number to start training from (usually 0 or loaded from checkpoint).
            initial_best_val_loss: The best validation loss loaded from checkpoint (or infinity).
        """
        if not all([self.model, self.train_loader, self.val_loader, self.optimizer, self.criterion, self.writer]):
             raise RuntimeError("Trainer components not fully initialized. Call load_data() and initialize_model() first.")

        print(f"\n{'='*20} Starting Training {'='*20}")
        print(f"Model: {self.model_name}, Epochs: {self.num_epochs}, Device: {self.device}")
        print(f"Batch Size: {self.batch_size}, LR: {self.learning_rate}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"{'='*56}")


        best_val_loss = initial_best_val_loss
        patience_counter = 0

        for epoch in range(start_epoch, self.num_epochs):
            epoch_start_time = time.time()

            # Train one epoch
            train_loss = self.train_epoch(epoch)

            # Validate one epoch
            val_loss = self.validate(epoch)

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            # Log metrics to TensorBoard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('EpochTime', epoch_duration, epoch)


            # Print epoch results
            print(f"Epoch [{epoch+1}/{self.num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.1e} | "
                  f"Time: {epoch_duration:.2f}s")


            # Learning rate scheduling step (based on validation loss)
            if self.scheduler:
                self.scheduler.step(val_loss)

            # Check for improvement in validation loss
            is_best = val_loss < best_val_loss
            if is_best:
                print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
                best_val_loss = val_loss
                patience_counter = 0 # Reset patience counter
            else:
                patience_counter += 1
                print(f"Validation loss did not improve. Patience: {patience_counter}/{self.early_stopping_patience}")

            # Save checkpoint (latest and potentially best)
            if (epoch + 1) % self.save_step == 0 or is_best:
                 self.save_checkpoint(epoch, val_loss, best_val_loss, is_best)

            # Early stopping check
            if patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs due to no improvement in validation loss.")
                break

        self.writer.close()
        print("\nTraining complete!")
        print(f"Best validation loss achieved: {best_val_loss:.4f}")


    def evaluate_on_examples(self, num_examples: int = 5) -> None:
        """
        Evaluate the model on a few random examples from the validation set and print captions.
        (This is primarily for quick checks during training, full evaluation is separate).

        Args:
            num_examples: Number of examples to evaluate.
        """
        if self.model is None or self.val_loader is None or self.dataset is None:
             print("Cannot evaluate examples: Model, validation loader, or dataset not available.")
             return

        self.model.eval() # Set to evaluation mode
        print("\n--- Evaluating on Random Validation Examples ---")

        # Get random examples from the validation subset of the dataset
        val_indices = self.val_loader.dataset.indices # Get indices used in validation split
        if not val_indices:
             print("No validation indices found.")
             return

        num_to_show = min(num_examples, len(val_indices))
        random_indices = np.random.choice(val_indices, num_to_show, replace=False)

        for i, data_idx in enumerate(random_indices):
             try:
                 # Get original image and caption using the main dataset
                 img_tensor, caption_indices = self.dataset[data_idx]
                 img_tensor = img_tensor.unsqueeze(0).to(self.device) # Add batch dim and move to device

                 # Generate captions
                 greedy_caption = "N/A"
                 beam_caption = "N/A"

                 if hasattr(self.model, 'caption_image_greedy'):
                      greedy_tokens = self.model.caption_image_greedy(img_tensor, self.dataset.vocab)
                      greedy_caption = self._tokens_to_caption(greedy_tokens)
                 else:
                      print(f"Model {self.model_name} does not implement caption_image_greedy.")

                 if hasattr(self.model, 'caption_image_beam_search'):
                      beam_tokens = self.model.caption_image_beam_search(img_tensor, self.dataset.vocab, beam_size=3) # Use small beam for quick check
                      beam_caption = self._tokens_to_caption(beam_tokens)
                 else:
                      print(f"Model {self.model_name} does not implement caption_image_beam_search.")


                 # Convert ground truth caption indices to words
                 true_caption_tokens = [self.dataset.vocab.itos[idx.item()] for idx in caption_indices]
                 true_caption = self._tokens_to_caption(true_caption_tokens) # Use helper to clean

                 print(f"\nExample {i+1} (Dataset index: {data_idx}):")
                 print(f"  Ground Truth: {true_caption}")
                 print(f"  Greedy Pred:  {greedy_caption}")
                 print(f"  Beam Pred:    {beam_caption}")

             except Exception as e:
                 print(f"Error evaluating example with dataset index {data_idx}: {e}")
                 continue # Continue to the next example

        print("--- End of Example Evaluation ---")
        self.model.train() # Set back to training mode if called during training loop


    def _tokens_to_caption(self, tokens: List[str]) -> str:
        """
        Convert a list of tokens (words) into a clean caption string.
        Removes special tokens like <SOS>, <EOS>, <PAD>, <UNK>.

        Args:
            tokens: List of token strings.

        Returns:
            Cleaned caption string.
        """
        # Filter out special tokens
        filtered_tokens = [
            token for token in tokens
            if token not in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]
        ]
        # Join tokens into a single string
        return ' '.join(filtered_tokens)