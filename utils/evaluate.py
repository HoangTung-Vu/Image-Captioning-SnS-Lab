import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Type
from utils.dataloader import get_loader
from functools import lru_cache
import pandas as pd
from collections import defaultdict
from model.base_model import BaseModel

class Evaluator:
    def __init__(
        self, 
        data_root: str = 'data/flickr8k/Flicker8k_Dataset',
        captions_file: str = 'data/flickr8k/captions.txt',
        checkpoint_path: str = 'checkpoints/latest_checkpoint.pth.tar',
        beam_search: bool = True,
        device: Optional[torch.device] = None,
        batch_size: int = 1,
        visualization_dir: str = 'visualization_results'
    ):
        """
        Initialize the Evaluator with parameters
        
        Args:
            data_root: Path to the dataset images
            captions_file: Path to the captions file
            checkpoint_path: Path to the model checkpoint
            beam_search: Whether to use beam search for caption generation
            device: Device to run the model on (cuda or cpu)
            batch_size: Batch size for evaluation
            visualization_dir: Directory to save visualization results
        """
        # Parameters
        self.data_root = data_root
        self.captions_file = captions_file
        self.checkpoint_path = checkpoint_path
        self.beam_search = beam_search
        self.batch_size = batch_size
        self.visualization_dir = visualization_dir
        
        # Create visualization directory if it doesn't exist
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Set device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # Download NLTK resources
        self._download_nltk_resources()
        
        # Initialize model, data loader, etc. to None
        self.model = None
        self.vocab = None
        self.data_loader = None
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
    def _download_nltk_resources(self) -> None:
        """Download required NLTK resources"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def load_model(
        self, 
        model_class: Type[BaseModel], 
        **model_kwargs
    ) -> Tuple[BaseModel, Any]:
        """
        Load model from checkpoint
        
        Args:
            model_class: The model class to instantiate
            model_kwargs: Additional keyword arguments for the model
            
        Returns:
            Tuple of (model, vocab)
        """
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {self.checkpoint_path}")
        
        # Load checkpoint with appropriate device mapping
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.vocab = checkpoint['vocab']
        
        # Add vocab_size to model parameters if not provided
        if 'vocab_size' not in model_kwargs and hasattr(self.vocab, '__len__'):
            model_kwargs['vocab_size'] = len(self.vocab)
        
        # Initialize model
        self.model = model_class(**model_kwargs).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return self.model, self.vocab
    
    def load_data(self) -> Any:
        """
        Load and prepare the dataset
        
        Returns:
            DataLoader object
        """
        # Load data
        self.data_loader, _ = get_loader(
            root_folder=self.data_root,
            annotation_file=self.captions_file,
            transform=self.transform,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle for evaluation
            num_workers=4,  # Use multiple workers for faster loading
        )
        
        return self.data_loader
    
    def _group_references_by_image(self) -> Dict[str, List[List[str]]]:
        """
        Group reference captions by image ID
        
        Returns:
            Dictionary mapping image IDs to lists of reference captions
        """
        # Load captions file
        df = pd.read_csv(self.captions_file)
        
        # Group by image ID
        references_by_image = defaultdict(list)
        
        for _, row in df.iterrows():
            image_id = row['image']
            caption = row['caption']
            
            # Tokenize caption
            tokens = self.vocab.tokenize(caption)
            
            # Add to references
            references_by_image[image_id].append(tokens)
        
        return references_by_image
    
    @torch.no_grad()  # Disable gradient calculation for evaluation
    def evaluate_bleu(self) -> Dict[str, float]:
        """
        Evaluate model using BLEU score with multiple references per image
        
        Returns:
            Dictionary of BLEU scores
        """
        if self.model is None:
            raise ValueError("Model must be loaded before evaluation")
            
        self.model.eval()
        
        # Group reference captions by image ID
        references_by_image = self._group_references_by_image()
        
        # Lists to store references and hypotheses
        all_references = []
        all_hypotheses = []
        
        # Dictionary to store image IDs and their predictions
        image_predictions = {}
        
        # Load the dataset
        df = pd.read_csv(self.captions_file)
        unique_images = df['image'].unique()
        
        # Process each unique image
        print("Generating captions for evaluation...")
        for image_id in tqdm(unique_images):
            # Load and process image
            image_path = os.path.join(self.data_root, image_id)
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Generate caption
            if self.beam_search and hasattr(self.model, 'caption_image_beam_search'):
                predicted_caption = self.model.caption_image_beam_search(image_tensor, self.vocab, beam_size=3)
            elif hasattr(self.model, 'caption_image_greedy'):
                predicted_caption = self.model.caption_image_greedy(image_tensor, self.vocab)
            else:
                raise ValueError("Model must have caption_image_greedy or caption_image_beam_search method")
            
            # Process predicted caption
            hypothesis = [token for token in predicted_caption if token not in ["<SOS>", "<PAD>"]]
            # Stop at EOS token
            if "<EOS>" in hypothesis:
                hypothesis = hypothesis[:hypothesis.index("<EOS>")]
            
            # Get references for this image
            references = references_by_image[image_id]
            
            # Add to lists for BLEU calculation
            all_references.append(references)
            all_hypotheses.append(hypothesis)
            
            # Store prediction for later use
            image_predictions[image_id] = hypothesis
        
        # Calculate BLEU scores
        print("Calculating BLEU scores...")
        smoothing = SmoothingFunction().method1
        
        # Calculate BLEU scores for different n-gram weights
        bleu_scores = {
            "BLEU-1": self._calculate_bleu(all_references, all_hypotheses, (1, 0, 0, 0), smoothing),
            "BLEU-2": self._calculate_bleu(all_references, all_hypotheses, (0.5, 0.5, 0, 0), smoothing),
            "BLEU-3": self._calculate_bleu(all_references, all_hypotheses, (0.33, 0.33, 0.33, 0), smoothing),
            "BLEU-4": self._calculate_bleu(all_references, all_hypotheses, (0.25, 0.25, 0.25, 0.25), smoothing)
        }
        
        return bleu_scores
    
    @staticmethod
    def _calculate_bleu(
        references: List[List[List[str]]], 
        hypotheses: List[List[str]], 
        weights: Tuple[float, ...], 
        smoothing_function: Callable
    ) -> float:
        """
        Calculate BLEU score with specific weights
        
        Args:
            references: List of lists of reference captions
            hypotheses: List of predicted captions
            weights: Weights for n-grams
            smoothing_function: Smoothing function for BLEU calculation
            
        Returns:
            BLEU score (0-100)
        """
        return corpus_bleu(
            references, 
            hypotheses, 
            weights=weights, 
            smoothing_function=smoothing_function
        ) * 100
    
    @torch.no_grad()  # Disable gradient calculation for visualization
    def visualize_examples(self, num_examples: int = 10) -> None:
        """
        Visualize examples with multiple references
        
        Args:
            num_examples: Number of examples to visualize
        """
        if self.model is None:
            raise ValueError("Model must be loaded before visualization")
            
        self.model.eval()
        
        # Load the dataset
        df = pd.read_csv(self.captions_file)
        
        # Group captions by image
        image_captions = df.groupby('image')['caption'].apply(list).reset_index()
        
        # Get random indices
        indices = np.random.choice(len(image_captions), min(num_examples, len(image_captions)), replace=False)
        
        for i, idx in enumerate(indices):
            # Get image and captions
            image_id = image_captions.iloc[idx]['image']
            reference_captions = image_captions.iloc[idx]['caption']
            
            # Load and process image
            image_path = os.path.join(self.data_root, image_id)
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Generate captions using different methods
            greedy_caption = "Greedy caption method not available"
            beam_caption = "Beam search method not available"
            
            if hasattr(self.model, 'caption_image_greedy'):
                greedy_tokens = self.model.caption_image_greedy(image_tensor, self.vocab)
                greedy_caption = ' '.join([token for token in greedy_tokens 
                                          if token not in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]])
            
            if hasattr(self.model, 'caption_image_beam_search'):
                beam_tokens = self.model.caption_image_beam_search(image_tensor, self.vocab, beam_size=5)
                beam_caption = ' '.join([token for token in beam_tokens 
                                        if token not in ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]])
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Display image
            plt.imshow(image)
            plt.axis('off')
            
            # Create caption text
            caption_text = ""
            caption_text += "References:\n"
            for j, ref in enumerate(reference_captions, 1):
                caption_text += f"{j}. {ref}\n"
            
            caption_text += f"\nGreedy: {greedy_caption}\n"
            caption_text += f"Beam: {beam_caption}"
            
            # Add caption text
            plt.figtext(0.5, 0.01, caption_text, wrap=True, horizontalalignment='center', fontsize=12)
            
            # Save figure
            output_path = os.path.join(self.visualization_dir, f"example_{i+1}.png")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5, dpi=150)
            plt.close()
            
        print(f"Visualization examples saved to {self.visualization_dir}/")
        
    def run_evaluation(
        self, 
        model_class: Type[BaseModel],
        visualize: bool = True, 
        num_examples: int = 10,
        **model_kwargs
    ) -> Dict[str, float]:
        """
        Run the full evaluation process
        
        Args:
            model_class: The model class to instantiate
            visualize: Whether to visualize examples
            num_examples: Number of examples to visualize
            model_kwargs: Additional keyword arguments for the model
            
        Returns:
            Dictionary of BLEU scores
        """
        # Load model
        self.load_model(model_class, **model_kwargs)
        
        # Load data
        self.load_data()
        
        # Evaluate BLEU score
        bleu_scores = self.evaluate_bleu()
        
        # Print results
        print("\nBLEU Score Evaluation:")
        for metric, score in bleu_scores.items():
            print(f"{metric}: {score:.2f}")
        
        # Save results to file
        with open('bleu_scores.json', 'w') as f:
            json.dump(bleu_scores, f, indent=4)
        
        # Visualize examples if requested
        if visualize:
            print("\nVisualizing examples...")
            self.visualize_examples(num_examples)
            print(f"Examples saved to '{self.visualization_dir}' directory")
            
        return bleu_scores

