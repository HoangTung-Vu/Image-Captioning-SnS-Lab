from PIL import Image
import os
import pandas as pd
import spacy
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from functools import lru_cache

# Load spaCy English model
try:
    spacy_eng = spacy.load("en_core_web_sm")
except OSError:
    # If model not found, download it
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, freq_threshold: int = 1):
        """
        Initialize vocabulary with special tokens
        
        Args:
            freq_threshold: Minimum frequency for a word to be included in vocabulary
        """
        # Special tokens
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}  # index to string 
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}  # string to index
        self.freq_threshold = freq_threshold
    
    def __len__(self) -> int:
        """Return vocabulary size"""
        return len(self.itos)
    
    @staticmethod
    @lru_cache(maxsize=10000)  # Cache tokenization results for efficiency
    def tokenize(text: str) -> List[str]:
        """
        Tokenize text using spaCy
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list: List[str]) -> None:
        """
        Build vocabulary from list of sentences
        
        Args:
            sentence_list: List of sentences
        """
        frequencies: Dict[str, int] = {}
        idx = 4  # Start after special tokens

        # Count word frequencies
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                # Add word to vocabulary if it reaches threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text: str) -> List[int]:
        """
        Convert text to list of indices
        
        Args:
            text: Input text
            
        Returns:
            List of token indices
        """
        tokenized_text = self.tokenize(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] 
            for token in tokenized_text
        ]

class FlickrDataset(Dataset):
    def __init__(
        self, 
        root: str = "data/flickr8k/Flicker8k_Dataset", 
        captions_file: str = "data/flickr8k/captions.txt", 
        transform: Optional[Callable] = None, 
        freq_threshold: int = 1,
        img_cache_size: int = 100  # Number of images to cache in memory
    ):
        """
        Initialize Flickr dataset
        
        Args:
            root: Path to image directory
            captions_file: Path to captions file
            transform: Image transformations
            freq_threshold: Minimum frequency for a word to be included in vocabulary
            img_cache_size: Number of images to cache in memory
        """
        self.root = root
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        self.img_cache_size = img_cache_size

        # Get img, caption columns
        self.imgs = self.df['image']
        self.captions = self.df['caption']

        # Initialize vocabulary and build
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())
        
        # Image cache
        self.img_cache: Dict[str, Image.Image] = {}

    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.df)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index
        
        Args:
            index: Item index
            
        Returns:
            Tuple of (image, caption)
        """
        caption = self.captions[index]
        img_id = self.imgs[index]
        
        # Load image (with caching)
        if img_id in self.img_cache:
            img = self.img_cache[img_id]
        else:
            img = Image.open(os.path.join(self.root, img_id)).convert("RGB")
            # Update cache
            if len(self.img_cache) >= self.img_cache_size:
                # Remove a random item if cache is full
                self.img_cache.pop(next(iter(self.img_cache)))
            self.img_cache[img_id] = img

        if self.transform is not None:
            img = self.transform(img)
        
        # Prepare caption
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)

class Collate:
    def __init__(self, pad_idx: int):
        """
        Initialize collate function
        
        Args:
            pad_idx: Padding token index
        """
        self.pad_idx = pad_idx
    
    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate function for DataLoader
        
        Args:
            batch: List of (image, caption) tuples
            
        Returns:
            Tuple of batched images and padded captions
        """
        # Stack images
        imgs = torch.stack([item[0] for item in batch], dim=0)
        
        # Pad captions
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets

def get_loader(
    root_folder: str, 
    annotation_file: str, 
    transform: Optional[Callable], 
    batch_size: int = 32, 
    num_workers: int = 4, 
    shuffle: bool = True, 
    pin_memory: bool = True
) -> Tuple[DataLoader, FlickrDataset]:
    """
    Create data loader for Flickr dataset
    
    Args:
        root_folder: Path to image directory
        annotation_file: Path to captions file
        transform: Image transformations
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (data_loader, dataset)
    """
    # Create dataset
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)
    
    # Get padding index
    pad_idx = dataset.vocab.stoi["<PAD>"]
    
    # Create data loader
    loader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=shuffle, 
        pin_memory=pin_memory, 
        collate_fn=Collate(pad_idx=pad_idx)
    )

    return loader, dataset

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    loader, dataset = get_loader("data/flickr8k/Flicker8k_Dataset", "data/flickr8k/captions.txt", transform=transform)
    for idx, (imgs, captions) in enumerate(loader):
        print(imgs.shape)
        print(captions.shape)
        break