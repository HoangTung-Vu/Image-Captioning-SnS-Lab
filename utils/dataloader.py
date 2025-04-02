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
from collections import OrderedDict # For LRU cache behavior

# Load spaCy English model
try:
    spacy_eng = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy en_core_web_sm model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, freq_threshold: int = 1):
        """
        Initialize vocabulary with special tokens.

        Args:
            freq_threshold: Minimum frequency for a word to be included in vocabulary.
        """
        # Special tokens
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}  # index to string
        self.stoi = {v: k for k, v in self.itos.items()} # string to index (inverted)
        self.freq_threshold = freq_threshold

    def __len__(self) -> int:
        """Return vocabulary size"""
        return len(self.itos)

    @staticmethod
    # Using Spacy's tokenizer directly is usually fast enough, caching might not be necessary
    # @lru_cache(maxsize=10000) # Cache tokenization results for efficiency
    def tokenize(text: str) -> List[str]:
        """
        Tokenize text using spaCy.

        Args:
            text: Input text string.

        Returns:
            List of tokens (lowercased).
        """
        return [token.text.lower() for token in spacy_eng.tokenizer(str(text))] # Ensure text is string

    def build_vocabulary(self, sentence_list: List[str]) -> None:
        """
        Build vocabulary from list of sentences.

        Args:
            sentence_list: List of sentences.
        """
        frequencies: Dict[str, int] = {}
        idx = len(self.itos) # Start index after special tokens

        print("Building vocabulary...")
        # Count word frequencies
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] = frequencies.get(word, 0) + 1

        # Add words to vocabulary if they meet frequency threshold
        # Sort frequencies to make vocabulary building deterministic (optional)
        # sorted_freq = sorted(frequencies.items(), key=lambda item: item[1], reverse=True)
        for word, freq in frequencies.items():
             if freq >= self.freq_threshold:
                 if word not in self.stoi: # Avoid adding duplicates if already exists
                     self.stoi[word] = idx
                     self.itos[idx] = word
                     idx += 1
        print(f"Built vocabulary with {len(self.stoi)} words.")

    def numericalize(self, text: str) -> List[int]:
        """
        Convert text to list of token indices.

        Args:
            text: Input text string.

        Returns:
            List of token indices, using <UNK> for unknown words.
        """
        tokenized_text = self.tokenize(text)
        unk_idx = self.stoi["<UNK>"]
        return [self.stoi.get(token, unk_idx) for token in tokenized_text]


class FlickrDataset(Dataset):
    def __init__(
        self,
        root: str = "data/flickr8k/Flicker8k_Dataset",
        captions_file: str = "data/flickr8k/captions.txt",
        transform: Optional[Callable] = None,
        freq_threshold: int = 1,
        img_cache_size: Optional[int] = 100 # Number of images to cache in memory, None to disable
    ):
        """
        Initialize Flickr dataset.

        Args:
            root: Path to image directory.
            captions_file: Path to captions file (CSV format expected: 'image', 'caption').
            transform: Image transformations.
            freq_threshold: Minimum frequency for a word to be included in vocabulary.
            img_cache_size: Max number of images to cache in memory (use OrderedDict for LRU). None disables cache.
        """
        self.root = root
        try:
            self.df = pd.read_csv(captions_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Captions file not found at {captions_file}")

        self.transform = transform
        self.img_cache_size = img_cache_size

        # Get img, caption columns
        if 'image' not in self.df.columns or 'caption' not in self.df.columns:
             raise ValueError("Captions file must contain 'image' and 'caption' columns.")
        self.imgs = self.df['image']
        self.captions = self.df['caption']

        # Initialize vocabulary and build
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

        # Image cache (using OrderedDict for LRU behavior if size is limited)
        if self.img_cache_size is not None and self.img_cache_size > 0:
            self.img_cache: Optional[OrderedDict[str, Image.Image]] = OrderedDict()
            print(f"Image cache enabled with size {self.img_cache_size}")
        else:
            self.img_cache = None
            print("Image cache disabled.")


    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.

        Args:
            index: Item index.

        Returns:
            Tuple of (image_tensor, caption_tensor)
        """
        caption = self.captions[index]
        img_id = self.imgs[index]
        img_path = os.path.join(self.root, img_id)

        # Load image (with caching)
        img = None
        if self.img_cache is not None:
            if img_id in self.img_cache:
                # Move accessed item to end for LRU
                self.img_cache.move_to_end(img_id)
                img = self.img_cache[img_id]
            else:
                try:
                    img = Image.open(img_path).convert("RGB")
                    # Update cache
                    if len(self.img_cache) >= self.img_cache_size:
                        # Remove the least recently used item (first item)
                        self.img_cache.popitem(last=False)
                    self.img_cache[img_id] = img
                except FileNotFoundError:
                     print(f"Warning: Image file not found at {img_path}. Skipping item {index}.")
                     # Return dummy data or handle appropriately
                     # For simplicity, we might just raise an error or return None and filter in collate_fn
                     # Returning the previous item might be problematic. Let's return dummy tensors.
                     dummy_img = torch.zeros((3, 224, 224)) # Example size
                     dummy_caption = torch.tensor([self.vocab.stoi["<SOS>"], self.vocab.stoi["<EOS>"]])
                     return dummy_img, dummy_caption
                except Exception as e:
                     print(f"Warning: Error loading image {img_path}: {e}. Skipping item {index}.")
                     dummy_img = torch.zeros((3, 224, 224))
                     dummy_caption = torch.tensor([self.vocab.stoi["<SOS>"], self.vocab.stoi["<EOS>"]])
                     return dummy_img, dummy_caption

        else: # No cache
             try:
                 img = Image.open(img_path).convert("RGB")
             except FileNotFoundError:
                 print(f"Warning: Image file not found at {img_path}. Skipping item {index}.")
                 dummy_img = torch.zeros((3, 224, 224))
                 dummy_caption = torch.tensor([self.vocab.stoi["<SOS>"], self.vocab.stoi["<EOS>"]])
                 return dummy_img, dummy_caption
             except Exception as e:
                 print(f"Warning: Error loading image {img_path}: {e}. Skipping item {index}.")
                 dummy_img = torch.zeros((3, 224, 224))
                 dummy_caption = torch.tensor([self.vocab.stoi["<SOS>"], self.vocab.stoi["<EOS>"]])
                 return dummy_img, dummy_caption


        # Apply transformations if they exist
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            # Default transform to tensor if none provided
            img_tensor = transforms.ToTensor()(img)

        # Prepare caption: <SOS> + numericalized_caption + <EOS>
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img_tensor, torch.tensor(numericalized_caption)


class Collate:
    def __init__(self, pad_idx: int):
        """
        Initialize collate function for padding.

        Args:
            pad_idx: Index of the padding token in the vocabulary.
        """
        self.pad_idx = pad_idx

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate function for DataLoader. Handles padding.

        Args:
            batch: A list of (image_tensor, caption_tensor) tuples.

        Returns:
            Tuple of batched images and padded captions (batch_first=False).
        """
        # Filter out None items if any occurred during __getitem__ error handling
        # batch = [item for item in batch if item is not None]
        # If returning dummy tensors instead of None:
        # No filtering needed here if dummy tensors are returned on error.

        # Separate images and captions
        imgs = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        # Stack images along the batch dimension
        # imgs shape: [batch_size, channels, height, width]
        imgs_tensor = torch.stack(imgs, dim=0)

        # Pad captions sequences to the maximum length in the batch
        # targets shape: [max_seq_len, batch_size]
        targets_tensor = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs_tensor, targets_tensor


def get_loader(
    root_folder: str,
    annotation_file: str,
    transform: Optional[Callable],
    batch_size: int = 32,
    num_workers: int = 0, # Default to 0 for easier debugging, increase later
    shuffle: bool = True,
    pin_memory: bool = False, # Often set based on CUDA availability
    freq_threshold: int = 1, # Vocab frequency threshold
    img_cache_size: Optional[int] = 100 # Image cache size
) -> Tuple[DataLoader, FlickrDataset]:
    """
    Create data loader for Flickr dataset.

    Args:
        root_folder: Path to image directory.
        annotation_file: Path to captions file.
        transform: Image transformations.
        batch_size: Batch size for the DataLoader.
        num_workers: Number of worker processes for data loading.
        shuffle: Whether to shuffle data each epoch.
        pin_memory: Whether to use pinned memory for faster CPU-GPU transfer.
        freq_threshold: Minimum word frequency for vocabulary.
        img_cache_size: Size of the image cache.

    Returns:
        Tuple of (DataLoader, FlickrDataset instance).
    """
    # Create dataset instance
    dataset = FlickrDataset(
        root=root_folder,
        captions_file=annotation_file,
        transform=transform,
        freq_threshold=freq_threshold,
        img_cache_size=img_cache_size
    )

    # Get padding index from the dataset's vocabulary
    pad_idx = dataset.vocab.stoi["<PAD>"]

    # Determine pin_memory based on CUDA availability if not explicitly set
    pin_memory = pin_memory and torch.cuda.is_available()

    # Create data loader instance
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=Collate(pad_idx=pad_idx) # Use custom collate function
    )

    print(f"DataLoader created with batch size {batch_size}, workers {num_workers}, shuffle {shuffle}, pin_memory {pin_memory}")
    return loader, dataset

# Example usage:
if __name__ == "__main__":
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # Imagenet normalization
    ])

    # Create DataLoader
    try:
        loader, dataset = get_loader(
            root_folder="data/flickr8k/Images", # Corrected typical path
            annotation_file="data/flickr8k/captions.txt",
            transform=transform,
            batch_size=4, # Small batch size for testing
            num_workers=0,
            shuffle=True,
            img_cache_size=10 # Small cache for testing
        )

        print(f"Vocabulary size: {len(dataset.vocab)}")

        # Iterate over one batch
        print("Loading one batch...")
        for idx, (imgs, captions) in enumerate(loader):
            print(f"Batch {idx+1}:")
            print(f"  Images shape: {imgs.shape}")    # Expected: [batch_size, 3, 224, 224]
            print(f"  Captions shape: {captions.shape}") # Expected: [max_seq_len, batch_size]
            print(f"  Example caption indices (first item): {captions[:, 0].tolist()}")
            # Decode example caption
            example_caption = [dataset.vocab.itos[i] for i in captions[:, 0].tolist() if i != dataset.vocab.stoi["<PAD>"]]
            print(f"  Decoded example caption: {' '.join(example_caption)}")
            break # Only process one batch for demonstration
        print("Example load complete.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the dataset path ('data/flickr8k/Images') and captions file path ('data/flickr8k/captions.txt') are correct.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")