import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, Any, Union

class BaseModel(nn.Module):
    """Base class for all models"""
    def __init__(self):
        self.has_mha_decoder = False
        super(BaseModel, self).__init__()
    
    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        raise NotImplementedError("Subclasses must implement forward method")
    
    @torch.no_grad()
    def caption_image_greedy(self, image: torch.Tensor, vocabulary: Any, max_length: int = 50) -> List[str]:
        """Generate a caption from the image using greedy search"""
        raise NotImplementedError("Subclasses must implement caption_image_greedy method")
    
    @torch.no_grad()
    def caption_image_beam_search(self, image: torch.Tensor, vocabulary: Any, beam_size: int = 3, max_length: int = 50) -> List[str]:
        """Generate a caption from the image using beam search"""
        raise NotImplementedError("Subclasses must implement caption_image_beam_search method")

