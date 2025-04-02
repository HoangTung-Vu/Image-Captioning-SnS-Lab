import torch
import torch.nn as nn
import math
from model.base_model import BaseModel
from typing import Tuple, List, Optional, Dict, Any, Union


def extract_patches(image_tensor, patch_size=16):
    # Get the dimensions of the image tensor
    bs, c, h, w = image_tensor.size()
    
    # Define the Unfold layer with appropriate parameters
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    
    # Apply Unfold to the image tensor
    unfolded = unfold(image_tensor)
    
    # Reshape the unfolded tensor to match the desired output shape
    # Output shape: BSxLxH, where L is the number of patches in each dimension
    unfolded = unfolded.transpose(1, 2).reshape(bs, -1, c * patch_size * patch_size)
    
    return unfolded

# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

# Define a module for attention blocks
class AttentionBlock(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4, masking=True):
        super(AttentionBlock, self).__init__()
        self.masking = masking

        # Multi-head attention mechanism
        self.multihead_attn = nn.MultiheadAttention(hidden_size,
                                                    num_heads=num_heads,
                                                    batch_first=True,
                                                    dropout=0.0)

    def forward(self, x_in, kv_in, key_mask=None):
        # Apply causal masking if enabled
        if self.masking:
            bs, l, h = x_in.shape
            mask = torch.triu(torch.ones(l, l, device=x_in.device), 1).bool()
        else:
            mask = None
            
        # Perform multi-head attention operation
        return self.multihead_attn(x_in, kv_in, kv_in, attn_mask=mask, 
                                   key_padding_mask=key_mask)[0]


# Define a module for a transformer block with self-attention 
# and optional causal masking
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4, decoder=False, masking=True):
        super(TransformerBlock, self).__init__()
        self.decoder = decoder

        # Layer normalization for the input
        self.norm1 = nn.LayerNorm(hidden_size)
        # Self-attention mechanism
        self.attn1 = AttentionBlock(hidden_size=hidden_size, num_heads=num_heads, 
                                    masking=masking)
        
        # Layer normalization for the output of the first attention layer
        if self.decoder:
            self.norm2 = nn.LayerNorm(hidden_size)
            # Self-attention mechanism for the decoder with no masking
            self.attn2 = AttentionBlock(hidden_size=hidden_size, 
                                        num_heads=num_heads, masking=False)
        
        # Layer normalization for the output before the MLP
        self.norm_mlp = nn.LayerNorm(hidden_size)
        # Multi-layer perceptron (MLP)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4),
                                 nn.ELU(),
                                 nn.Linear(hidden_size * 4, hidden_size))
                
    def forward(self, x, input_key_mask=None, cross_key_mask=None, kv_cross=None):
        # Perform self-attention operation
        x = self.attn1(x, x, key_mask=input_key_mask) + x
        x = self.norm1(x)

        # If decoder, perform additional cross-attention layer
        if self.decoder:
            x = self.attn2(x, kv_cross, key_mask=cross_key_mask) + x
            x = self.norm2(x)

        # Apply MLP and layer normalization
        x = self.mlp(x) + x
        return self.norm_mlp(x)

    
# Define a decoder module for the Transformer architecture
class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_layers=3, num_heads=4):
        super(Decoder, self).__init__()
        
        # Create an embedding layer for tokens
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # Initialize the embedding weights
        self.embedding.weight.data = 0.001 * self.embedding.weight.data

        # Initialize sinusoidal positional embeddings
        self.pos_emb = SinusoidalPosEmb(hidden_size)
        
        # Create multiple transformer blocks as layers
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, 
                             decoder=True) for _ in range(num_layers)
        ])
                
        # Define a linear layer for output prediction
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_seq, encoder_output, input_padding_mask=None, 
                encoder_padding_mask=None):        
        # Embed the input sequence
        input_embs = self.embedding(input_seq)
        bs, l, h = input_embs.shape

        # Add positional embeddings to the input embeddings
        seq_indx = torch.arange(l, device=input_seq.device)
        pos_emb = self.pos_emb(seq_indx).reshape(1, l, h).expand(bs, l, h)
        embs = input_embs + pos_emb
        
        # Pass the embeddings through each transformer block
        for block in self.blocks:
            embs = block(embs, 
                           input_key_mask=input_padding_mask, 
                           cross_key_mask=encoder_padding_mask, 
                           kv_cross=encoder_output)
        
        return self.fc_out(embs)

    
# Define an Vision Encoder module for the Transformer architecture
class VisionEncoder(nn.Module):
    def __init__(self, image_size, channels_in, patch_size=16, hidden_size=128, 
                 num_layers=3, num_heads=4):
        super(VisionEncoder, self).__init__()
        
        self.patch_size = patch_size
        self.fc_in = nn.Linear(channels_in * patch_size * patch_size, hidden_size)
        
        seq_length = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, 
                                                      hidden_size).normal_(std=0.02))
        
        # Create multiple transformer blocks as layers
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, 
                             decoder=False, masking=False) for _ in range(num_layers)
        ])
                
    def forward(self, image):  
        bs = image.shape[0]

        patch_seq = extract_patches(image, patch_size=self.patch_size)
        patch_emb = self.fc_in(patch_seq)

        # Add a unique embedding to each token embedding
        embs = patch_emb + self.pos_embedding
        
        # Pass the embeddings through each transformer block
        for block in self.blocks:
            embs = block(embs)
        
        return embs
    
    
# Define an Vision Encoder-Decoder module for the Transformer architecture
class VisionEncoderDecoder(BaseModel):
    def __init__(self, image_size, channels_in, vocab_size, patch_size=16, 
                 hidden_size=128, num_layers=(3, 3), num_heads=4):
        super(VisionEncoderDecoder, self).__init__()
        
        # Create an encoder and decoder with specified parameters
        self.encoder = VisionEncoder(image_size=image_size, channels_in=channels_in, 
                                     patch_size=patch_size, hidden_size=hidden_size, 
                                     num_layers=num_layers[0], num_heads=num_heads)
        
        self.decoder = Decoder(vocab_size=vocab_size, hidden_size=hidden_size, 
                               num_layers=num_layers[1], num_heads=num_heads)
        
        self.has_mha_decoder = True
    def forward(self, input_image, target_seq, padding_mask):
        # Generate padding masks for the target sequence
        bool_padding_mask = padding_mask == 0

        # Encode the input sequence
        encoded_seq = self.encoder(image=input_image)
        
        # Decode the target sequence using the encoded sequence
        decoded_seq = self.decoder(input_seq=target_seq, 
                                   encoder_output=encoded_seq, 
                                   input_padding_mask=bool_padding_mask)
        return decoded_seq
    
    @torch.no_grad()
    def caption_image_greedy(self, image: torch.Tensor, vocabulary: Any, max_length: int = 50) -> List[str]:
        """Generate a caption from the image using greedy search"""
        # Encode the image
        encoded_image = self.encoder(image)
        
        # Start with a batch of start tokens
        batch_size = image.size(0)
        current_tokens = torch.ones(batch_size, 1, dtype=torch.long, device=image.device) * vocabulary.stoi["<start>"]
        
        # Generate tokens one by one
        for _ in range(max_length):
            # Get predictions for the next token
            predictions = self.decoder(current_tokens, encoded_image)
            # Take the last prediction (for the next token)
            predictions = predictions[:, -1, :]
            # Get the most likely token (greedy search)
            next_token = torch.argmax(predictions, dim=-1, keepdim=True)
            # Add the token to our sequence
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
            
            # Stop if all sequences have generated an end token
            if (next_token == vocabulary.stoi["<end>"]).all():
                break
        
        # Convert token indices to words
        captions = []
        for tokens in current_tokens:
            caption = []
            for token in tokens:
                word = vocabulary.itos[token.item()]
                if word == "<start>":
                    continue
                if word == "<end>":
                    break
                caption.append(word)
            captions.append(" ".join(caption))
        
        return captions
    
    @torch.no_grad()
    def caption_image_beam_search(self, image: torch.Tensor, vocabulary: Any, beam_size: int = 3, max_length: int = 50) -> List[str]:
        """Generate a caption from the image using beam search"""
        # Encode the image
        encoded_image = self.encoder(image)
        
        # Start with a batch of start tokens
        batch_size = image.size(0)
        start_token = vocabulary.stoi["<start>"]
        end_token = vocabulary.stoi["<end>"]
        
        # Initialize beams for each image in the batch
        all_captions = []
        
        for i in range(batch_size):
            # Get the encoded features for this specific image
            img_features = encoded_image[i:i+1]
            
            # Initialize beam with start token
            beams = [([start_token], 0.0)]  # (sequence, score)
            complete_beams = []
            
            # Generate tokens step by step
            for _ in range(max_length):
                if len(beams) == 0:
                    break
                    
                new_beams = []
                # Expand each current beam
                for seq, score in beams:
                    # Skip if the sequence is already complete
                    if seq[-1] == end_token:
                        complete_beams.append((seq, score))
                        continue
                    
                    # Convert sequence to tensor
                    current_tokens = torch.tensor([seq], dtype=torch.long, device=image.device)
                    
                    # Get predictions
                    predictions = self.decoder(current_tokens, img_features)
                    # Get the last prediction
                    predictions = predictions[0, -1, :]
                    # Apply softmax to get probabilities
                    probs = torch.nn.functional.softmax(predictions, dim=-1)
                    
                    # Get top-k next tokens
                    topk_probs, topk_indices = torch.topk(probs, beam_size)
                    
                    # Add new beams
                    for j in range(beam_size):
                        token = topk_indices[j].item()
                        prob = topk_probs[j].item()
                        new_score = score - math.log(prob)  # Using negative log likelihood
                        new_beams.append((seq + [token], new_score))
                
                # Keep only the top beam_size beams
                beams = sorted(new_beams, key=lambda x: x[1])[:beam_size]
                
                # Check if all beams end with end_token
                if all(beam[0][-1] == end_token for beam in beams):
                    complete_beams.extend(beams)
                    break
            
            # Add any incomplete beams to complete_beams
            complete_beams.extend(beams)
            
            # Sort by score and take the best one
            best_beam = sorted(complete_beams, key=lambda x: x[1])[0][0]
            
            # Convert token indices to words
            caption = []
            for token in best_beam:
                word = vocabulary.itos[token]
                if word == "<start>":
                    continue
                if word == "<end>":
                    break
                caption.append(word)
            
            all_captions.append(" ".join(caption))
        
        return all_captions