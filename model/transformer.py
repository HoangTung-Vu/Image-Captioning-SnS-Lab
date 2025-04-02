import torch
import torch.nn as nn
import math
from model.base_model import BaseModel
from typing import Tuple, List, Optional, Dict, Any, Union

# Helper function to extract patches
def extract_patches(image_tensor: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    Extracts patches from an image tensor.
    Args:
        image_tensor: Input image tensor [bs, c, h, w]
        patch_size: Size of the square patches.
    Returns:
        Patch tensor [bs, num_patches, patch_dim]
    """
    bs, c, h, w = image_tensor.size()
    # Calculate number of patches
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    num_patches = num_patches_h * num_patches_w
    patch_dim = c * patch_size * patch_size

    # Use unfold to extract patches
    patches = image_tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # Reshape to [bs, c, num_patches_h, num_patches_w, patch_size, patch_size]
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    # Reshape to [bs, num_patches, c, patch_size, patch_size]
    patches = patches.view(bs, num_patches, c, patch_size, patch_size)
    # Reshape to [bs, num_patches, patch_dim]
    patches = patches.view(bs, num_patches, patch_dim)
    return patches

# Sinusoidal positional embeddings
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generates sinusoidal positional embeddings.
        Args:
            seq_len: Length of the sequence.
            device: Device to create the tensor on.
        Returns:
            Positional embedding tensor [seq_len, 1, dim]
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # pos shape: [seq_len]
        pos = torch.arange(seq_len, device=device).float()
        # emb shape: [seq_len, half_dim]
        emb = pos[:, None] * emb[None, :]
        # SinCos shape: [seq_len, dim]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        # Return shape: [seq_len, 1, dim] for broadcasting with [seq_len, batch_size, dim]
        return emb.unsqueeze(1)


# Define a module for attention blocks
class AttentionBlock(nn.Module):
    def __init__(self, hidden_size: int = 128, num_heads: int = 4, is_causal: bool = False):
        """
        Attention Block using MultiheadAttention.
        Args:
            hidden_size: Dimension of the hidden state.
            num_heads: Number of attention heads.
            is_causal: If True, applies a causal mask for decoder self-attention.
        """
        super(AttentionBlock, self).__init__()
        self.is_causal = is_causal
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.1, # Added dropout
            batch_first=False # IMPORTANT: Set to False for [seq_len, batch_size, hidden_size] inputs
        )

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the attention block.
        Args:
            query: Query tensor [target_seq_len, batch_size, hidden_size]
            key: Key tensor [source_seq_len, batch_size, hidden_size]
            value: Value tensor [source_seq_len, batch_size, hidden_size]
            key_padding_mask: Mask for key padding [batch_size, source_seq_len]. True indicates padded positions.
        Returns:
            Attention output [target_seq_len, batch_size, hidden_size]
        """
        target_seq_len = query.size(0)
        attn_mask = None
        # Apply causal mask if needed (for decoder self-attention)
        if self.is_causal:
            # Create upper triangular mask
            attn_mask = torch.triu(torch.ones(target_seq_len, target_seq_len, device=query.device, dtype=torch.bool), diagonal=1)
            # MHA expects mask where True means "don't attend"

        # Perform multi-head attention
        # Input shapes: [seq_len, batch_size, embed_dim]
        # key_padding_mask shape: [batch_size, seq_len]
        # attn_mask shape: [target_seq_len, source_seq_len]
        attn_output, _ = self.mha(query, key, value,
                                  key_padding_mask=key_padding_mask,
                                  attn_mask=attn_mask,
                                  need_weights=False) # Don't need weights for standard forward pass
        return attn_output


# Define a transformer block
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int = 128, num_heads: int = 4, is_decoder: bool = False):
        """
        Transformer Block consisting of attention and feed-forward layers.
        Args:
            hidden_size: Dimension of the hidden state.
            num_heads: Number of attention heads.
            is_decoder: If True, includes cross-attention layer.
        """
        super(TransformerBlock, self).__init__()
        self.is_decoder = is_decoder

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        if self.is_decoder:
            self.norm3 = nn.LayerNorm(hidden_size)

        # Attention layers
        # Decoder self-attention needs causal masking
        self.self_attn = AttentionBlock(hidden_size, num_heads, is_causal=self.is_decoder)
        if self.is_decoder:
            self.cross_attn = AttentionBlock(hidden_size, num_heads, is_causal=False)

        # Feed-forward layer
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(), # Changed from ELU to GELU, common in Transformers
            nn.Dropout(0.1), # Added dropout
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.dropout = nn.Dropout(0.1) # Added dropout after attention and FFN

    def forward(self,
                x: torch.Tensor,
                encoder_output: Optional[torch.Tensor] = None,
                self_attn_padding_mask: Optional[torch.Tensor] = None,
                cross_attn_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the transformer block.
        Args:
            x: Input tensor [seq_len, batch_size, hidden_size]
            encoder_output: Output from the encoder (for cross-attention) [enc_seq_len, batch_size, hidden_size]
            self_attn_padding_mask: Padding mask for self-attention [batch_size, seq_len]
            cross_attn_padding_mask: Padding mask for cross-attention [batch_size, enc_seq_len]
        Returns:
            Output tensor [seq_len, batch_size, hidden_size]
        """
        # 1. Self-Attention
        residual = x
        x_norm = self.norm1(x)
        self_attn_output = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=self_attn_padding_mask)
        x = residual + self.dropout(self_attn_output)

        # 2. Cross-Attention (Decoder only)
        if self.is_decoder:
            if encoder_output is None:
                raise ValueError("Encoder output must be provided for decoder cross-attention")
            residual = x
            x_norm = self.norm2(x) # Use norm2 here
            cross_attn_output = self.cross_attn(x_norm, encoder_output, encoder_output, key_padding_mask=cross_attn_padding_mask)
            x = residual + self.dropout(cross_attn_output)
            # Use norm3 before FFN in decoder
            norm_layer_ffn = self.norm3
        else:
            # Use norm2 before FFN in encoder
            norm_layer_ffn = self.norm2

        # 3. Feed-Forward Network
        residual = x
        x_norm = norm_layer_ffn(x)
        ffn_output = self.ffn(x_norm)
        x = residual + self.dropout(ffn_output)

        return x


# Define a decoder module for the Transformer architecture
class Decoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 128, num_layers: int = 3, num_heads: int = 4):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # Initialize embedding weights (optional, but can help)
        # self.embedding.weight.data.uniform_(-0.1, 0.1)
        nn.init.normal_(self.embedding.weight, mean=0, std=hidden_size**-0.5)

        # Sinusoidal positional embeddings
        self.pos_emb = SinusoidalPosEmb(hidden_size)
        self.dropout = nn.Dropout(0.1) # Dropout after embedding + pos encoding

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, is_decoder=True) for _ in range(num_layers)
        ])

        # Final layer normalization and output linear layer
        self.norm_out = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self,
                target_seq: torch.Tensor,
                encoder_output: torch.Tensor,
                target_padding_mask: Optional[torch.Tensor] = None,
                encoder_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the Decoder.
        Args:
            target_seq: Target sequence tensor [seq_len, batch_size]
            encoder_output: Encoder output tensor [enc_seq_len, batch_size, hidden_size]
            target_padding_mask: Padding mask for target sequence [batch_size, seq_len]. True indicates padding.
            encoder_padding_mask: Padding mask for encoder output [batch_size, enc_seq_len]. True indicates padding.
        Returns:
            Output logits [seq_len, batch_size, vocab_size]
        """
        seq_len, batch_size = target_seq.shape
        device = target_seq.device

        # 1. Embeddings and Positional Encoding
        # target_emb shape: [seq_len, batch_size, hidden_size]
        target_emb = self.embedding(target_seq) * math.sqrt(self.hidden_size) # Scale embedding
        # pos_encoding shape: [seq_len, 1, hidden_size]
        pos_encoding = self.pos_emb(seq_len, device)
        # x shape: [seq_len, batch_size, hidden_size]
        x = self.dropout(target_emb + pos_encoding) # Add pos encoding and apply dropout

        # 2. Pass through Transformer Blocks
        for block in self.blocks:
            x = block(x,
                      encoder_output=encoder_output,
                      self_attn_padding_mask=target_padding_mask,
                      cross_attn_padding_mask=encoder_padding_mask)

        # 3. Final Normalization and Output Layer
        x = self.norm_out(x)
        # logits shape: [seq_len, batch_size, vocab_size]
        logits = self.fc_out(x)

        return logits


# Define a Vision Encoder module for the Transformer architecture
class VisionEncoder(nn.Module):
    def __init__(self, image_size: int, channels_in: int, patch_size: int = 16, hidden_size: int = 128,
                 num_layers: int = 3, num_heads: int = 4):
        super(VisionEncoder, self).__init__()
        if image_size % patch_size != 0:
             raise ValueError("Image size must be divisible by patch size")

        self.patch_size = patch_size
        self.hidden_size = hidden_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels_in * patch_size * patch_size

        # Linear projection for patches
        self.patch_proj = nn.Linear(patch_dim, hidden_size)

        # Learnable positional embedding (alternative to sinusoidal)
        self.pos_embedding = nn.Parameter(torch.randn(num_patches, 1, hidden_size)) # [num_patches, 1, hidden_size]
        self.dropout = nn.Dropout(0.1) # Dropout after embedding + pos encoding

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, is_decoder=False) for _ in range(num_layers)
        ])

        # Final layer normalization
        self.norm_out = nn.LayerNorm(hidden_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Vision Encoder.
        Args:
            image: Input image tensor [batch_size, channels, height, width]
        Returns:
            Encoded sequence [seq_len (num_patches), batch_size, hidden_size]
        """
        batch_size = image.shape[0]

        # 1. Extract and Project Patches
        # patches shape: [batch_size, num_patches, patch_dim]
        patches = extract_patches(image, patch_size=self.patch_size)
        # patch_emb shape: [batch_size, num_patches, hidden_size]
        patch_emb = self.patch_proj(patches)

        # 2. Add Positional Embedding
        # Transpose to [num_patches, batch_size, hidden_size] for batch_first=False
        x = patch_emb.transpose(0, 1)
        # Add positional embedding (broadcasts batch dim)
        x = x + self.pos_embedding
        x = self.dropout(x)

        # 3. Pass through Transformer Blocks
        # Note: Encoder doesn't use padding masks unless input images can vary in size (not typical here)
        for block in self.blocks:
            x = block(x)

        # 4. Final Normalization
        x = self.norm_out(x)

        # Output shape: [num_patches, batch_size, hidden_size]
        return x


# Define a Vision Encoder-Decoder module for the Transformer architecture
class VisionEncoderDecoder(BaseModel):
    def __init__(self, image_size: int, channels_in: int, vocab_size: int, patch_size: int = 16,
                 hidden_size: int = 128, num_layers: Tuple[int, int] = (1, 1), num_heads: int = 4):
        """
        Vision Encoder-Decoder Transformer Model.
        Args:
            image_size: Size of the input image (e.g., 224).
            channels_in: Number of input image channels (e.g., 3).
            vocab_size: Size of the output vocabulary.
            patch_size: Size of image patches (e.g., 16).
            hidden_size: Dimension of hidden layers (e.g., 128).
            num_layers: Tuple of (encoder_layers, decoder_layers).
            num_heads: Number of attention heads.
        """
        super(VisionEncoderDecoder, self).__init__()

        # Create encoder and decoder
        self.encoder = VisionEncoder(image_size=image_size, channels_in=channels_in,
                                     patch_size=patch_size, hidden_size=hidden_size,
                                     num_layers=num_layers[0], num_heads=num_heads)

        self.decoder = Decoder(vocab_size=vocab_size, hidden_size=hidden_size,
                               num_layers=num_layers[1], num_heads=num_heads)

        self.has_mha_decoder = True # Set base model flag

    def forward(self,
                input_image: torch.Tensor,
                target_seq: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the VisionEncoderDecoder.
        Args:
            input_image: Input image tensor [batch_size, channels, height, width]
            target_seq: Target sequence tensor [seq_len, batch_size] (excluding <EOS>)
            padding_mask: Padding mask for target sequence [batch_size, seq_len]. True indicates padding.
        Returns:
            Output logits [seq_len, batch_size, vocab_size]
        """
        # Encode the input image
        # encoded_seq shape: [num_patches, batch_size, hidden_size]
        encoded_seq = self.encoder(image=input_image)

        # Decode the target sequence using the encoded image features
        # Note: Encoder output doesn't typically have padding, so no encoder_padding_mask needed
        # decoded_seq shape: [seq_len, batch_size, vocab_size]
        decoded_seq = self.decoder(target_seq=target_seq,
                                   encoder_output=encoded_seq,
                                   target_padding_mask=padding_mask, # Pass the target padding mask
                                   encoder_padding_mask=None) # No padding mask for encoder output
        return decoded_seq

    @torch.no_grad()
    def caption_image_greedy(self, image: torch.Tensor, vocabulary: Any, max_length: int = 50) -> List[str]:
        """
        Generate a caption from the image using greedy search.
        Args:
            image: Input image tensor [1, channels, height, width]
            vocabulary: Vocabulary object (needs stoi, itos)
            max_length: Maximum length of the generated caption
        Returns:
            List of tokens (words) in the generated caption
        """
        assert image.shape[0] == 1, "Greedy search requires batch size 1"
        self.eval() # Ensure model is in eval mode
        device = image.device
        sos_idx = vocabulary.stoi["<SOS>"]
        eos_idx = vocabulary.stoi["<EOS>"]

        # Encode the image
        # encoder_output shape: [num_patches, 1, hidden_size]
        encoder_output = self.encoder(image)

        # Start with <SOS> token
        # current_tokens shape: [1, 1] (seq_len=1, batch=1)
        current_tokens = torch.tensor([[sos_idx]], dtype=torch.long, device=device)

        # Generate tokens one by one
        for _ in range(max_length - 1): # Max length includes <SOS>
            # Get predictions for the next token
            # No padding mask needed for greedy decoding (only one sequence)
            # output shape: [current_seq_len, 1, vocab_size]
            output = self.decoder(current_tokens, encoder_output, target_padding_mask=None, encoder_padding_mask=None)

            # Take the logits for the last token in the sequence
            # next_token_logits shape: [1, vocab_size]
            next_token_logits = output[-1, 0, :] # Last token, first batch item

            # Get the most likely token index (greedy search)
            # next_token shape: [1]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append the token to the sequence
            # current_tokens shape: [new_seq_len, 1]
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=0)

            # Stop if <EOS> token is generated
            if next_token.item() == eos_idx:
                break

        # Convert token indices to words
        caption_indices = current_tokens.squeeze(1).tolist() # Remove batch dim and convert to list
        caption = [vocabulary.itos[idx] for idx in caption_indices]

        # Remove <SOS> and <EOS> tokens for the final output
        return [word for word in caption if word not in ["<SOS>", "<EOS>"]]


    @torch.no_grad()
    def caption_image_beam_search(self, image: torch.Tensor, vocabulary: Any, beam_size: int = 3, max_length: int = 50) -> List[str]:
        """
        Generate a caption from the image using beam search.
        (Note: This is a simplified implementation. More advanced versions might handle batching within the beam.)
        Args:
            image: Input image tensor [1, channels, height, width]
            vocabulary: Vocabulary object (needs stoi, itos)
            beam_size: Number of beams.
            max_length: Maximum length of the generated caption.
        Returns:
            List of tokens (words) in the best generated caption.
        """
        assert image.shape[0] == 1, "Beam search implementation requires batch size 1"
        self.eval() # Ensure model is in eval mode
        device = image.device
        sos_idx = vocabulary.stoi["<SOS>"]
        eos_idx = vocabulary.stoi["<EOS>"]

        # Encode the image
        # encoder_output shape: [num_patches, 1, hidden_size]
        encoder_output = self.encoder(image)

        # Initialize beams: list of tuples (sequence_tensor [seq_len, 1], score)
        initial_beam = (torch.tensor([[sos_idx]], dtype=torch.long, device=device), 0.0)
        beams = [initial_beam]
        completed_beams = []

        # Generate step-by-step
        for _ in range(max_length - 1):
            if not beams: # Stop if no active beams left
                break

            new_beams = []
            for current_tokens, current_score in beams:
                # If last token is <EOS>, this beam is complete
                if current_tokens[-1, 0].item() == eos_idx:
                    completed_beams.append((current_tokens, current_score))
                    continue

                # Get predictions for the next token
                # output shape: [current_seq_len, 1, vocab_size]
                output = self.decoder(current_tokens, encoder_output, target_padding_mask=None, encoder_padding_mask=None)
                # next_token_logits shape: [vocab_size]
                next_token_logits = output[-1, 0, :]
                # Convert to log probabilities
                log_probs = torch.log_softmax(next_token_logits, dim=-1)

                # Get top-k next tokens and their log probabilities
                top_log_probs, top_indices = torch.topk(log_probs, beam_size)

                # Create new beams by extending the current beam
                for j in range(beam_size):
                    next_token_idx = top_indices[j].item()
                    log_prob = top_log_probs[j].item()

                    # Create new sequence tensor
                    # next_token_tensor shape: [1, 1]
                    next_token_tensor = torch.tensor([[next_token_idx]], dtype=torch.long, device=device)
                    # new_tokens shape: [new_seq_len, 1]
                    new_tokens = torch.cat([current_tokens, next_token_tensor], dim=0)
                    new_score = current_score + log_prob # Add log probabilities

                    new_beams.append((new_tokens, new_score))

            # Combine new beams with potentially completed beams from previous steps
            # Sort all potential beams (ongoing and newly created) by score
            all_potential_beams = new_beams # In this step, only consider newly generated beams
            beams = sorted(all_potential_beams, key=lambda x: x[1], reverse=True)[:beam_size] # Keep top `beam_size`

            # Optimization: If beam_size is large, pruning completed beams earlier can save computation
            # completed_beams = sorted(completed_beams, key=lambda x: x[1], reverse=True)
            # beams = beams[:beam_size - len(completed_beams)] # Reduce active beams if many completed

        # After the loop, add any remaining active beams to the completed list
        completed_beams.extend(beams)

        # Find the best beam among all completed beams
        if not completed_beams:
             return ["<UNK>"] # Should not happen if max_length > 0

        best_beam = sorted(completed_beams, key=lambda x: x[1], reverse=True)[0]
        best_sequence_tensor, _ = best_beam

        # Convert token indices to words
        caption_indices = best_sequence_tensor.squeeze(1).tolist()
        caption = [vocabulary.itos[idx] for idx in caption_indices]

        # Remove <SOS> and <EOS>
        return [word for word in caption if word not in ["<SOS>", "<EOS>"]]