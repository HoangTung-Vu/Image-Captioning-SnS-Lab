import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from typing import Tuple, List, Optional, Dict, Any, Union
from model.base_model import BaseModel
import math # Added for beam search log prob

class Encoder(nn.Module):
    def __init__(self, embed_size: int, train_CNN: bool = False, dropout_rate: float = 0.5):
        """
        Load the pretrained ResNet-18 and replace top fc layer.

        Args:
            embed_size: Size of the embedding vector
            train_CNN: Whether to train the CNN layers (beyond the final fc layer)
            dropout_rate: Dropout rate for regularization
        """
        super(Encoder, self).__init__()
        self.train_CNN = train_CNN

        # Load pretrained ResNet-18
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Replace the final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, embed_size)

        # Add normalization and regularization
        self.relu = nn.ReLU()
        # Using BatchNorm1d as the output of ResNet fc is [batch_size, embed_size]
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.dropout = nn.Dropout(dropout_rate)

        # Freeze all layers initially except the new fc layer
        for name, param in self.resnet.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_CNN # Controlled by train_CNN flag

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder

        Args:
            images: Batch of images [batch_size, channels, height, width]

        Returns:
            Features: Encoded image features [batch_size, embed_size]
        """
        # Update requires_grad based on current train_CNN state (if changed during training)
        for name, param in self.resnet.named_parameters():
            if "fc.weight" not in name and "fc.bias" not in name:
                param.requires_grad = self.train_CNN

        # Extract features from ResNet
        features = self.resnet(images) # [batch_size, embed_size]

        # Apply normalization and regularization
        # Note: ReLU is often applied *before* BatchNorm, but applying it after fc
        # and before BN is also common. Let's stick to the original code's order.
        # However, typically BN comes before ReLU. Consider swapping if results are poor.
        # features = self.relu(self.bn(features)) # Alternative order
        features = self.bn(self.relu(features)) # Original order

        return self.dropout(features)


class Decoder(nn.Module):
    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        vocab_size: int,
        num_layers: int,
        dropout_rate: float = 0.5
    ):
        """
        Set the hyper-parameters and build the layers.

        Args:
            embed_size: Size of the embedding vector
            hidden_size: Size of the LSTM hidden state
            vocab_size: Size of the vocabulary
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate for regularization
        """
        super(Decoder, self).__init__()

        # Embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)

        # LSTM layer
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=False,  # Expects input: [seq_length, batch_size, embed_size]
            dropout=dropout_rate if num_layers > 1 else 0
        )

        # Output layer
        self.linear = nn.Linear(hidden_size, vocab_size)

        # Dropout for regularization on embeddings
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        features: torch.Tensor,
        captions: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode image feature vectors and generates captions.

        Args:
            features: Image features from encoder [batch_size, embed_size]
            captions: Target captions [seq_length, batch_size] (excluding <EOS>)

        Returns:
            outputs: Predicted word probabilities [seq_length, batch_size, vocab_size]
        """
        # Embed captions
        # captions shape: [seq_length, batch_size]
        embeddings = self.dropout(self.embed(captions))
        # embeddings shape: [seq_length, batch_size, embed_size]

        # Prepare features to be the first input to the LSTM
        # features shape: [batch_size, embed_size] -> [1, batch_size, embed_size]
        features = features.unsqueeze(0) # Add sequence dimension

        # Concatenate features with embeddings
        # inputs shape: [1 + seq_length, batch_size, embed_size]
        inputs = torch.cat((features, embeddings), dim=0)

        # Pass through LSTM
        # hiddens shape: [1 + seq_length, batch_size, hidden_size]
        hiddens, _ = self.lstm(inputs)

        # Pass through linear layer
        # outputs shape: [1 + seq_length, batch_size, vocab_size]
        outputs = self.linear(hiddens)

        # We only need predictions for the caption sequence, not the initial feature step
        # Return predictions corresponding to the input captions
        # outputs shape: [seq_length, batch_size, vocab_size]
        return outputs[1:] # Shifted output corresponding to input captions

class CNNtoRNN(BaseModel):
    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        vocab_size: int,
        num_layers: int,
        trainCNN: bool = False,
        dropout_rate: float = 0.5
    ):
        """
        Initialize the encoder-decoder model.

        Args:
            embed_size: Size of the embedding vector
            hidden_size: Size of the LSTM hidden state
            vocab_size: Size of the vocabulary
            num_layers: Number of LSTM layers
            trainCNN: Whether to train the CNN initially
            dropout_rate: Dropout rate for regularization
        """
        super(CNNtoRNN, self).__init__()

        # Initialize encoder and decoder
        self.encoder = Encoder(embed_size, trainCNN, dropout_rate)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers, dropout_rate)
        self.has_mha_decoder = False # Set base model flag

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None # Added for compatibility, but not used
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            images: Batch of images [batch_size, channels, height, width]
            captions: Target captions [seq_length, batch_size] (excluding <EOS>)
            padding_mask: Ignored for this model.

        Returns:
            outputs: Predicted word probabilities [seq_length, batch_size, vocab_size]
        """
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    @torch.no_grad()
    def caption_image_greedy(
        self,
        image: torch.Tensor,
        vocabulary: Any,
        max_length: int = 50
    ) -> List[str]:
        """
        Generate a caption from the image using greedy search.

        Args:
            image: Input image [1, channels, height, width]
            vocabulary: Vocabulary object (needs stoi, itos)
            max_length: Maximum caption length

        Returns:
            result_caption: List of words in the generated caption
        """
        assert image.shape[0] == 1, "Greedy search requires batch size 1"
        self.eval() # Ensure model is in eval mode
        result_caption_indices = []
        states = None # Initial hidden/cell states

        # Encode image
        # features shape: [1, embed_size]
        features = self.encoder(image)
        # Input to LSTM needs shape [1, 1, embed_size] (seq_len=1, batch=1)
        inputs = features.unsqueeze(0)

        # Start generating caption word by word
        for _ in range(max_length):
            # hiddens: [1, 1, hidden_size], states: tuple of (h, c) each [num_layers, 1, hidden_size]
            hiddens, states = self.decoder.lstm(inputs, states)
            # output: [1, 1, vocab_size]
            output = self.decoder.linear(hiddens) # No need to squeeze if batch_first=False
            # predicted_index: [1, 1] -> scalar
            predicted_index = output.argmax(2).item() # Get index of max logit

            # Append predicted index
            result_caption_indices.append(predicted_index)

            # If <EOS> token is predicted, stop generation
            if vocabulary.itos[predicted_index] == "<EOS>":
                break

            # Prepare the predicted token as the next input
            # inputs: [1, 1, embed_size]
            inputs = self.decoder.embed(torch.tensor([[predicted_index]], device=image.device))

        # Convert indices to words, skipping <SOS> if it was added (it shouldn't be here)
        return [vocabulary.itos[idx] for idx in result_caption_indices if vocabulary.itos[idx] not in ["<SOS>"]]


    @torch.no_grad()
    def caption_image_beam_search(
        self,
        image: torch.Tensor,
        vocabulary: Any,
        beam_size: int = 3,
        max_length: int = 50
    ) -> List[str]:
        """
        Generate a caption from the image using beam search.

        Args:
            image: Input image [1, channels, height, width]
            vocabulary: Vocabulary object (needs stoi, itos)
            beam_size: Beam size for search
            max_length: Maximum caption length

        Returns:
            best_sequence: List of words in the generated caption
        """
        assert image.shape[0] == 1, "Beam search requires batch size 1"
        self.eval() # Ensure model is in eval mode

        # Encode image
        # features shape: [1, embed_size]
        features = self.encoder(image)
        # Input to LSTM needs shape [1, 1, embed_size] (seq_len=1, batch=1)
        inputs = features.unsqueeze(0)

        # Initialize
        start_token_idx = vocabulary.stoi["<SOS>"]
        end_token_idx = vocabulary.stoi["<EOS>"]

        # Hidden and cell states initialization
        states = None # Start with None states for the first step (using image features)

        # Top k sequences: (log_probability_score, sequence_indices, hidden_state, cell_state)
        # Start with the <SOS> token
        initial_sequences = [(0.0, [start_token_idx], states)]
        sequences = initial_sequences
        completed_sequences = []

        # Loop for max_length steps
        for _ in range(max_length):
            all_candidates = []
            new_sequences = []

            # Expand each current candidate sequence
            for score, seq, current_states in sequences:
                # If the sequence already ended, add it to completed sequences
                if seq[-1] == end_token_idx:
                    # Ensure completed sequences have score normalized by length? Optional.
                    # score /= len(seq) # Example length normalization
                    completed_sequences.append((score, seq))
                    continue # Don't expand completed sequences

                # Prepare input for LSTM
                last_token_idx = seq[-1]
                # For the very first step (after <SOS>), the input is the image feature
                if len(seq) == 1: # Only contains <SOS>
                    lstm_input = inputs # Use image features: [1, 1, embed_size]
                else:
                    # Use embedding of the last predicted token
                    lstm_input = self.decoder.embed(torch.tensor([[last_token_idx]], device=image.device))
                    # lstm_input shape: [1, 1, embed_size]

                # Get LSTM output and next states
                # hiddens: [1, 1, hidden_size], next_states: tuple of (h, c)
                hiddens, next_states = self.decoder.lstm(lstm_input, current_states)
                # output: [1, 1, vocab_size]
                output = self.decoder.linear(hiddens)
                # log_probs: [1, 1, vocab_size] -> [vocab_size]
                log_probs = torch.log_softmax(output.squeeze(0).squeeze(0), dim=-1)

                # Get top k next words and their log probabilities
                # top_log_probs, top_indices: [beam_size]
                top_log_probs, top_indices = torch.topk(log_probs, beam_size)

                # Create new candidate sequences
                for i in range(beam_size):
                    token_idx = top_indices[i].item()
                    token_log_prob = top_log_probs[i].item()

                    new_seq = seq + [token_idx]
                    new_score = score + token_log_prob # Add log probabilities
                    all_candidates.append((new_score, new_seq, next_states))

            # If no sequences were expanded (e.g., all ended), break
            if not all_candidates:
                break

            # Sort all candidates by score (higher log prob is better)
            ordered_candidates = sorted(all_candidates, key=lambda x: x[0], reverse=True)

            # Select top k candidates for the next step
            sequences = []
            added_count = 0
            for cand_score, cand_seq, cand_states in ordered_candidates:
                 # Check if sequence ended
                 if cand_seq[-1] == end_token_idx:
                     completed_sequences.append((cand_score, cand_seq))
                 else:
                     # Add to sequences for next iteration only if we haven't filled the beam
                     if added_count < beam_size:
                         sequences.append((cand_score, cand_seq, cand_states))
                         added_count += 1
                 # Stop if we have enough completed and ongoing sequences to potentially fill the beam
                 if added_count >= beam_size and len(completed_sequences) >= beam_size:
                     break # Optimization: No need to check further candidates if beam is full

            # Prune sequences list to beam size just in case
            sequences = sequences[:beam_size]

            # Early stopping: If the best completed sequence is better than the best ongoing sequence
            # This requires careful score normalization (e.g., by length)
            # Simple version: Stop if sequences list becomes empty
            if not sequences:
                 break

        # If no sequence completed, choose the best from the ongoing ones
        if not completed_sequences:
            # Add the remaining sequences if they exist
            completed_sequences.extend([(s, seq) for s, seq, _ in sequences])
            if not completed_sequences: # Handle case where no sequences were ever generated
                 return ["<UNK>"] # Or some error indication

        # Sort completed sequences by score (higher is better)
        completed_sequences.sort(key=lambda x: x[0], reverse=True)

        # Get the sequence with the highest score
        best_seq_indices = completed_sequences[0][1]

        # Convert indices to words, skipping <SOS> and stopping at <EOS>
        result_caption = []
        for idx in best_seq_indices:
            word = vocabulary.itos[idx]
            if word == "<SOS>":
                continue
            if word == "<EOS>":
                break
            result_caption.append(word)

        return result_caption