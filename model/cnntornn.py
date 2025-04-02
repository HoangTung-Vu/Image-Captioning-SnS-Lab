import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from typing import Tuple, List, Optional, Dict, Any, Union
from model.base_model import BaseModel

class Encoder(nn.Module):
    def __init__(self, embed_size: int, train_CNN: bool = False, dropout_rate: float = 0.5):
        """
        Load the pretrained ResNet-18 and replace top fc layer.
        
        Args:
            embed_size: Size of the embedding vector
            train_CNN: Whether to train the CNN
            dropout_rate: Dropout rate for regularization
        """
        super(Encoder, self).__init__()
        self.train_CNN = train_CNN
        
        # Load pretrained ResNet-18
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Replace the final fully connected layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
        
        # Add normalization and regularization
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder
        
        Args:
            images: Batch of images [batch_size, channels, height, width]
            
        Returns:
            Features: Encoded image features [batch_size, embed_size]
        """
        # Extract features from ResNet
        features = self.resnet(images)
        
        # Set gradients based on train_CNN flag
        for name, param in self.resnet.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_CNN
                
        # Apply normalization and regularization
        return self.dropout(self.bn(self.relu(features)))

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
            batch_first=False,  # Sequence first for compatibility
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Output layer
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        # Dropout for regularization
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
            captions: Target captions [seq_length, batch_size]
            
        Returns:
            outputs: Predicted word probabilities [seq_length, batch_size, vocab_size]
        """
        # Embed captions
        embeddings = self.dropout(self.embed(captions))  # [seq_length, batch_size, embed_size]
        
        # Prepare features to be the first input to the LSTM
        # features: [batch_size, embed_size] -> [1, batch_size, embed_size]
        features = features.unsqueeze(0)
        
        # Concatenate features with embeddings
        # Make sure both tensors have the same batch size dimension
        if features.size(1) != embeddings.size(1):
            # Adjust features to match embeddings batch size
            if features.size(1) < embeddings.size(1):
                # Repeat features to match batch size
                features = features.repeat(1, embeddings.size(1) // features.size(1), 1)
            else:
                # Truncate features to match batch size
                features = features[:, :embeddings.size(1), :]
        
        # Concatenate features as first input to LSTM
        inputs = torch.cat((features, embeddings), dim=0)
        
        # Pass through LSTM
        hiddens, _ = self.lstm(inputs)
        
        # Pass through linear layer
        outputs = self.linear(hiddens)
        
        return outputs

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
            trainCNN: Whether to train the CNN
            dropout_rate: Dropout rate for regularization
        """
        super(CNNtoRNN, self).__init__()
        
        # Initialize encoder and decoder
        self.encoder = Encoder(embed_size, trainCNN, dropout_rate)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers, dropout_rate)
        self.has_mha_decoder = False
    
    def forward(
        self, 
        images: torch.Tensor, 
        captions: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            images: Batch of images [batch_size, channels, height, width]
            captions: Target captions [seq_length, batch_size]
            
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
            vocabulary: Vocabulary object
            max_length: Maximum caption length
            
        Returns:
            result_caption: List of words in the generated caption
        """
        result_caption = []
        
        # Encode image
        x = self.encoder(image).unsqueeze(0)
        states = None

        # Generate caption word by word
        for _ in range(max_length):
            hiddens, states = self.decoder.lstm(x, states)
            output = self.decoder.linear(hiddens.squeeze(0))
            predicted = output.argmax(1)
            result_caption.append(predicted.item())
            
            # If end token is predicted, stop generation
            if vocabulary.itos[predicted.item()] == "<EOS>":
                break
                
            # Use predicted token as next input
            x = self.decoder.embed(predicted).unsqueeze(0)

        # Convert indices to words
        return [vocabulary.itos[idx] for idx in result_caption]
        
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
            vocabulary: Vocabulary object
            beam_size: Beam size for search
            max_length: Maximum caption length
            
        Returns:
            best_sequence: List of words in the generated caption
        """
        # Encode image
        features = self.encoder(image).unsqueeze(0)
        
        # Initialize
        start_token = vocabulary.stoi["<SOS>"]
        end_token = vocabulary.stoi["<EOS>"]
        
        # Top k sequences: (score, sequence, hidden_state, cell_state)
        sequences = [(0.0, [start_token], None, None)]
        completed_sequences = []
        
        # Loop until max_length or all beams have ended
        for _ in range(max_length):
            all_candidates = []
            
            # Expand each current candidate
            for score, seq, h, c in sequences:
                # If we've reached the end token, add to completed sequences
                if seq[-1] == end_token:
                    completed_sequences.append((score, seq, h, c))
                    continue
                
                # Feed the last token of the sequence to the model
                x = self.decoder.embed(torch.LongTensor([seq[-1]]).to(image.device)).unsqueeze(0)
                
                # For the first token, use the image features
                if len(seq) == 1:
                    states = None
                    x = features
                else:
                    states = (h, c)
                
                # Get the next output and states
                hiddens, states = self.decoder.lstm(x, states)
                h, c = states
                output = self.decoder.linear(hiddens.squeeze(0))
                
                # Get the top k words
                log_probs, indices = torch.topk(torch.log_softmax(output, dim=1), beam_size)
                
                # Create new candidates
                for i in range(beam_size):
                    token = indices[0][i].item()
                    log_prob = log_probs[0][i].item()
                    new_score = score + log_prob
                    new_seq = seq + [token]
                    all_candidates.append((new_score, new_seq, h, c))
            
            # If we have completed sequences, check if we should terminate
            if completed_sequences and len(completed_sequences) >= beam_size:
                # Sort completed sequences by score
                completed_sequences = sorted(completed_sequences, key=lambda x: x[0], reverse=True)
                # Return the best sequence
                best_seq = completed_sequences[0][1]
                return [vocabulary.itos[idx] for idx in best_seq]
            
            # Select k best candidates
            sequences = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:beam_size]
            
            # Check if all sequences have ended
            if all(s[1][-1] == end_token for s in sequences):
                break
        
        # If no sequence completed, return the best incomplete sequence
        if not completed_sequences:
            best_seq = max(sequences, key=lambda x: x[0])[1]
        else:
            # Return the best completed sequence
            completed_sequences = sorted(completed_sequences, key=lambda x: x[0], reverse=True)
            best_seq = completed_sequences[0][1]
            
        return [vocabulary.itos[idx] for idx in best_seq]