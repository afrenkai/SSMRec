import math
import torch
from torch import nn

#file provided by Professor

class EmbeddingLayer(nn.Module):
    """
    Purpose: Converts item IDs and positional information into dense embeddings. 
      This helps transform sparse categorical data into a format suitable for processing 
      by neural networks.
    """
    def __init__(
        self,
        num_items: int,
        hidden_dim: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        # Initialize item embeddings and positional embeddings
        self.hidden_dim = hidden_dim
        self.item_emb_matrix = nn.Embedding(
            num_embeddings=num_items + 1,  # Add 1 for padding or unknown token
            embedding_dim=hidden_dim,
        )

        self.positional_emb = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=hidden_dim,
        )

    def forward(self, x):
        """
        Purpose: Convert item indices to dense embeddings
        Parameters: x: A tensor of item indices, where each row represents a sequence of item interactions.
        Returns: A tensor of dense embeddings with positional information added.
        """
        ##################################################
        #ADD YOUR CODE HERE
        # Transform each item index into its corresponding dense vector.
        x = self.item_emb_matrix(x)
        # Scale the embeddings by multiplying them with the square root of 
        # hidden_dim to maintain consistency in the magnitude of the embeddings.
        x *= math.sqrt(self.hidden_dim)
        # Extract batch_size and seq_len from the shape of x
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        device = x.device.type
        # Create a range of values to represent the positions in the sequence
        seq_len_range = torch.tensor(range(seq_len))
        # Replicate the position indices across the batch(Hint: torch.tile)
        positions = torch.tile(input=seq_len_range, dims=(batch_size, 1))
        positions = positions.to(device)
        # Generate positional embeddings
        positional_embs = self.positional_emb(positions)
        # Add positional embeddings to item embeddings
        x += positional_embs

        return x

class PointWiseFFNN(nn.Module):
    """Purpose: Initializes the layers of the feed-forward neural network."""
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # Define two linear transformations (W1 and W2)
        self.W1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.W2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        # ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Purpose: Executes the forward pass of the FFNN
        Returns: A tensor of the same shape as the input, with features refined 
          by the two linear transformations and ReLU activation.
        """
        ##################################################
        #ADD YOUR CODE HERE
        # Apply first linear layer (W1) and ReLU activation
        x_1 = self.relu(self.W1(x))
        # Apply second linear layer (W2) to refine the features
        x_2 = self.W2(x_1)

        return x_2

class SelfAttnBlock(nn.Module):
    """
    Purpose: Implements a self-attention block that processes input sequences 
        by applying multi-head self-attention and a feed-forward neural network (FFNN).
    Preconditions: 
        Num_heads specify the number of attention heads in the multi-head self-attention mechanism.
        Hidden_dim, dropout_p, max_seq_len, and device ensure that tensors are processed with the correct shapes and devices.
        Input tensor x for self-attention and feed-forward processing
    """
    def __init__(
        self,
        num_heads: int,
        max_seq_len: int,
        hidden_dim: int,
        dropout_p: float,
        device: str,
    ) -> None:
        super().__init__()
        # Store max_seq_len as class attributes
        self.max_seq_len = max_seq_len
        # Initialize layer normalization and dropout
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        # Initialize multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_p,
            batch_first=True,
        )
        # Initialize feed-forward neural network
        self.ffnn = PointWiseFFNN(hidden_dim=hidden_dim)

    def dropout_layernorm(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Purpose: normalize the input and add regularization to prevent overfitting
        Preconditions: Input tensor x to be normalized and regularized.
        Returns: A tensor after applying layer normalization and dropout.
        """
        # Normalize the input and then apply dropout
        layer_norm_output = self.layer_norm(x)
        dropout_output = self.dropout(layer_norm_output)

        return dropout_output

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """ 
        Purpose: Executes the forward pass of the self-attention block.
        Preconditions: 
            Input tensor x of shape (batch_size, seq_len, hidden_dim).
            Mask padding_mask to ignore padded positions in the sequence.
        Returns: A tensor with shape (batch_size, seq_len, hidden_dim).
        """
        # Create a lower triangular matrix of seq_len by seq_len as attention mask
        seq_len = x.shape[1]
        attention_mask = ~torch.tril(
            torch.ones(size=(seq_len, seq_len), dtype=torch.bool)
        )
        device = x.device.type
        attention_mask = attention_mask.to(device)

        # Pass the input tensor through self_attn layer, 
        # using layer_normed x as key, and using x as query, value
        x_attn, _ = self.self_attn(
            key=self.layer_norm(x),
            query=x,
            value=x,
            attn_mask=attention_mask,
        )

        # Apply dropout_layernorm x_attn on resulting output
        x_residual = self.dropout_layernorm(x_attn)

        # Add dropout_layernormed x_attn to input tensor for residual connection
        x_attn_output = x + x_residual

        # Apply feed-forward neural network with dropout and residual connection
        x_ffnn = self.ffnn(x_attn_output)
        x_ffnn_output = x_attn_output + self.dropout_layernorm(x_ffnn)

        # Apply padding mask to ensure that padding tokens do not affect the output.
        output = x_ffnn_output * padding_mask.unsqueeze(-1)
        return output


class SASRec(nn.Module):
    """ Purpose: Implements the SASRec (Self-Attentive Sequential Recommendation) model."""
    def __init__(
        self,
        num_items: int,
        num_blocks: int,
        num_heads: int,
        hidden_dim: int,
        max_seq_len: int,
        dropout_p: float,
        share_item_emb: bool,
        device: str,
    ) -> None:
        super().__init__()
        # Store device as class attribute
        self.device = device
        # Initialize embedding layer
        self.embedding_layer = EmbeddingLayer(
            num_items=num_items,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
        )
        # Create and stack self-attention blocks
        self_attn_blocks = [
            SelfAttnBlock(
                max_seq_len=max_seq_len,
                hidden_dim=hidden_dim,
                dropout_p=dropout_p,
                num_heads = num_heads,
                device=device,
            )
            for _ in range(num_blocks)
        ]
        # Use nn.Sequential to stack the blocks
        self.self_attn_blocks = nn.Sequential(*self_attn_blocks)
        # Initialize dropout and layer normalization
        self.dropout = nn.Dropout(p=dropout_p)
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)

    def get_padding_mask(self, seqs: torch.Tensor) -> torch.Tensor:
        """ 
        Purpose: Creates a padding mask to identify and ignore padded positions in the input sequences.
        Preconditions: Input tensor seqs containing sequences of item IDs.
        Returns: A binary mask tensor where 1 indicates a real item, and 0 indicates padding.
        """
        ##################################################
        #ADD YOUR CODE HERE
        # Identify padding tokens
        is_padding = torch.tensor(seqs == 0, dtype=torch.bool)
        # Create padding mask
        padding_mask = ~is_padding

        return padding_mask

    def forward(
        self,
        input_seqs: torch.Tensor,
        item_idxs: torch.Tensor = None,
        positive_seqs: torch.Tensor = None,
        negative_seqs: torch.Tensor = None,
    ) -> torch.Tensor:
        """ 
        Purpose: Executes the forward pass of the SASRec model
        Preconditions: 
            Input_seqs (torch.Tensor): Input sequences containing item IDs.
            Item_idxs (torch.Tensor, optional): Indices of items for inference.
            Positive_seqs (torch.Tensor, optional): Positive item sequences for training.
            Negative_seqs (torch.Tensor, optional): Negative item sequences for training.
        Returns: A tensor containing logits for predictions.
        """
        ##################################################
        #ADD YOUR CODE HERE
        # Generate padding mask and move to the correct device
        padding_mask = self.get_padding_mask(seqs=input_seqs).to(self.device)
        # Apply embedding layer and dropout, then mask padding
        input_embs = self.dropout(self.embedding_layer(input_seqs))
        input_embs *= padding_mask.unsqueeze(-1)
        # Pass the embeddings sequentially through stacked self-attention blocks
        # (Hint: use For loop because nn.Sequential can't handle multiple inputs.)
        attn_output = input_embs
        for block in self.self_attn_blocks:
            attn_output = block(x=attn_output, padding_mask=padding_mask)
        attn_output = self.layer_norm(attn_output)

        # Inference: For item_idxs is not None, predict next item
        if item_idxs is not None:  
            # Convert item_idxs to embeddings for computing similarity scores.
            item_embs = self.embedding_layer.item_emb_matrix(item_idxs)
            # Calculate dot product between the output of the self-attention and the item embeddings as logits.           
            logits = attn_output @ item_embs.transpose(2, 1)
            # Return the logits for the last time step
            logits = logits[:, -1, :]
            outputs = (logits,)
        # Training case: For positive_seqs and negative_seqs are not None
        elif (positive_seqs is not None) and (negative_seqs is not None):  
            # Convert positive and negative samples into embeddings.
            positive_embs = self.dropout(self.embedding_layer(positive_seqs))
            negative_embs = self.dropout(self.embedding_layer(negative_seqs))
            # Calculate dot products between the attention output and the corresponding embeddings as positive and negative logits
            positive_logits = (attn_output * positive_embs).sum(dim=-1)
            negative_logits = (attn_output * negative_embs).sum(dim=-1)
            # Return both positive and negative logits for loss calculation during training
            outputs = (positive_logits,)
            outputs += (negative_logits,)

        return outputs
