import math
import torch
from torch import nn


class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        num_items: int,
        hidden_dim: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.item_emb_matrix = nn.Embedding(
            num_embeddings=num_items + 1,
            embedding_dim=hidden_dim,
        )

        self.positional_emb = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=hidden_dim,
        )

    def forward(self, x):
        x = self.item_emb_matrix(x)
        x *= math.sqrt(self.hidden_dim)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        device = x.device.type
        seq_len_range = torch.tensor(range(seq_len))
        positions = torch.tile(input=seq_len_range, dims=(batch_size, 1))
        positions = positions.to(device)
        positional_embs = self.positional_emb(positions)
        x += positional_embs

        return x


class PointWiseFFNN(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.W2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_1 = self.relu(self.W1(x))
        x_2 = self.W2(x_1)

        return x_2


class SelfAttnBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        max_seq_len: int,
        hidden_dim: int,
        dropout_p: float,
        device: str,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_p,
            batch_first=True,
        )
        self.ffnn = PointWiseFFNN(hidden_dim=hidden_dim)

    def dropout_layernorm(self, x: torch.Tensor) -> torch.Tensor:
        layer_norm_output = self.layer_norm(x)
        dropout_output = self.dropout(layer_norm_output)

        return dropout_output

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        attention_mask = ~torch.tril(
            torch.ones(size=(seq_len, seq_len), dtype=torch.bool)
        )
        device = x.device.type
        attention_mask = attention_mask.to(device)

        x_attn, _ = self.self_attn(
            key=self.layer_norm(x),
            query=x,
            value=x,
            attn_mask=attention_mask,
        )

        x_residual = self.dropout_layernorm(x_attn)
        x_attn_output = x + x_residual

        x_ffnn = self.ffnn(x_attn_output)
        x_ffnn_output = x_attn_output + self.dropout_layernorm(x_ffnn)

        output = x_ffnn_output * padding_mask.unsqueeze(-1)
        return output


class SASRec(nn.Module):
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
        self.device = device
        self.embedding_layer = EmbeddingLayer(
            num_items=num_items,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
        )
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
        self.self_attn_blocks = nn.Sequential(*self_attn_blocks)
        self.dropout = nn.Dropout(p=dropout_p)
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)

    def get_padding_mask(self, seqs: torch.Tensor) -> torch.Tensor:
        is_padding = torch.tensor(seqs == 0, dtype=torch.bool)
        padding_mask = ~is_padding

        return padding_mask

    def forward(
        self,
        input_seqs: torch.Tensor,
        item_idxs: torch.Tensor = None,
        positive_seqs: torch.Tensor = None,
        negative_seqs: torch.Tensor = None,
    ) -> torch.Tensor:
        padding_mask = self.get_padding_mask(seqs=input_seqs).to(self.device)
        input_embs = self.dropout(self.embedding_layer(input_seqs))
        input_embs *= padding_mask.unsqueeze(-1)
        attn_output = input_embs
        for block in self.self_attn_blocks:
            attn_output = block(x=attn_output, padding_mask=padding_mask)
        attn_output = self.layer_norm(attn_output)

        if item_idxs is not None:  
            item_embs = self.embedding_layer.item_emb_matrix(item_idxs)
            logits = attn_output @ item_embs.transpose(2, 1)
            logits = logits[:, -1, :]
            outputs = (logits,)
        elif (positive_seqs is not None) and (negative_seqs is not None):  
            positive_embs = self.dropout(self.embedding_layer(positive_seqs))
            negative_embs = self.dropout(self.embedding_layer(negative_seqs))
            positive_logits = (attn_output * positive_embs).sum(dim=-1)
            negative_logits = (attn_output * negative_embs).sum(dim=-1)
            outputs = (positive_logits,)
            outputs += (negative_logits,)

        return outputs
