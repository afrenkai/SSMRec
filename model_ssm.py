import math
from model import EmbeddingLayer, PointWiseFFNN
import torch
from torch import nn
import torch.nn.functional as F

class MambaLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        state_dim: int = 64,
        expand_factor: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.expand_dim = hidden_dim * expand_factor

        self.in_proj = nn.Linear(hidden_dim, self.expand_dim * 2)

        self.conv1d = nn.Conv1d(
            in_channels=self.expand_dim,
            out_channels=self.expand_dim,
            kernel_size=3,
            padding=1,
            groups=self.expand_dim,
        )

        self.x_proj = nn.Linear(self.expand_dim, state_dim + state_dim + state_dim)

        self.A_log = nn.Parameter(torch.randn(state_dim) * 0.1)
        self.D = nn.Parameter(torch.ones(self.expand_dim))

        self.out_proj = nn.Linear(self.expand_dim, hidden_dim)

        self.dropout = nn.Dropout(p=dropout_p)
        
        # Add layer norm for stability
        self.norm = nn.LayerNorm(self.expand_dim)

    def forward(self, u):
        batch_size, seq_len, _ = u.shape

        xz = self.in_proj(u)
        x, z = xz.chunk(2, dim=-1)  # (B, L, expand_dim) each

        x = x.transpose(1, 2)  # (B, expand_dim, L)
        x = self.conv1d(x)
        x = x.transpose(1, 2)  # (B, L, expand_dim)
        x = F.silu(x)

        ssm_params = self.x_proj(x)
        delta, B, C = torch.split(
            ssm_params, [self.state_dim, self.state_dim, self.state_dim], dim=-1
        )
        delta = F.softplus(delta)

        A = -torch.exp(torch.clamp(self.A_log.float(), min=-5, max=5))

        delta_A = delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        A_discrete = torch.exp(torch.clamp(delta_A, min=-10, max=0)) 

        B_discrete = (delta.unsqueeze(-1) * B.unsqueeze(-1)).expand(
            -1, -1, -1, self.expand_dim
        )
        h = torch.zeros(batch_size, self.state_dim, self.expand_dim, device=u.device)
        outputs = torch.zeros(batch_size, seq_len, self.expand_dim, device=u.device)
 
        for t in range(seq_len):
            h = torch.bmm(A_discrete[:, t], h) + torch.bmm(B_discrete[:, t], x[:, t].unsqueeze(-1))
            h = torch.clamp(h, min=-100, max=100)  
            y = torch.einsum('bsd,bs->bd', h, C[:, t]) + self.D * x[:, t]
            outputs[:, t] = y

        y = self.norm(outputs) 
        y = y * F.silu(z)
        output = self.out_proj(y)

        return self.dropout(output)


class MambaBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        expand_factor: int,
        dropout_p: float,
        device: str,
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)

        self.mamba = MambaLayer(
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            expand_factor=expand_factor,
            dropout_p=dropout_p,
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:

        x_mamba = self.mamba(self.layer_norm(x))
        x = x + x_mamba

        output = x * padding_mask.unsqueeze(-1)

        return output


class SSMRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_blocks: int,
        hidden_dim: int,
        max_seq_len: int,
        dropout_p: float,
        share_item_emb: bool,
        num_heads: int,
        device: str,
        state_dim: int = 64,
        expand_factor: int = 2,
        use_dummy_embeddings: bool = False,
    ) -> None:
        super().__init__()

        self.device = device

        self.embedding_layer = EmbeddingLayer(
            num_items=num_items,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
        )
        
        if use_dummy_embeddings:
            with torch.no_grad():
                nn.init.constant_(self.embedding_layer.item_emb_matrix.weight, 0.01)
                nn.init.constant_(self.embedding_layer.positional_emb.weight, 0.01)
            self.embedding_layer.item_emb_matrix.weight.requires_grad = False
            self.embedding_layer.positional_emb.weight.requires_grad = False

        ssm_blocks = [
                MambaBlock(
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                expand_factor=expand_factor,
                dropout_p=dropout_p,
                device=device,
            )
            for block in range(num_blocks)
        ]



        self.ssm_blocks = nn.ModuleList(ssm_blocks)

        self.dropout = nn.Dropout(p=dropout_p)
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)

    def get_padding_mask(self, seqs: torch.Tensor) -> torch.Tensor:
        is_padding = (seqs == 0).detach().clone()
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

        ssm_output = input_embs
        for block in self.ssm_blocks:
            ssm_output = block(x=ssm_output, padding_mask=padding_mask)
        ssm_output = self.layer_norm(ssm_output)

        if item_idxs is not None:
            item_embs = self.embedding_layer.item_emb_matrix(item_idxs)
            logits = ssm_output @ item_embs.transpose(2, 1)
            logits = logits[:, -1, :]
            outputs = (logits,)
        elif (positive_seqs is not None) and (negative_seqs is not None):
            positive_embs = self.dropout(self.embedding_layer(positive_seqs))
            negative_embs = self.dropout(self.embedding_layer(negative_seqs))
            positive_logits = (ssm_output * positive_embs).sum(dim=-1)
            negative_logits = (ssm_output * negative_embs).sum(dim=-1)
            outputs = (positive_logits,)
            outputs += (negative_logits,)

        return outputs
