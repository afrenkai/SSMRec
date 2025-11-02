import math
from model import EmbeddingLayer, PointWiseFFNN 
import torch
from torch import nn
import torch.nn.functional as F



class S4Layer(nn.Module):
    """
    Purpose: Implements a simplified State Space Model (S4) layer.
    This is a linear time-invariant state space model that processes sequences efficiently.
    
    The model is defined by:
    x'(t) = Ax(t) + Bu(t)
    y(t) = Cx(t) + Du(t)
    
    where A, B, C, D are learned parameters.
    note that A should be diag for stablity
    """
    def __init__(
        self,
        hidden_dim: int,
        state_dim: int = 64,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        self.log_A_real = nn.Parameter(torch.randn(state_dim))
        self.A_imag = nn.Parameter(torch.randn(state_dim))
        
        self.B = nn.Parameter(torch.randn(state_dim, hidden_dim))
        
        self.C = nn.Parameter(torch.randn(hidden_dim, state_dim, 2))
        self.D = nn.Parameter(torch.randn(hidden_dim))
        
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.log_step = nn.Parameter(torch.randn(1))
        
    def get_discrete_params(self):
        step = torch.exp(self.log_step)
        
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        
        A_bar = torch.exp(A * step)
        
        B_bar = (A_bar - 1) / A * self.B.to(A.dtype)
        return A_bar, B_bar
    
    def forward(self, u):
        batch_size, seq_len, _ = u.shape
        
        A_bar, B_bar = self.get_discrete_params()
        x = torch.zeros(batch_size, self.state_dim, dtype=torch.complex64, device=u.device)
        
        outputs = []
        
        for t in range(seq_len):
            x = A_bar[None, :] * x + (B_bar @ u[:, t].T).T.to(torch.complex64)
            
            C_complex = torch.view_as_complex(self.C.contiguous())
            y = torch.real((C_complex @ x.unsqueeze(-1)).squeeze(-1))
            y = y + self.D[None, :] * u[:, t]
            
            outputs.append(y)
        
        output = torch.stack(outputs, dim=1)
        
        return self.dropout(output)


class SSMBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        dropout_p: float,
        device: str,
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.ssm = S4Layer(
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            dropout_p=dropout_p,
        )
        
        self.ffnn = PointWiseFFNN(hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x_ssm = self.ssm(self.layer_norm(x))
        x = x + self.dropout(x_ssm)
        
        x_ffnn = self.ffnn(self.layer_norm(x))
        x = x + self.dropout(x_ffnn)
        
        output = x * padding_mask.unsqueeze(-1)
        
        return output


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
        
        self.x_proj = nn.Linear(self.expand_dim, state_dim + state_dim + hidden_dim)
        
        self.A_log = nn.Parameter(torch.randn(state_dim))
        self.D = nn.Parameter(torch.ones(hidden_dim))
        
        self.out_proj = nn.Linear(self.expand_dim, hidden_dim)
        
        self.dropout = nn.Dropout(p=dropout_p)
        
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
            ssm_params, 
            [self.state_dim, self.state_dim, self.hidden_dim], 
            dim=-1
        )
        delta = F.softplus(delta)
        
        # this should use a custom CUDA kernel for efficiency but im lazy
        A = -torch.exp(self.A_log.float())
        
        A_discrete = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        B_discrete = delta.unsqueeze(-1) * B.unsqueeze(-1)
        
        h = torch.zeros(batch_size, self.state_dim, self.hidden_dim, device=u.device)
        outputs = []
        
        for t in range(seq_len):
            h = A_discrete[:, t] * h + B_discrete[:, t] * x[:, t].unsqueeze(1)
            y = (h * C[:, t].unsqueeze(1)).sum(dim=1)
            y = y + self.D * u[:, t]
            outputs.append(y)
        
        y = torch.stack(outputs, dim=1)
        
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
        # Apply Mamba layer with residual connection
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
        device: str,
        ssm_type: str = "s4",  # "s4" or "mamba"
        state_dim: int = 64,
        expand_factor: int = 2,
    ) -> None:
        super().__init__()
        self.device = device
        self.ssm_type = ssm_type
        
        self.embedding_layer = EmbeddingLayer(
            num_items=num_items,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
        )
        
        if ssm_type == "s4":
            ssm_blocks = [
                SSMBlock(
                    hidden_dim=hidden_dim,
                    state_dim=state_dim,
                    dropout_p=dropout_p,
                    device=device,
                )
                for block in range(num_blocks)
            ]
        elif ssm_type == "mamba":
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
        else:
            raise ValueError(f"Unknown SSM type: {ssm_type}")
        
        self.ssm_blocks = nn.ModuleList(ssm_blocks)
        
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


