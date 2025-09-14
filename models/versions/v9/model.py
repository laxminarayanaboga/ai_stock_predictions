from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    input_size: int
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    attn_dim: int = 64


class LSTMAttnNextDayOHLC(nn.Module):
    """LSTM with simple attention pooling over time.

    - Encoder: LSTM returns sequence of hidden states
    - Attention: additive attention to weight timesteps
    - Heads: regression (open_rel, dh_rel, dl_rel, dc_rel) and direction logits (2)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout,
        )
        self.attn = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.attn_dim),
            nn.Tanh(),
            nn.Linear(cfg.attn_dim, 1),
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.head_reg = nn.Sequential(
            nn.Linear(cfg.hidden_size, 96),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(96, 4),
        )
        self.head_dir = nn.Sequential(
            nn.Linear(cfg.hidden_size, 48),
            nn.GELU(),
            nn.Linear(48, 2),
        )

    def forward(self, x):
        # x: [B, T, F]
        h_seq, _ = self.lstm(x)
        # attention weights over time: [B, T, 1]
        attn_scores = self.attn(h_seq)
        attn_weights = torch.softmax(attn_scores.squeeze(-1), dim=-1).unsqueeze(-1)
        # context as weighted sum: [B, H]
        context = (h_seq * attn_weights).sum(dim=1)
        context = self.dropout(context)
        y_reg = self.head_reg(context)
        y_dir = self.head_dir(context)
        return y_reg, y_dir
