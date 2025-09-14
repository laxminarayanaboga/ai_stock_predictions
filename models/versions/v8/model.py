from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    input_size: int
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2


class LSTMNextDayOHLC(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.lstm = nn.LSTM(input_size=cfg.input_size, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, batch_first=True, dropout=cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head_reg = nn.Sequential(
            nn.Linear(cfg.hidden_size, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 4),
        )
        self.head_dir = nn.Sequential(
            nn.Linear(cfg.hidden_size, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        h = self.dropout(h)
        y_reg = self.head_reg(h)
        y_dir = self.head_dir(h)
        return y_reg, y_dir
