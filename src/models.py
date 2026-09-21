"""Three architecturally different one-step-ahead forecasters (input: [B, L, 5] -> output: [B])."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_FEATURES = 5


class LSTMForecaster(nn.Module):
    """Recurrent: gated memory cell reads the window step by step."""
    def __init__(self, hidden=64, layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(IN_FEATURES, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class _CausalBlock(nn.Module):
    def __init__(self, cin, cout, k, dilation, dropout):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.c1 = nn.Conv1d(cin, cout, k, dilation=dilation)
        self.c2 = nn.Conv1d(cout, cout, k, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.drop(F.relu(self.c1(F.pad(x, (self.pad, 0)))))
        h = self.drop(F.relu(self.c2(F.pad(h, (self.pad, 0)))))
        return F.relu(h + self.res(x))


class TCNForecaster(nn.Module):
    """Convolutional: stack of dilated causal residual blocks (Temporal Convolutional Network)."""
    def __init__(self, channels=32, levels=5, kernel_size=3, dropout=0.1):
        super().__init__()
        blocks, cin = [], IN_FEATURES
        for i in range(levels):
            blocks.append(_CausalBlock(cin, channels, kernel_size, 2 ** i, dropout)); cin = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, 1)
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** levels - 1)

    def forward(self, x):
        h = self.tcn(x.transpose(1, 2))          # [B, C, L]
        return self.head(h[:, :, -1]).squeeze(-1)


class TransformerForecaster(nn.Module):
    """Attention: encoder-only Transformer, sinusoidal positions, prediction read from the last token."""
    def __init__(self, d_model=64, n_heads=4, layers=2, dim_ff=128, dropout=0.1, max_len=2048):
        super().__init__()
        self.embed = nn.Linear(IN_FEATURES, d_model)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        h = self.embed(x) + self.pe[: x.size(1)]
        return self.head(self.encoder(h)[:, -1]).squeeze(-1)


def build_model(name, p):
    if name == "lstm":
        return LSTMForecaster(p["hidden"], p["layers"], p["dropout"])
    if name == "tcn":
        return TCNForecaster(p["channels"], p["levels"], p["kernel_size"], p["dropout"])
    if name == "transformer":
        return TransformerForecaster(p["d_model"], p["n_heads"], p["layers"], p["dim_ff"], p["dropout"])
    raise ValueError(name)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
