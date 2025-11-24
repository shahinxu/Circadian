import torch
import torch.nn as nn


def circular_node(x: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(x, dim=1, keepdim=True) + 1e-8
    return x / norm


class PhaseAutoEncoder(nn.Module):
    def __init__(self, input_dim, dropout=0.2, nhead=5):
        super(PhaseAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.nhead = nhead

        if input_dim % nhead == 0:
            transformer_dim = input_dim
        else:
            transformer_dim = input_dim + (nhead - (input_dim % nhead))
        self.transformer_dim = transformer_dim

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=nhead,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            batch_first=False,
            activation='gelu',
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)

        self.encoder = nn.Linear(transformer_dim, 2)
        self.decoder = nn.Linear(2, input_dim)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x):
        seq = x.unsqueeze(1)
        trans_out = self.transformer(seq)
        trans_out = trans_out.squeeze(1)
        phase_coords = self.encoder(trans_out)
        phase_coords_normalized = circular_node(phase_coords)
        phase_angles = torch.atan2(phase_coords_normalized[:, 1], phase_coords_normalized[:, 0])
        phase_angles = torch.remainder(phase_angles + 2 * torch.pi, 2 * torch.pi)
        return phase_coords_normalized, phase_angles

    def decode(self, phase_coords_normalized):
        decoded = self.decoder(phase_coords_normalized)
        return decoded

    def forward(self, x):
        phase_coords_normalized, phase_angles = self.encode(x)
        reconstructed = self.decode(phase_coords_normalized)
        return phase_coords_normalized, phase_angles, reconstructed