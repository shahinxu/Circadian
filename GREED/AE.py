import torch
import torch.nn as nn


def circular_node(x: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(x, dim=1, keepdim=True) + 1e-8
    return x / norm


class PhaseAutoEncoder(nn.Module):
    def __init__(self, input_dim, dropout=0.0):
        super(PhaseAutoEncoder, self).__init__()
        self.input_dim = input_dim
        # transformer that treats each sample as a token in a sequence
        # d_model equals the PCA dimensionality so transformer input/output sizes match
        transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=128, dropout=dropout, batch_first=False)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)

        self.encoder = nn.Linear(input_dim, 2)
        self.decoder = nn.Linear(2, input_dim)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x):
        x = self.dropout(x)
        # transformer expects input shape (seq_len, batch, embed_dim)
        # here we treat the whole dataset as a single "sentence" so batch=1
        seq = x.unsqueeze(1)  # (seq_len, 1, input_dim)
        trans_out = self.transformer(seq)  # (seq_len, 1, input_dim)
        trans_out = trans_out.squeeze(1)  # (seq_len, input_dim)
        phase_coords = self.encoder(trans_out)
        phase_coords_normalized = circular_node(phase_coords)
        phase_angles = torch.atan2(phase_coords_normalized[:, 1], phase_coords_normalized[:, 0])
        phase_angles = torch.remainder(phase_angles + 2 * torch.pi, 2 * torch.pi)
        return phase_coords_normalized, phase_angles

    def decode(self, phase_coords_normalized):
        return self.decoder(phase_coords_normalized)

    def forward(self, x):
        phase_coords_normalized, phase_angles = self.encode(x)
        reconstructed = self.decode(phase_coords_normalized)
        return phase_coords_normalized, phase_angles, reconstructed