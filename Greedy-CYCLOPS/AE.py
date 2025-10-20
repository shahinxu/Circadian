import torch
import torch.nn as nn


def circular_node(x: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(x, dim=1, keepdim=True) + 1e-8
    return x / norm


class PhaseAutoEncoder(nn.Module):
    def __init__(self, input_dim, dropout=0.0):
        super(PhaseAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoder = nn.Linear(input_dim, 2)
        self.decoder = nn.Linear(1, input_dim)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x):
        x = self.dropout(x)
        phase_coords = self.encoder(x)
        phase_coords_normalized = circular_node(phase_coords)
        phase_angles = torch.atan2(phase_coords_normalized[:, 1], phase_coords_normalized[:, 0])
        phase_angles = torch.remainder(phase_angles + 2 * torch.pi, 2 * torch.pi)
        return phase_coords_normalized, phase_angles

    def decode(self, phase_coords_normalized):
        return self.decoder(phase_coords_normalized.unsqueeze(1))

    def forward(self, x):
        phase_coords_normalized, phase_angles = self.encode(x)
        reconstructed = self.decode(phase_angles)
        return phase_coords_normalized, phase_angles, reconstructed