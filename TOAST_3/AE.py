import torch
import torch.nn as nn
import torch.nn.functional as F

class SetPhaseAutoEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.2):
        super(SetPhaseAutoEncoder, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.to_latent = nn.Linear(d_model, 2)
        self.decoder = nn.Linear(2, input_dim)
        self.dropout = nn.Dropout(dropout)

    def circular_node(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) + 1e-8
        return x / norm

    def encode(self, x):
        x_proj = self.input_proj(x)
        x_proj = self.dropout(x_proj)
        trans_out = self.transformer(x_proj)
        
        phase_coords = self.to_latent(trans_out)
        
        phase_coords_normalized = self.circular_node(phase_coords)
        phase_angles = torch.atan2(phase_coords_normalized[:, :, 1], 
                                   phase_coords_normalized[:, :, 0])
        phase_angles = torch.remainder(phase_angles + 2 * torch.pi, 2 * torch.pi)
        
        return phase_coords_normalized, phase_angles

    def decode(self, phase_coords_normalized):
        decoded = self.decoder(phase_coords_normalized)
        decoded = decoded / (torch.norm(decoded, dim=-1, keepdim=True) + 1e-8)
        
        return decoded

    def forward(self, x):
        phase_coords_normalized, phase_angles = self.encode(x)
        reconstructed = self.decode(phase_coords_normalized)
        return phase_coords_normalized, phase_angles, reconstructed