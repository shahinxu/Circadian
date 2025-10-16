import torch
import torch.nn as nn
import math
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(dim, dim)

    def forward(self, x):
        # 残差连接： F(x) + x
        return self.layer2(self.activation(self.norm(self.layer1(x)))) + x
    
def coords_to_phase(coords):
    # coords: (batch, 2)
    x, y = coords[:, 0], coords[:, 1]
    phase = torch.atan2(y, x)
    phase = torch.where(phase < 0, phase + 2 * math.pi, phase)
    return phase

class PhaseAutoEncoder(nn.Module):
    def __init__(self, input_dim, n_celltypes=0, celltype_embedding_dim=4, dropout=0.1,
                 fourier_K=5, fourier_hidden=128, fourier_activation=nn.SiLU):
        super(PhaseAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.n_celltypes = n_celltypes
        self.use_celltype = n_celltypes > 0
        self.celltype_embedding_dim = celltype_embedding_dim
        
        if self.use_celltype:
            self.celltype_embedding = nn.Embedding(n_celltypes, celltype_embedding_dim)
            self.scale_transform = nn.Linear(celltype_embedding_dim, input_dim)
            self.additive_transform = nn.Linear(celltype_embedding_dim, input_dim)
            self.global_bias = nn.Parameter(torch.zeros(input_dim))
            encoder_input_dim = input_dim
        else:
            encoder_input_dim = input_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 128),
            ResidualBlock(128),
            ResidualBlock(128),
            nn.Linear(128, 2)
        )

        # Fourier-series MLP decoder option (maps [sin(k*phase), cos(k*phase)]_{k=1..K} -> output)
        self.fourier_K = int(fourier_K) if fourier_K is not None else 5
        in_dim = 2 * self.fourier_K
        act = fourier_activation
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, fourier_hidden),
            act(),
            nn.Linear(fourier_hidden, fourier_hidden),
            act(),
            nn.Linear(fourier_hidden, input_dim),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, celltype_indices=None):
        phase_coords_normalized = self.encode(x, celltype_indices)
        phase_angles = coords_to_phase(phase_coords_normalized)
        reconstructed = self.decode(phase_angles)
        return phase_coords_normalized, phase_angles, reconstructed
    
    def encode(self, x, celltype_indices=None):
        if self.use_celltype and celltype_indices is not None:
            celltype_emb = self.celltype_embedding(celltype_indices)
            scale_factor = self.scale_transform(celltype_emb)
            additive_factor = self.additive_transform(celltype_emb)
            modified_input = x * (1 + scale_factor) + additive_factor + self.global_bias
        else:
            modified_input = x

        modified_input = self.dropout(modified_input)
        phase_coords = self.encoder(modified_input)
        norm = torch.norm(phase_coords, dim=1, keepdim=True) + 1e-8
        phase_coords_normalized = phase_coords / norm
        return phase_coords_normalized
    
    def decode(self, phase_angles):
        if phase_angles.dim() == 1:
            phase = phase_angles.unsqueeze(1)
        else:
            phase = phase_angles

        if getattr(self, 'fourier_K', 0) > 0:
            # build [sin(k*phase), cos(k*phase)] features
            K = self.fourier_K
            harmonics = []
            for k in range(1, K + 1):
                harmonics.append(torch.sin(k * phase))
                harmonics.append(torch.cos(k * phase))
            phase_input = torch.cat(harmonics, dim=1)
            # If any extra dims expected/truncated, decoder init matched in_features to 2*K
        else:
            phase_input = phase

        return self.decoder(phase_input)