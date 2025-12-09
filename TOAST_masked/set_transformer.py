#!/usr/bin/env python3
"""
Set Transformer for Sample-Level Clustering
Each token is a sample, not a gene
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAB(nn.Module):
    """Multihead Attention Block"""
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q_proj = self.fc_q(Q)
        K_proj = self.fc_k(K)
        V_proj = self.fc_v(K)
        
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q_proj.split(dim_split, 2), 0)
        K_ = torch.cat(K_proj.split(dim_split, 2), 0)
        V_ = torch.cat(V_proj.split(dim_split, 2), 0)
        
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / (self.dim_V ** 0.5), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    """Self-Attention Block"""
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    """Induced Self-Attention Block (for efficiency with many samples)"""
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class SetTransformerEncoder(nn.Module):
    """
    Set Transformer Encoder where each token is a SAMPLE
    Input: [batch_size=1, num_samples, num_genes]
    Output: [batch_size=1, num_samples, embed_dim]
    """
    def __init__(self, num_genes, dim_hidden=128, num_heads=4, use_isab=False, num_inds=32):
        super().__init__()
        
        # 1. Per-sample feature extraction (independent)
        # 将每个样本的20000维基因表达降到128维
        self.input_proj = nn.Sequential(
            nn.Linear(num_genes, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, dim_hidden),
            nn.LayerNorm(dim_hidden)
        )
        
        # 2. Sample interaction layers
        # 让样本之间互相"看"，利用群体信息
        if use_isab:
            # Use ISAB for large number of samples (more efficient)
            self.encoder = nn.Sequential(
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=True),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=True)
            )
        else:
            # Use SAB for smaller datasets
            self.encoder = nn.Sequential(
                SAB(dim_hidden, dim_hidden, num_heads, ln=True),
                SAB(dim_hidden, dim_hidden, num_heads, ln=True)
            )
    
    def forward(self, x):
        """
        x: [batch_size, num_samples, num_genes]
        Returns: [batch_size, num_samples, dim_hidden]
        """
        # Independent feature extraction per sample
        x = self.input_proj(x)  # [B, N, dim_hidden]
        
        # Sample-level interaction
        x = self.encoder(x)  # [B, N, dim_hidden]
        
        return x


class SetPhaseAutoEncoder(nn.Module):
    """
    Set Transformer based autoencoder for circadian phase prediction
    """
    def __init__(self, input_dim, dim_hidden=128, dropout=0.1, use_isab=False, num_inds=32):
        super().__init__()
        
        # Set Transformer encoder (processes all samples together)
        self.encoder_transformer = SetTransformerEncoder(
            num_genes=input_dim,
            dim_hidden=dim_hidden,
            num_heads=4,
            use_isab=use_isab,
            num_inds=num_inds
        )
        
        # Phase prediction head (maps to 2D unit circle)
        self.phase_head = nn.Sequential(
            nn.Linear(dim_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
        # Reconstruction decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x, return_embeddings=False):
        """
        x: [batch_size, num_samples, num_genes]
        
        Returns:
            phase_coords: [batch_size, num_samples, 2] - normalized phase coordinates
            phase_angles: [batch_size, num_samples] - phase in radians
            reconstructed: [batch_size, num_samples, num_genes] - reconstructed expression
            embeddings: [batch_size, num_samples, dim_hidden] - intermediate embeddings (optional, for analysis)
        """
        # Encode all samples with Set Transformer
        embeddings = self.encoder_transformer(x)  # [B, N, dim_hidden]
        
        # Phase prediction
        phase_coords = self.phase_head(embeddings)  # [B, N, 2]
        phase_coords_norm = F.normalize(phase_coords, p=2, dim=-1)  # Normalize to unit circle
        
        # Calculate phase angles
        phase_angles = torch.atan2(phase_coords_norm[..., 1], phase_coords_norm[..., 0])
        phase_angles = torch.remainder(phase_angles + 2 * torch.pi, 2 * torch.pi)
        
        # Reconstruction
        reconstructed = self.decoder(phase_coords_norm)
        reconstructed = F.normalize(reconstructed, p=2, dim=-1)
        
        if return_embeddings:
            return phase_coords_norm, phase_angles, reconstructed, embeddings
        
        return phase_coords_norm, phase_angles, reconstructed
    
    def encode_single(self, x_sample):
        x = x_sample.unsqueeze(0).unsqueeze(0)
        embeddings = self.encoder_transformer(x)
        phase_coords = self.phase_head(embeddings)
        phase_coords_norm = F.normalize(phase_coords, p=2, dim=-1)
        phase_angle = torch.atan2(phase_coords_norm[0, 0, 1], phase_coords_norm[0, 0, 0])
        phase_angle = torch.remainder(phase_angle + 2 * torch.pi, 2 * torch.pi)
        return phase_coords_norm.squeeze(0), phase_angle.squeeze(0)

