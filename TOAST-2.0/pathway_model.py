import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class PathwayTokenizer(nn.Module):
    def __init__(self, pathway_indices: List[List[int]], embed_dim: int = 128):
        super().__init__()
        self.indices = pathway_indices
        self.projectors = nn.ModuleList([
            nn.Linear(len(idx), embed_dim) for idx in pathway_indices
        ])
    
    def forward(self, x):
        tokens = [proj(x[:, idx]) for idx, proj in zip(self.indices, self.projectors)]
        return torch.stack(tokens, dim=1)


class PathwayAttentionEncoder(nn.Module):
    def __init__(
        self, 
        pathway_map: List[List[int]],
        num_tissues: int,
        embed_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.tokenizer = PathwayTokenizer(pathway_map, embed_dim)
        self.tissue_embed = nn.Embedding(num_tissues, embed_dim)
        self.to_latent = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
    
    def forward(self, x_genes, x_tissue):
        H = F.relu(self.tokenizer(x_genes))
        Q = self.tissue_embed(x_tissue).unsqueeze(1)
        
        scores = torch.bmm(Q, H.transpose(1, 2)) / (H.size(-1) ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, H).squeeze(1)
        
        latent = self.to_latent(context)
        latent_norm = F.normalize(latent, p=2, dim=1)
        
        return latent_norm


class PathwayAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        pathway_map: List[List[int]],
        num_tissues: int,
        embed_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = PathwayAttentionEncoder(
            pathway_map=pathway_map,
            num_tissues=num_tissues,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        self.decoder = nn.Linear(2, input_dim)
    
    def encode(self, x_genes, x_tissue):
        phase_coords = self.encoder(x_genes, x_tissue)
        phase_angles = torch.atan2(phase_coords[:, 1], phase_coords[:, 0])
        phase_angles = torch.remainder(phase_angles + 2 * torch.pi, 2 * torch.pi)
        
        return phase_coords, phase_angles
    
    def decode(self, phase_coords):
        return self.decoder(phase_coords)
    
    def forward(self, x_genes, x_tissue):
        phase_coords, phase_angles = self.encode(x_genes, x_tissue)
        reconstructed = self.decode(phase_coords)
        reconstructed = F.normalize(reconstructed, p=2, dim=1)
        
        return phase_coords, phase_angles, reconstructed


class PathwayAutoencoderWithTissue(nn.Module):
    def __init__(
        self,
        input_dim: int,
        pathway_map: List[List[int]],
        num_tissues: int = 1,
        embed_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.tokenizer = PathwayTokenizer(pathway_map, embed_dim)
        self.tissue_embed = nn.Embedding(num_tissues, embed_dim)
        
        self.to_latent = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
        
        self.decoder = nn.Linear(2, input_dim)
    
    def encode_single(self, x_gene_sample, tissue_idx_scalar):
        H = F.relu(self.tokenizer(x_gene_sample.unsqueeze(0)))
        Q = self.tissue_embed(tissue_idx_scalar.unsqueeze(0)).unsqueeze(1)
        
        scores = torch.bmm(Q, H.transpose(1, 2)) / (H.size(-1) ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, H).squeeze(1)
        
        phase_coords = self.to_latent(context)
        phase_coords_norm = F.normalize(phase_coords, p=2, dim=1)
        
        phase_angle = torch.atan2(phase_coords_norm[:, 1], phase_coords_norm[:, 0])
        phase_angle = torch.remainder(phase_angle + 2 * torch.pi, 2 * torch.pi)
        
        return phase_coords_norm.squeeze(0), phase_angle.squeeze(0)
    
    def encode_batch(self, x_genes, tissue_idx):
        H = F.relu(self.tokenizer(x_genes))
        Q = self.tissue_embed(tissue_idx).unsqueeze(1)
        
        scores = torch.bmm(Q, H.transpose(1, 2)) / (H.size(-1) ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, H).squeeze(1)
        
        phase_coords = self.to_latent(context)
        phase_coords_norm = F.normalize(phase_coords, p=2, dim=1)
        
        phase_angles = torch.atan2(phase_coords_norm[:, 1], phase_coords_norm[:, 0])
        phase_angles = torch.remainder(phase_angles + 2 * torch.pi, 2 * torch.pi)
        
        return phase_coords_norm, phase_angles
    
    def decode(self, phase_coords):
        return self.decoder(phase_coords)
    
    def forward(self, x_genes, tissue_idx):
        if x_genes.dim() == 3:
            batch_size, n_samples, n_genes = x_genes.shape
            x_genes_flat = x_genes.view(-1, n_genes)
            tissue_idx_flat = tissue_idx.view(-1) if tissue_idx.dim() == 2 else tissue_idx
            
            phase_coords, phase_angles = self.encode_batch(x_genes_flat, tissue_idx_flat)
            reconstructed = self.decode(phase_coords)
            reconstructed = F.normalize(reconstructed, p=2, dim=1)
            
            phase_coords = phase_coords.view(batch_size, n_samples, 2)
            phase_angles = phase_angles.view(batch_size, n_samples)
            reconstructed = reconstructed.view(batch_size, n_samples, n_genes)
            
            return phase_coords, phase_angles, reconstructed
        else:
            phase_coords, phase_angles = self.encode_single(x_genes, tissue_idx)
            reconstructed = self.decode(phase_coords.unsqueeze(0))
            reconstructed = F.normalize(reconstructed, p=2, dim=1)
            return phase_coords.unsqueeze(0), phase_angles.unsqueeze(0), reconstructed
