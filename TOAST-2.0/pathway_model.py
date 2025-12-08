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
        
        # Learnable query combining tissue and global expression pattern
        self.query_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Sample-level self-attention within tissue (after pathway-tissue cross-attention)
        self.sample_attn = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm_sample = nn.LayerNorm(embed_dim)
        
        self.to_latent = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
    
    def forward(self, x_genes, x_tissue):
        pathway_tokens = F.relu(self.tokenizer(x_genes))
        tissue_emb = self.tissue_embed(x_tissue)
        pathway_mean = pathway_tokens.mean(dim=1)
        combined = torch.cat([tissue_emb, pathway_mean], dim=-1)
        query = self.query_proj(combined).unsqueeze(1)
        scores = torch.bmm(query, pathway_tokens.transpose(1, 2)) / (pathway_tokens.size(-1) ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, pathway_tokens).squeeze(1)
        unique_tissues = torch.unique(x_tissue, sorted=True)
        context_list = []
        
        for tissue_id in unique_tissues:
            tissue_mask = (x_tissue == tissue_id)
            tissue_contexts = context[tissue_mask]
            n_samples = tissue_contexts.size(0)
            
            if n_samples > 1:
                tissue_contexts_unsqueezed = tissue_contexts.unsqueeze(0)
                attn_out, _ = self.sample_attn(tissue_contexts_unsqueezed, tissue_contexts_unsqueezed, tissue_contexts_unsqueezed)
                tissue_contexts_attended = attn_out.squeeze(0)
                tissue_contexts = self.norm_sample(0.3 * tissue_contexts_attended + 0.7 * tissue_contexts)
            
            context_list.append(tissue_contexts)
        
        context_new = torch.zeros_like(context)
        idx = 0
        for tissue_id in unique_tissues:
            tissue_mask = (x_tissue == tissue_id)
            context_new[tissue_mask] = context_list[idx]
            idx += 1
        context = context_new
        
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
        self.encoder = PathwayAttentionEncoder(
            pathway_map=pathway_map,
            num_tissues=num_tissues,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        self.decoder = nn.Linear(2, input_dim)
    
    def encode_single(self, x_gene_sample, tissue_idx_scalar):
        phase_coords_norm = self.encoder(x_gene_sample.unsqueeze(0), tissue_idx_scalar.unsqueeze(0))
        
        phase_angle = torch.atan2(phase_coords_norm[:, 1], phase_coords_norm[:, 0])
        phase_angle = torch.remainder(phase_angle + 2 * torch.pi, 2 * torch.pi)
        
        return phase_coords_norm.squeeze(0), phase_angle.squeeze(0)
    
    def encode_batch(self, x_genes, tissue_idx):
        phase_coords_norm = self.encoder(x_genes, tissue_idx)
        
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
