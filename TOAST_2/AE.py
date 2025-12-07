import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# Part 1: 非学习型编码器组件 (Frozen Encoders)
# ==========================================

class FrozenContinuousEncoder(nn.Module):
    def __init__(self, d_model, max_val=100.0):
        super().__init__()
        d_model = (d_model // 2) * 2
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)
        self.max_val = max_val

    def forward(self, x):
        x_scaled = x * (100.0 / self.max_val) 
        phase = x_scaled * self.div_term
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

class FrozenCategoricalEncoder(nn.Module):
    def __init__(self, num_categories, d_model):
        super().__init__()
        self.embedding = nn.Embedding(num_categories, d_model)
        self.embedding.weight.requires_grad = False
        nn.init.orthogonal_(self.embedding.weight)

    def forward(self, x):
        return self.embedding(x)

class HybridCovariateEncoder(nn.Module):
    def __init__(self, continuous_dims, categorical_cards, total_d_model=128):
        super().__init__()
        self.continuous_dims = continuous_dims
        self.encoders = nn.ModuleList()
        
        num_vars = continuous_dims + len(categorical_cards)
        part_dim = (total_d_model // num_vars // 2) * 2
        if part_dim < 2: part_dim = 2
        
        self.part_dims = []

        for _ in range(continuous_dims):
            self.encoders.append(FrozenContinuousEncoder(d_model=part_dim))
            self.part_dims.append(part_dim)
            
        for card in categorical_cards:
            self.encoders.append(FrozenCategoricalEncoder(num_categories=card, d_model=part_dim))
            self.part_dims.append(part_dim)

        self.output_dim = sum(self.part_dims)

    def forward(self, covariates):
        outputs = []
        for i in range(self.continuous_dims):
            val = covariates[..., i:i+1] # float
            outputs.append(self.encoders[i](val))
            
        for j, encoder in enumerate(self.encoders[self.continuous_dims:]):
            col_idx = self.continuous_dims + j
            val = covariates[..., col_idx].long() # int/long
            outputs.append(encoder(val))
            
        return torch.cat(outputs, dim=-1)

# ==========================================
# Part 2: AdaLN 和 Transformer Block
# ==========================================

class AdaLN(nn.Module):
    def __init__(self, d_model, d_condition):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.cond_proj = nn.Sequential(
            nn.Linear(d_condition, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model)
        )

    def forward(self, x, condition):
        style = self.cond_proj(condition)
        gamma, beta = style.chunk(2, dim=-1)
        return (1 + gamma) * self.norm(x) + beta

class AdaLNTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, d_condition, dropout=0.1):
        super().__init__()
        self.ada_ln1 = AdaLN(d_model, d_condition)
        self.ada_ln2 = AdaLN(d_model, d_condition)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, condition):
        x_norm = self.ada_ln1(x, condition)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        x_norm = self.ada_ln2(x, condition)
        ff_out = self.feed_forward(x_norm)
        x = x + self.dropout(ff_out)
        return x

# ==========================================
# Part 3: 修改后的主模型
# ==========================================

class ConditionalTOAST(nn.Module):
    def __init__(
        self, 
        input_dim, 
        continuous_dims, 
        categorical_cards, 
        d_model=128, 
        nhead=4, 
        num_layers=2, 
        dropout=0.2):
        super().__init__()
        self.cov_encoder = HybridCovariateEncoder(
            continuous_dims=continuous_dims,
            categorical_cards=categorical_cards,
            total_d_model=d_model
        )
        self.d_cond = self.cov_encoder.output_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            AdaLNTransformerBlock(d_model, nhead, d_condition=self.d_cond, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.to_latent = nn.Linear(d_model, 2)
        # self.decoder = nn.Sequential(
        #     nn.Linear(2 + self.d_cond, 64),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(64, input_dim)
        # )
        self.decoder = nn.Linear(2, input_dim)

    def circular_node(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) + 1e-8
        return x / norm

    def encode(self, x, covariates):
        c_emb = self.cov_encoder(covariates)
        x_proj = self.input_proj(x)
        x_proj = self.dropout(x_proj)
        curr_x = x_proj
        for layer in self.layers:
            curr_x = layer(curr_x, c_emb)
        phase_coords = self.to_latent(curr_x)
        phase_coords_normalized = self.circular_node(phase_coords)
        phase_angles = torch.atan2(phase_coords_normalized[:, :, 1], phase_coords_normalized[:, :, 0])
        phase_angles = torch.remainder(phase_angles + 2 * torch.pi, 2 * torch.pi)
        return phase_coords_normalized, phase_angles, c_emb

    def decode(self, phase_coords_normalized, c_emb):
        z_combined = torch.cat([phase_coords_normalized, c_emb], dim=-1)
        return self.decoder(phase_coords_normalized)

    def forward(self, x, covariates):
        phase_norm, phase_angles, c_emb = self.encode(x, covariates)
        reconstructed = self.decode(phase_norm, c_emb)
        return phase_norm, phase_angles, reconstructed