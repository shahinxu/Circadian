import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math

class AttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int = 8):
        super(AttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        assert embed_dim % n_heads == 0
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, embed_dim = x.shape
        Q = self.query(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attention_weights, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out(context)
        
        return output, attention_weights

class PointerNetwork(nn.Module):    
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 3, n_heads: int = 8):
        super(PointerNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input embedding
        self.input_embed = nn.Linear(input_dim, hidden_dim)
        
        # Encoder: stack of attention layers
        self.encoder_layers = nn.ModuleList([
            AttentionLayer(hidden_dim, n_heads) for _ in range(n_layers)
        ])
        
        # Decoder: attention-based pointer mechanism
        self.decoder_start = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.decoder_attention = AttentionLayer(hidden_dim, n_heads)
        
        # Pointer mechanism
        self.pointer_query = nn.Linear(hidden_dim, hidden_dim)
        self.pointer_key = nn.Linear(hidden_dim, hidden_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input features"""
        # x shape: (batch_size, seq_len, input_dim)
        embedded = self.input_embed(x)  # (batch_size, seq_len, hidden_dim)
        
        encoded = embedded
        for layer in self.encoder_layers:
            residual = encoded
            encoded, _ = layer(encoded)
            encoded = self.layer_norm(encoded + residual)
            encoded = self.dropout(encoded)
            
        return encoded
    
    def decode_step(self, decoder_input: torch.Tensor, encoder_output: torch.Tensor, 
                   mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        context, _ = self.decoder_attention(decoder_input)
        
        query = self.pointer_query(context)  # (batch_size, 1, hidden_dim)
        keys = self.pointer_key(encoder_output)  # (batch_size, seq_len, hidden_dim)
        
        scores = torch.matmul(query, keys.transpose(-2, -1))  # (batch_size, 1, seq_len)
        scores = scores.squeeze(1)  # (batch_size, seq_len)
        
        scores = scores.masked_fill(mask == 0, -1e9)
        
        # Get probabilities
        probabilities = F.softmax(scores, dim=-1)
        
        return probabilities, context
    
    def forward(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[List[torch.Tensor], torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Encode input
        encoder_output = self.encode(x)  # (batch_size, seq_len, hidden_dim)
        
        # Initialize decoder
        decoder_input = self.decoder_start.expand(batch_size, 1, -1)  # (batch_size, 1, hidden_dim)
        
        # Track visited nodes
        mask = torch.ones(batch_size, seq_len, device=device)  # 1 for available, 0 for visited
        
        selected_indices = []
        log_probs = []
        
        for step in range(seq_len):
            probabilities, context = self.decode_step(decoder_input, encoder_output, mask)
            
            if deterministic:
                selected_idx = torch.argmax(probabilities, dim=-1)
            else:
                selected_idx = torch.multinomial(probabilities, 1).squeeze(-1)
            
            log_prob = torch.log(probabilities.gather(1, selected_idx.unsqueeze(-1)).squeeze(-1) + 1e-10)
            
            selected_indices.append(selected_idx)
            log_probs.append(log_prob)
            
            mask.scatter_(1, selected_idx.unsqueeze(-1), 0)
            
            selected_features = encoder_output.gather(1, selected_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.hidden_dim))
            decoder_input = selected_features
        
        total_log_prob = torch.stack(log_probs, dim=1).sum(dim=1)  # (batch_size,)
        
        return selected_indices, total_log_prob

class NeuralTSPOptimizer:    
    def __init__(self, input_dim: int, hidden_dim: int = 128, learning_rate: float = 1e-4, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.network = PointerNetwork(input_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        self.baseline_alpha = 0.95
        self.baseline = None
        
    def compute_tour_length(self, x: torch.Tensor, tour_indices: List[torch.Tensor], 
                           weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        device = x.device
        
        if weights is None:
            weights = torch.ones(self.input_dim, device=device)
        
        total_length = torch.zeros(batch_size, device=device)
        
        for i in range(len(tour_indices)):
            current_idx = tour_indices[i]
            next_idx = tour_indices[(i + 1) % len(tour_indices)]
            
            current_nodes = x.gather(1, current_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.input_dim)).squeeze(1)
            next_nodes = x.gather(1, next_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.input_dim)).squeeze(1)
            
            distances = torch.sum(weights * torch.abs(current_nodes - next_nodes), dim=-1)
            total_length += distances
        
        return total_length
    
    def train_step(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None, 
                   n_samples: int = 1) -> Dict[str, float]:
        self.network.train()
        all_tours = []
        all_log_probs = []
        all_lengths = []
        
        for _ in range(n_samples):
            tour_indices, log_probs = self.network(x, deterministic=False)
            tour_length = self.compute_tour_length(x, tour_indices, weights)
            
            all_tours.append(tour_indices)
            all_log_probs.append(log_probs)
            all_lengths.append(tour_length)
        
        all_log_probs = torch.stack(all_log_probs, dim=0)  # (n_samples, batch_size)
        all_lengths = torch.stack(all_lengths, dim=0)  # (n_samples, batch_size)
        
        best_indices = torch.argmin(all_lengths, dim=0)
        best_lengths = all_lengths.gather(0, best_indices.unsqueeze(0)).squeeze(0)
        best_log_probs = all_log_probs.gather(0, best_indices.unsqueeze(0)).squeeze(0)
        
        if self.baseline is None:
            self.baseline = best_lengths.mean().item()
        else:
            self.baseline = self.baseline_alpha * self.baseline + (1 - self.baseline_alpha) * best_lengths.mean().item()
        
        advantage = self.baseline - best_lengths.detach()
        
        loss = -(best_log_probs * advantage).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'avg_length': best_lengths.mean().item(),
            'baseline': self.baseline,
            'advantage': advantage.mean().item()
        }
    
    def optimize(self, x: np.ndarray, weights: Optional[np.ndarray] = None, 
                 n_epochs: int = 100, batch_size: int = 32) -> np.ndarray:
        x_tensor = torch.FloatTensor(x).to(self.device)
        n_samples, n_features = x.shape
        if weights is None:
            weights = np.ones(n_features, dtype=float)
        weights_tensor = torch.FloatTensor(np.array(weights, dtype=float)).to(self.device)
        
        n_samples, n_features = x.shape
        
        print(f"Training neural TSP optimizer for {n_epochs} epochs...")
        
        for epoch in range(n_epochs):
            x_batch = x_tensor.unsqueeze(0)
            
            metrics = self.train_step(x_batch, weights_tensor, n_samples=5)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={metrics['loss']:.4f}, "
                      f"Avg Length={metrics['avg_length']:.4f}, "
                      f"Baseline={metrics['baseline']:.4f}")
        
        self.network.eval()
        with torch.no_grad():
            x_batch = x_tensor.unsqueeze(0)
            tour_indices, _ = self.network(x_batch, deterministic=True)
            
            tour = [idx.cpu().numpy()[0] for idx in tour_indices]
        
        ranks = np.zeros(n_samples, dtype=int)
        ranks[tour] = np.arange(n_samples)
        
        return ranks.reshape(-1, 1)

def create_neural_optimizer(input_dim: int, **kwargs) -> NeuralTSPOptimizer:
    return NeuralTSPOptimizer(input_dim, **kwargs)

def neural_multi_scale_optimize(x: np.ndarray, 
                               weights: Optional[np.ndarray] = None,
                               n_epochs: int = 100,
                               hidden_dim: int = 128,
                               learning_rate: float = 1e-4,
                               device: str = 'cuda') -> np.ndarray:
    optimizer = NeuralTSPOptimizer(
        input_dim=x.shape[1],
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        device=device
    )

    return optimizer.optimize(x, weights, n_epochs=n_epochs)