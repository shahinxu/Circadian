from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

__all__ = [
    "ReconstructionDecoder",
    "Autoencoder",
    "compute_component_weights",
    "smoothness_loss",
]


def _ensure_finite(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError(f"{name} 出现 NaN 或 Inf，请检查上游计算是否稳定")
    return tensor


def _bin_sorted_components(sorted_components: torch.Tensor, n_bins: int) -> torch.Tensor:
    if n_bins <= 1:
        return sorted_components
    n_samples = sorted_components.size(0)
    if n_samples <= n_bins:
        return sorted_components
    base = n_samples // n_bins
    remainder = n_samples % n_bins
    bin_sizes = [base + (1 if i < remainder else 0) for i in range(n_bins)]
    segments = []
    start = 0
    for size in bin_sizes:
        if size <= 0:
            continue
        end = start + size
        segment = sorted_components[start:end].mean(dim=0, keepdim=True)
        segments.append(segment)
        start = end
    if not segments:
        return sorted_components
    return torch.cat(segments, dim=0)


def compute_component_weights(n_components: int, decay: float = 0.85, device: Optional[torch.device] = None) -> torch.Tensor:
    bases = torch.arange(n_components, dtype=torch.float32, device=device)
    weights = torch.pow(decay, bases)
    weights = weights / (weights.sum() + 1e-12)
    return weights


def smoothness_loss(
    sorted_components: torch.Tensor,
    component_weights: torch.Tensor,
    n_bins: int = 20,
) -> torch.Tensor:
    # Robust version: sanitize NaN/Inf, clamp extreme diffs and use Huber-like reduction
    binned = _bin_sorted_components(sorted_components, n_bins)
    if binned.size(0) < 3:
        return torch.zeros((), device=sorted_components.device, dtype=sorted_components.dtype)

    # sanitize non-finite values (best-effort; presence of non-finite indicates upstream instability)
    if not torch.isfinite(binned).all():
        # replace NaN with 0 and clamp Infs to large finite values
        binned = torch.nan_to_num(binned, nan=0.0, posinf=1e6, neginf=-1e6)

    # compute second differences
    second_diffs = binned[2:] - 2.0 * binned[1:-1] + binned[:-2]

    # clamp extremely large values to avoid producing Inf in gradients
    second_diffs = torch.clamp(second_diffs, min=-1e6, max=1e6)

    # Huber-like reduction (smooth L1) to limit gradient spikes from outliers
    huber_delta = 1e-3
    abs_sd = second_diffs.abs()
    if huber_delta <= 0.0:
        per_component = abs_sd.sum(dim=0)
    else:
        small_mask = abs_sd <= huber_delta
        robust = torch.empty_like(abs_sd)
        robust[small_mask] = 0.5 * (second_diffs[small_mask] ** 2) / (huber_delta + 1e-12)
        robust[~small_mask] = abs_sd[~small_mask] - 0.5 * huber_delta
        per_component = robust.sum(dim=0)

    # guard weights
    if not torch.isfinite(component_weights).all():
        component_weights = torch.nan_to_num(component_weights, nan=0.0, posinf=1.0, neginf=0.0)

    weights = component_weights / (component_weights.sum() + 1e-12)
    loss = torch.dot(weights, per_component)
    return loss


class ReconstructionDecoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        recon = self.net(z)
        return recon



def _sinkhorn(logits: torch.Tensor, n_iters: int = 20, eps: float = 1e-6) -> torch.Tensor:
    logits = torch.clamp(logits, min=-50.0, max=50.0)

    log_matrix = logits
    for _ in range(n_iters):
        # normalize rows in log-space
        log_matrix = log_matrix - torch.logsumexp(log_matrix, dim=-1, keepdim=True)
        # normalize columns in log-space
        log_matrix = log_matrix - torch.logsumexp(log_matrix, dim=-2, keepdim=True)

    matrix = torch.exp(log_matrix)
    matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + eps)
    matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    return matrix


class SinkhornOrderingNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        sinkhorn_iters: int = 30,
        sinkhorn_tau: float = 0.1,
    ):
        super().__init__()
        layers = []
        last_dim = input_dim
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(last_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, hidden_dim))
        self.embed = nn.Sequential(*layers)
        self.score_head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_tau = sinkhorn_tau

    def forward(self, components: torch.Tensor) -> dict:
        if components.dim() != 2:
            raise ValueError("SinkhornOrderingNet expects components with shape (n_samples, n_components)")
        seq_len = components.size(0)
        if seq_len == 0:
            raise ValueError("SinkhornOrderingNet requires at least one sample")
        _ensure_finite(components, "输入组件")
        embed = self.embed(components)
        _ensure_finite(embed, "嵌入")
        embed = self.dropout(embed)
        embed = _ensure_finite(embed, "Sinkhorn 嵌入")
        raw_scores = self.score_head(embed).squeeze(-1)
        raw_scores = _ensure_finite(raw_scores, "Sinkhorn 排序得分")
        raw_scores = raw_scores - raw_scores.mean()

        device = components.device
        dtype = components.dtype
        positions = torch.linspace(-1.0, 1.0, steps=seq_len, device=device, dtype=dtype)
        logits = -((raw_scores.unsqueeze(-1) - positions.unsqueeze(0)) ** 2) / self.sinkhorn_tau

        perm_matrix = _sinkhorn(logits, n_iters=self.sinkhorn_iters)
        perm_matrix = _ensure_finite(perm_matrix, "Sinkhorn 排列矩阵")

        positions_idx = torch.arange(seq_len, device=device, dtype=dtype)
        expected_ranks = perm_matrix.transpose(0, 1) @ positions_idx
        expected_ranks = _ensure_finite(expected_ranks, "Sinkhorn 期望排序")

        return {
            'perm_matrix': perm_matrix,
            'logits': logits,
            'expected_ranks': expected_ranks,
            'attention_rows': perm_matrix,
            'raw_scores': raw_scores,
        }


class Autoencoder(nn.Module):
    def __init__(
        self,
        n_components: int = 8,
        ordering_hidden: int = 128,
        pointer_layers: int = 3,
        pointer_dropout: float = 0.1,
        decoder_hidden: int = 128,
        decoder_dropout: float = 0.1,
        sinkhorn_iters: int = 30,
        sinkhorn_tau: float = 0.1,
    ):
        super().__init__()
        self.encoder_net = SinkhornOrderingNet(
            input_dim=n_components,
            hidden_dim=ordering_hidden,
            n_layers=pointer_layers,
            dropout=pointer_dropout,
            sinkhorn_iters=sinkhorn_iters,
            sinkhorn_tau=sinkhorn_tau,
        )
        self.n_components = n_components
        self.ordering_mode = 'sinkhorn'
        self.decoder_hidden = decoder_hidden
        self.decoder_dropout = decoder_dropout
        self._decoder_input_dim: Optional[int] = None
        self.decoder: Optional[ReconstructionDecoder] = None

    def _validate_input(self, components: torch.Tensor) -> None:
        if components.dim() != 2:
            raise ValueError("SelectorPointerModel expects components with shape (n_samples, n_components)")
        if components.size(1) != self.n_components:
            raise ValueError(
                f"Expected components with {self.n_components} columns, got {components.size(1)}"
            )
        _ensure_finite(components, "模型输入组件")

    def encode(self, components: torch.Tensor) -> dict:
        """Encode inputs into a soft permutation latent representation."""
        pointer_out = self.encoder_net(components)
        perm = pointer_out['perm_matrix']
        scores = pointer_out['expected_ranks']
        scores = _ensure_finite(scores, "排序分数")
        return {
            'components': components,
            'perm_matrix': perm,
            'scores': scores,
            'pointer_logits': pointer_out['logits'],
            'pointer_attention': pointer_out['attention_rows'],
        }

    def decode(self, latent: dict) -> dict:
        """Decode sorted components back to the original ordering."""
        perm = latent['perm_matrix']
        components = latent['components']
        if self.decoder is None or self._decoder_input_dim != perm.size(-1):
            self.decoder = ReconstructionDecoder(
                input_dim=perm.size(-1),
                hidden_dim=self.decoder_hidden,
                output_dim=self.n_components,
                dropout=self.decoder_dropout,
            ).to(perm.device)
            self._decoder_input_dim = perm.size(-1)

        sorted_components = perm @ components
        sorted_components = _ensure_finite(sorted_components, "排序后组分")

        reconstructed_sorted = self.decoder(perm)
        reconstructed_sorted = _ensure_finite(reconstructed_sorted, "decoder 重建输出")
        reconstructed_original = perm.transpose(0, 1) @ reconstructed_sorted
        reconstructed_original = _ensure_finite(reconstructed_original, "重建后的原始顺序组件")
        return {
            'sorted_components': sorted_components,
            'reconstructed_components': reconstructed_sorted,
            'reconstructed_original': reconstructed_original,
        }

    def forward(self, components: torch.Tensor) -> dict:
        self._validate_input(components)
        latent = self.encode(components)
        decoded = self.decode(latent)
        outputs = {**latent, **decoded}
        return outputs
