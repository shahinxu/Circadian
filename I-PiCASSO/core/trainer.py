from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from .data import Dataset
from .features import compute_pca_components
from .align import to_24, best_shift
from .plots import phase_scatter, component_expression_scatter
from .models import (
    Autoencoder,
    compute_component_weights,
    smoothness_loss,
)

__all__ = [
    "EndToEndConfig",
    "EndToEndTrainer",
]


@dataclass
class EndToEndConfig:
    steps: int = 1000
    n_components: int = 8
    ordering_hidden: int = 128
    pointer_heads: int = 8
    pointer_layers: int = 3
    pointer_dropout: float = 0.1
    decoder_hidden: int = 512
    decoder_dropout: float = 0.1
    device: str = 'auto'
    lr: float = 1e-3
    weight_decay: float = 1e-5
    decay: float = 0.85
    log_interval: int = 10
    smooth_bins: int = 20
    reconstruction_weight: float = 1.0


class EndToEndTrainer:
    def __init__(self, ds: Dataset, cfg: EndToEndConfig, out_dir: str):
        self.ds = ds
        self.cfg = cfg
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def run(self):
        X = self.ds.X
        components_np = compute_pca_components(X, self.cfg.n_components)

        device_str = (self.cfg.device or 'auto').lower()
        if device_str == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            try:
                device = torch.device(device_str)
            except (RuntimeError, ValueError) as exc:
                raise ValueError(f"Invalid device string '{self.cfg.device}': {exc}") from exc

        components = torch.tensor(components_np, dtype=torch.float32, device=device)

        model = Autoencoder(
            n_components=self.cfg.n_components,
            ordering_hidden=self.cfg.ordering_hidden,
            pointer_layers=self.cfg.pointer_layers,
            pointer_dropout=self.cfg.pointer_dropout,
            decoder_hidden=self.cfg.decoder_hidden,
            decoder_dropout=self.cfg.decoder_dropout,
        ).to(device)

        component_weights = compute_component_weights(self.cfg.n_components, decay=self.cfg.decay, device=device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

        history = []

        for step in range(1, self.cfg.steps + 1):
            model.train()
            out = model(components)
            loss_smooth = smoothness_loss(
                sorted_components=out['sorted_components'],
                component_weights=component_weights,
                n_bins=self.cfg.smooth_bins,
            )
            if torch.isnan(loss_smooth).any() or torch.isinf(loss_smooth).any():
                raise ValueError("smoothness_loss returned NaN/Inf; check sorted outputs")
            total_loss = loss_smooth

            loss_recon = torch.zeros_like(loss_smooth)
            if self.cfg.reconstruction_weight > 0.0:
                reconstructed_original = out['reconstructed_original']
                target_components = components.detach()
                loss_recon = F.mse_loss(reconstructed_original, target_components)
                if torch.isnan(loss_recon).any() or torch.isinf(loss_recon).any():
                    raise ValueError("Reconstruction loss returned NaN/Inf; aborting training")
                total_loss = total_loss + self.cfg.reconstruction_weight * loss_recon

            if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                raise ValueError("Total loss returned NaN/Inf; aborting training")

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if step % self.cfg.log_interval == 0:
                log_entry = {
                    'step': step,
                    'total_loss': float(total_loss.detach().cpu().item()),
                    'loss_smooth': float(loss_smooth.detach().cpu().item()),
                }
                if self.cfg.reconstruction_weight > 0.0:
                    log_entry['loss_recon'] = float(loss_recon.detach().cpu().item())

                history.append(log_entry)
                pd.DataFrame(history).to_csv(os.path.join(self.out_dir, 'history_metrics.csv'), index=False)

        model.eval()
        with torch.no_grad():
            outputs = model(components)
            scores_np = outputs['scores'].cpu().numpy()
            components_np = outputs['sorted_components'].cpu().numpy()

        order = np.argsort(scores_np)
        ranks = np.zeros_like(order)
        ranks[order] = np.arange(len(order))
        ranks = ranks.astype(int)

        pd.DataFrame({'study_sample': self.ds.meta_joined['study_sample'].values, 'Rank': ranks}).to_csv(
            os.path.join(self.out_dir, 'final_ranks.csv'), index=False
        )

        # y_hours is retrieved after training for evaluation only, keeping the optimisation unsupervised.
        y = self.ds.y_hours.astype(float) % 24.0
        x0 = to_24(ranks)
        res = best_shift(x0, y, step=0.05)
        summary = pd.DataFrame([{
            'n': int(len(y)),
            'pearson_r': float(res.corr) if res.corr is not None else np.nan,
            'r_square': float(res.r2) if res.r2 is not None else np.nan,
            'slope': float(res.slope) if res.slope is not None else np.nan,
            'shift': float(res.shift) if res.shift is not None else np.nan,
            'orientation': res.orientation,
        }])
        summary.to_csv(os.path.join(self.out_dir, 'ipicasso_compare_summary.csv'), index=False)

        x_use = x0 if res.orientation == 'normal' else (24.0 - x0) % 24.0
        x_shift = (x_use + (res.shift if res.shift is not None else 0.0)) % 24.0
        phase_scatter(
            x_shift,
            y,
            f"r={res.corr:.3f}, shift={res.shift:.2f}h, {res.orientation}",
            os.path.join(self.out_dir, 'ipicasso_rank_vs_time.png'),
        )

        if components_np.ndim == 2 and components_np.shape[0] == ranks.shape[0]:
            for idx in range(components_np.shape[1]):
                component_expression_scatter(
                    ranks,
                    components_np[:, idx],
                    title=f"PC {idx + 1}",
                    out_png=os.path.join(self.out_dir, f"component_pc{idx + 1:02d}_expression_vs_rank.png"),
                    color='tab:blue',
                )
