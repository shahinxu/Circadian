# I-PiCASSO (separate runner)

This folder hosts a self-contained runner built around the `SelectorPointerModel`, a fully differentiable stack that no longer depends on the legacy PiCASSO pipeline. The workflow learns:

1. **Gene selection weights** through a neural selector fed with per-gene statistics.
2. **Latent eigengenes** via a differentiable PCA layer (implemented with `torch.linalg.svd`) applied after weighting genes with the selector; the resulting trajectories are smoothed by binning the ordered samples（默认20段）再计算二阶差分，并内置振幅奖励，持续推动主成分覆盖更大的取值范围。
3. **Sample ordering** through a neural pointer network that produces a soft permutation over samples.

The loss optimizes weighted smoothness of eigengene trajectories under the learned ordering, while自动奖励高熵选择器与大振幅主成分，因此整个栈（selector + projector + ordering）在保持平滑的同时也会鼓励「选得多」与「摆得开」；可选的直通估计（STE）可以让选择器在前向中输出0/1权重，同时保留可导梯度。

## Usage

Invoke the runner from the repo root:

```bash
python I-PiCASSO/run.py --dataset GSE146773 \
	--steps 1500 \
	--components 8 \
	--temperature 0.8 \
	--selector-hidden 192 \
	--ordering-hidden 192
```

Key arguments:

| Flag | Purpose |
| --- | --- |
| `--dataset` | Folder under `data/` containing `expression.csv` and `metadata.csv`; overrides `--expression/--metadata` if provided. |
| `--steps` | Training iterations for the end-to-end optimizer. |
| `--components` | Number of latent components learned by the projector. |
| `--temperature` | Softmax temperature for gene weights (lower sharpens selection). |
| `--pointer-heads`, `--pointer-layers`, `--pointer-dropout` | Architecture knobs for the pointer ordering head. |
| `--selector-hidden`, `--ordering-hidden` | Hidden widths for the selector and ordering networks (the latter becomes the pointer hidden size). |
| `--lr`, `--weight-decay` | Optimizer hyperparameters. |
| `--smooth-bins` | Number of rank bins averaged before computing second differences in the smoothness loss (default 20). |
| `--decay` | Exponential decay applied to component weights (earlier PCs get higher emphasis). |
| `--selector-use-ste` | Enable straight-through estimation so forward-pass selector weights are hard 0/1 while gradients flow through the soft distribution. |
| `--selector-hard-threshold` | Threshold for hard selector weights when STE is active (≤0 uses the mean weight). |
| `--disable-selector` | Bypass the gene-selector network and treat all genes as equally weighted (useful for isolating ordering-module behavior). |
> 注：选择器熵奖励与主成分振幅奖励现已内置且无须配置权重，会自动尝试「越多越好、越大越好」。

If you prefer explicit file paths, pass `--expression` and `--metadata` instead of `--dataset`.

### Dependencies

Install the runner dependencies (scikit-learn is still required for the iterative baseline utilities) with:

```bash
pip install torch scikit-learn pandas numpy matplotlib
```

## Outputs

Results are written to `I-PiCASSO/ipicasso_result_<dataset>_<timestamp>/` and include:

- `final_ranks.csv` — inferred sample ordering.
- `final_selected_genes.csv` — top genes ranked by learned weights.
- `history_metrics.csv` — training snapshots with smoothness、选择器熵奖励、主成分振幅奖励、有效基因数、学习率与 baseline，便于监控训练动态。
- `ipicasso_compare_summary.csv` — correlation/shift metrics against metadata time labels.
- `ipicasso_rank_vs_time.png` — scatter plot of aligned ordering vs. metadata.

The runner already performs comparison/plotting, so invoking `tools/compare_with_metadata.py` is optional for secondary analyses.
