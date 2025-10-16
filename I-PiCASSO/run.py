import os
import sys
import argparse
import datetime





# Repo/layout helpers
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CUR_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.data import build_dataset
from core.trainer import EndToEndTrainer, EndToEndConfig


def run_end2end(expression_path: str, metadata_path: str, out_dir: str, cfg: dict):
    ds = build_dataset(expression_path, metadata_path)
    et = EndToEndTrainer(ds, EndToEndConfig(**cfg), out_dir)
    et.run()


def main():
    parser = argparse.ArgumentParser(description='I-PiCASSO: Interpretable and Composable Analysis of Single-Cell States and Oscillations')
    parser.add_argument('--expression', required=False, default=None, 
                        help='Path to expression.csv')
    parser.add_argument('--metadata', required=False, default=None, 
                        help='Path to metadata.csv')
    parser.add_argument('--dataset', required=False, default='GSE146773', 
                        help='Dataset folder under data/')
    parser.add_argument('--out-prefix', required=False, default='ipicasso_result', 
                        help='Output folder prefix under I-PiCASSO/')
    parser.add_argument('--components', type=int, default=8, 
                        help='Number of latent components to project onto')
    parser.add_argument('--steps', type=int, default=2000, 
                        help='Training steps for end-to-end optimization')
    parser.add_argument('--ordering-hidden', type=int, default=1024, 
                        help='Hidden width for the ordering network')
    parser.add_argument('--pointer-layers', type=int, default=3,
                        help='Number of self-attention layers in the pointer ordering network encoder')
    parser.add_argument('--pointer-dropout', type=float, default=0.1,
                        help='Dropout rate inside the pointer ordering network')
    parser.add_argument('--decoder-hidden', type=int, default=512,
                        help='Hidden width for the reconstruction decoder')
    parser.add_argument('--decoder-dropout', type=float, default=0.1,
                        help='Dropout rate applied inside the reconstruction decoder')
    parser.add_argument('--device', default='auto',
                        help="Torch device to use (e.g., 'auto', 'cpu', 'cuda', 'cuda:0')")
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='Learning rate for optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-5, 
                        help='Weight decay for optimizer')
    parser.add_argument('--decay', type=float, default=0.85, 
                        help='Decay factor for component weighting scheme')
    parser.add_argument('--smooth-bins', type=int, default=20,
                        help='Number of bins to average before computing second differences for smoothness loss')
    parser.add_argument('--reconstruction-weight', type=float, default=1,
                        help='Weight for reconstruction MSE loss term in end-to-end training')
    args = parser.parse_args()

    # Resolve default paths relative to repo root
    if args.expression is None or args.metadata is None:
        dataset_dir = os.path.join(REPO_ROOT, 'data', args.dataset)
        expression = os.path.join(dataset_dir, 'expression.csv')
        metadata = os.path.join(dataset_dir, 'metadata.csv')
    else:
        expression = args.expression
        metadata = args.metadata

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(REPO_ROOT, 'I-PiCASSO', f"{args.out_prefix}_{args.dataset}_{ts}")

    end_cfg = {
        'steps': args.steps,
        'n_components': args.components,
        'ordering_hidden': args.ordering_hidden,
        'pointer_layers': args.pointer_layers,
        'pointer_dropout': args.pointer_dropout,
        'decoder_hidden': args.decoder_hidden,
        'decoder_dropout': args.decoder_dropout,
        'device': args.device,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'decay': args.decay,
        'smooth_bins': args.smooth_bins,
        'reconstruction_weight': args.reconstruction_weight,
    }
    run_end2end(
        expression_path=expression,
        metadata_path=metadata,
        out_dir=out_dir,
        cfg=end_cfg,
    )


if __name__ == '__main__':
    main()
