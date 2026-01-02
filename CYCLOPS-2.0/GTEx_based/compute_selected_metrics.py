#!/usr/bin/env python3
"""Compute circadian alignment metrics for selected CYCLOPS runs.

The metrics align with the TOAST-3.0 summary columns so the two
approaches can be compared directly.
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_CSV = RESULTS_DIR / "jr_cor_selected.csv"

TARGET_RESULT_DIRS = [
    "GSE54651_adrenal_gland_20251204_191257",
    "GSE54651_aorta_20251204_191614",
    "GSE54651_atria_20251204_191926",
    "GSE54651_bladder_20251204_192248",
    "GSE54651_brain_stem_20251204_192553",
    "GSE54651_brown_fat_20251204_192901",
    "GSE54651_cerebellum_20251204_193211",
    "GSE54651_colon_20251204_193521",
    "GSE54651_heart_20251204_193829",
    "GSE54651_hypothalamus_20251204_194155",
]


def time_to_phase(time_hours: np.ndarray, period_hours: float = 24.0) -> np.ndarray:
    return 2 * np.pi * time_hours / period_hours


def circular_correlation_jr_signed(alpha: np.ndarray, beta: np.ndarray) -> float:
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    mask = np.isfinite(alpha) & np.isfinite(beta)
    alpha = alpha[mask]
    beta = beta[mask]
    if alpha.size < 2:
        return float("nan")
    alpha_bar = math.atan2(np.mean(np.sin(alpha)), np.mean(np.cos(alpha)))
    beta_bar = math.atan2(np.mean(np.sin(beta)), np.mean(np.cos(beta)))
    sin_alpha = np.sin(alpha - alpha_bar)
    sin_beta = np.sin(beta - beta_bar)
    numerator = np.sum(sin_alpha * sin_beta)
    denominator = math.sqrt(np.sum(sin_alpha ** 2) * np.sum(sin_beta ** 2))
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def circular_correlation_jr_abs(alpha: np.ndarray, beta: np.ndarray) -> float:
    signed = circular_correlation_jr_signed(alpha, beta)
    return float(abs(signed)) if math.isfinite(signed) else signed


def best_align_phase_for_comparison(
    x_rad: np.ndarray,
    y_rad: np.ndarray,
    step: float = 0.1,
) -> tuple[np.ndarray, float, float, float, float, bool]:
    two_pi = 2 * np.pi
    x_arr = np.asarray(x_rad, dtype=float) % two_pi
    y_arr = np.asarray(y_rad, dtype=float) % two_pi
    shifts = np.arange(0.0, two_pi, step, dtype=float)

    best_r = -np.inf
    best_aligned = x_arr
    best_r2 = float("nan")
    best_spearman = float("nan")
    best_shift = 0.0
    best_flipped = False

    for flipped in (False, True):
        x0 = (two_pi - x_arr) % two_pi if flipped else x_arr
        for shift in shifts:
            xs = (x0 + shift) % two_pi
            try:
                r = float(pearsonr(xs, y_arr)[0])
            except Exception:
                r = float("nan")
            if not math.isfinite(r):
                continue
            if r > best_r:
                best_r = r
                best_r2 = r * r
                try:
                    spearman = float(spearmanr(xs, y_arr)[0])
                except Exception:
                    spearman = float("nan")
                best_spearman = spearman
                best_aligned = xs
                best_shift = float(shift)
                best_flipped = flipped
    return best_aligned, best_r, best_r2, best_spearman, best_shift, best_flipped


@dataclass
class MetricRow:
    dataset: str
    timestamp: str
    n_samples: int
    pearson_r: float
    pearson_r2: float
    spearman_r: float
    circular_r_signed: float
    circular_r_abs: float


def parse_dataset_parts(result_dir: Path) -> tuple[str, Path, Path]:
    name = result_dir.name
    tokens = name.split("_")
    if len(tokens) < 3:
        raise ValueError(f"Unexpected directory name: {name}")
    dataset_token = "_".join(tokens[:-2])
    if "_" not in dataset_token:
        raise ValueError(f"Cannot split dataset token: {dataset_token}")
    prefix, tissue = dataset_token.split("_", 1)
    dataset = f"{prefix}/{tissue}"
    metadata_dir = DATA_DIR / prefix / tissue
    return dataset, metadata_dir, dataset_token


def find_fit_output(result_dir: Path) -> Path:
    for entry in sorted(result_dir.glob("Fit_Output_*.csv")):
        return entry
    raise FileNotFoundError(f"No Fit_Output_*.csv found in {result_dir}")


def load_metadata(metadata_dir: Path) -> pd.DataFrame:
    meta_path = metadata_dir / "metadata.csv"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    meta = pd.read_csv(meta_path)
    if "Sample" not in meta.columns:
        raise ValueError(f"metadata missing 'Sample' column: {meta_path}")
    if "Time_Hours" not in meta.columns:
        raise ValueError(f"metadata missing 'Time_Hours' column: {meta_path}")
    meta = meta[["Sample", "Time_Hours"]].copy()
    meta["Time_Hours"] = pd.to_numeric(meta["Time_Hours"], errors="coerce")
    meta.dropna(subset=["Time_Hours"], inplace=True)
    return meta


def compute_metrics(result_dir: Path) -> MetricRow | None:
    dataset, metadata_dir, _ = parse_dataset_parts(result_dir)
    try:
        fit_path = find_fit_output(result_dir)
    except FileNotFoundError as exc:
        print(f"[SKIP] {exc}")
        return None
    timestamp = fit_path.stem.replace("Fit_Output_", "")

    preds = pd.read_csv(fit_path, usecols=["ID", "Phase"])
    preds.rename(columns={"ID": "Sample", "Phase": "Predicted_Phase_Rad"}, inplace=True)
    preds["Predicted_Phase_Rad"] = pd.to_numeric(preds["Predicted_Phase_Rad"], errors="coerce")
    preds.dropna(subset=["Predicted_Phase_Rad"], inplace=True)

    try:
        meta = load_metadata(metadata_dir)
    except FileNotFoundError as exc:
        print(f"[SKIP] {exc}")
        return None

    joined = preds.merge(meta, on="Sample", how="inner")
    joined.dropna(subset=["Predicted_Phase_Rad", "Time_Hours"], inplace=True)
    if joined.empty:
        print(f"[SKIP] No overlapping samples for {dataset}")
        return None

    phase_rad = (joined["Predicted_Phase_Rad"].to_numpy(dtype=float) % (2 * np.pi))
    metadata_hours = np.mod(joined["Time_Hours"].to_numpy(dtype=float), 24.0)
    metadata_rad = time_to_phase(metadata_hours, period_hours=24.0)

    _, pearson_r, pearson_r2, spearman_r, _, _ = best_align_phase_for_comparison(
        phase_rad, metadata_rad, step=0.1
    )
    circ_signed = circular_correlation_jr_signed(phase_rad, metadata_rad)
    circ_abs = circular_correlation_jr_abs(phase_rad, metadata_rad)

    return MetricRow(
        dataset=dataset,
        timestamp=timestamp,
        n_samples=joined.shape[0],
        pearson_r=float(pearson_r),
        pearson_r2=float(pearson_r2),
        spearman_r=float(spearman_r),
        circular_r_signed=float(circ_signed),
        circular_r_abs=float(circ_abs),
    )


def resolve_result_dir(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = RESULTS_DIR / path
    return path


def discover_all_result_dirs() -> List[Path]:
    dirs: List[Path] = []
    for entry in sorted(RESULTS_DIR.iterdir()):
        if entry.is_dir():
            dirs.append(entry)
    return dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute circadian alignment metrics for selected CYCLOPS runs",
    )
    parser.add_argument(
        "dirs",
        nargs="*",
        help="Result directories (relative to results/ or absolute paths).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every subdirectory inside results/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help="Destination CSV path (default: results/jr_cor_selected.csv)",
    )
    return parser.parse_args()


def collect_target_dirs(args: argparse.Namespace) -> List[Path]:
    if args.all:
        return discover_all_result_dirs()
    if args.dirs:
        return [resolve_result_dir(d) for d in args.dirs]
    return [RESULTS_DIR / name for name in TARGET_RESULT_DIRS]


def main() -> int:
    args = parse_args()
    target_dirs = collect_target_dirs(args)

    rows: List[MetricRow] = []
    for result_dir in target_dirs:
        if not result_dir.is_dir():
            print(f"[WARN] Missing directory: {result_dir}")
            continue
        summary = compute_metrics(result_dir)
        if summary is not None:
            rows.append(summary)

    if not rows:
        print("No metrics computed.")
        return 0

    df = pd.DataFrame([asdict(r) for r in rows]).sort_values("dataset")
    output_csv = args.output if args.output.is_absolute() else (BASE_DIR / args.output)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")
    print(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
