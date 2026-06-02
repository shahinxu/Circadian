"""Compare Tempo cell-phase predictions against CYCLOPS2 reference phases.

This script is intentionally lightweight: it reads only metadata from the H5AD
file and a Tempo cell posterior table, then evaluates sample-level phase
agreement after aggregating cells by projid and major cell class.

Default evaluation is against the overlapping 409-sample CYCLOPS2 reference
table used by the AD223 analysis.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import torch


try:
    from tempo.cell_posterior import ThetaPosteriorDist
except Exception:  # pragma: no cover - optional dependency resolution
    ThetaPosteriorDist = None


TWO_PI = 2.0 * np.pi


def decode_value(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "ignore")
    return value


def circular_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = np.angle(np.exp(1j * (a - b)))
    return delta


def circular_mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(circular_difference(a, b))))


def circular_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sin_a = np.sin(a - np.mean(a))
    sin_b = np.sin(b - np.mean(b))
    numerator = np.sum(sin_a * sin_b)
    denominator = np.sqrt(np.sum(sin_a**2) * np.sum(sin_b**2))
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def normalize_major_class(value: str) -> str:
    mapping = {
        "Vasc_Epithelia": "Vasc",
        "Vas": "Vasc",
        "Mic": "Mic_Immune",
        "Micro": "Mic_Immune",
        "Opc": "OPC",
        "Olig": "Oli",
    }
    return mapping.get(value, value)


def read_obs_columns(h5ad_path: Path, columns: Iterable[str]) -> pd.DataFrame:
    with h5py.File(h5ad_path, "r") as handle:
        obs = handle["obs"]
        data = {}
        for column in columns:
            if column not in obs:
                continue
            item = obs[column]
            if isinstance(item, h5py.Group):
                if item.attrs.get("encoding-type") != "categorical":
                    continue
                categories = item["categories"][:]
                categories = [decode_value(v) for v in categories]
                codes = item["codes"][:]
                values = [categories[code] if code >= 0 else np.nan for code in codes]
                data[column] = values
                continue
            values = item[:]
            if values.dtype.kind in {"S", "O"}:
                values = [decode_value(v) for v in values]
            data[column] = values

        if "index" in obs:
            index_values = obs["index"][:]
            data["barcode"] = [decode_value(v) for v in index_values]
        elif "Barcodes" in obs:
            barcode_values = obs["Barcodes"][:]
            data["barcode"] = [decode_value(v) for v in barcode_values]
        else:
            raise KeyError("Could not find a barcode/index column in obs")

    df = pd.DataFrame(data)
    if "barcode" not in df.columns:
        raise KeyError("barcode column is required")
    return df


def load_tempo_cell_phases(cell_posterior_path: Path) -> pd.DataFrame:
    posterior_df = pd.read_table(cell_posterior_path, sep="\t", index_col=0)
    posterior_df = posterior_df.apply(pd.to_numeric, errors="coerce")
    if ThetaPosteriorDist is not None:
        posterior = ThetaPosteriorDist(torch.tensor(posterior_df.to_numpy(dtype=float)))
        phases = posterior.map_phase.detach().cpu().numpy()
    else:
        map_index = posterior_df.to_numpy(dtype=float).argmax(axis=1)
        num_bins = posterior_df.shape[1]
        phases = (map_index / num_bins) * TWO_PI
    return pd.DataFrame({"barcode": posterior_df.index.astype(str), "tempo_phase": phases})


def aggregate_phases(cell_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, grp in cell_df.groupby(group_cols, dropna=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        phase = np.angle(np.mean(np.exp(1j * grp["tempo_phase"].to_numpy(dtype=float)))) % TWO_PI
        row = dict(zip(group_cols, keys))
        row["n_cells"] = int(len(grp))
        row["tempo_phase"] = float(phase)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(tempo_grouped: pd.DataFrame, cyclops_df: pd.DataFrame) -> pd.DataFrame:
    merged = tempo_grouped.merge(cyclops_df, on="projid", how="inner")
    if merged.empty:
        raise ValueError("No overlapping projid values between Tempo and CYCLOPS tables")

    summary_rows = []
    group_columns = [col for col in merged.columns if col not in {"projid", "tempo_phase", "CYCLOPS_phase", "n_cells"}]
    if not group_columns:
        group_columns = ["all"]
        merged["all"] = "all"

    for keys, grp in merged.groupby(group_columns, dropna=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_samples"] = int(len(grp))
        row["circular_corr"] = circular_correlation(grp["tempo_phase"].to_numpy(), grp["CYCLOPS_phase"].to_numpy())
        row["circular_mae_rad"] = circular_mae(grp["tempo_phase"].to_numpy(), grp["CYCLOPS_phase"].to_numpy())
        row["circular_mae_hours"] = row["circular_mae_rad"] * 12.0 / np.pi
        summary_rows.append(row)

    return pd.DataFrame(summary_rows).sort_values([c for c in group_columns if c != "all"])


@dataclass
class CompareInputs:
    h5ad_path: Path
    cell_posterior_path: Path
    cyclops_path: Path
    out_path: Path


def run(inputs: CompareInputs) -> pd.DataFrame:
    obs = read_obs_columns(
        inputs.h5ad_path,
        columns=["projid", "ADdiag2types", "ADdiag3types", "RNA.Class.Jun30_2024", "major.celltype.0"],
    )
    obs["projid"] = pd.to_numeric(obs["projid"], errors="coerce").astype("Int64")
    obs = obs.dropna(subset=["projid"])
    obs["projid"] = obs["projid"].astype(int)
    obs["ad_group"] = obs["ADdiag2types"].astype(str)
    obs["major_celltype"] = obs["RNA.Class.Jun30_2024"].astype(str).map(normalize_major_class)

    tempo_cell = load_tempo_cell_phases(inputs.cell_posterior_path)
    merged = obs.merge(tempo_cell, on="barcode", how="inner")
    if merged.empty:
        raise ValueError("No overlapping barcodes between the H5AD obs table and Tempo output")

    tempo_sample_level = aggregate_phases(merged, ["projid", "ad_group", "major_celltype"])

    cyclops_df = pd.read_csv(inputs.cyclops_path)
    cyclops_df["projid"] = pd.to_numeric(cyclops_df["projid"], errors="coerce").astype("Int64")
    cyclops_df = cyclops_df.dropna(subset=["projid"])
    cyclops_df["projid"] = cyclops_df["projid"].astype(int)

    summary = evaluate(tempo_sample_level, cyclops_df)
    inputs.out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(inputs.out_path, index=False)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Tempo output against CYCLOPS2 phases")
    parser.add_argument("--h5ad", required=True, type=Path, help="Path to ad427_compressed.h5ad")
    parser.add_argument("--cell-posterior", required=True, type=Path, help="Tempo tempo_results/opt/cell_posterior.tsv")
    parser.add_argument("--cyclops", default=Path("Circadian/02_AD223/Cyclops2/ROSMAP_CYCLOPS_phases_409.csv"), type=Path)
    parser.add_argument("--out", default=Path("Circadian/tempo_vs_cyclops_summary.csv"), type=Path)
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    summary = run(
        CompareInputs(
            h5ad_path=args.h5ad,
            cell_posterior_path=args.cell_posterior,
            cyclops_path=args.cyclops,
            out_path=args.out,
        )
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()