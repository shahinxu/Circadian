import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


def load_pathway_dataset(pathway_csv: str) -> pd.DataFrame:
    return pd.read_csv(pathway_csv)


def build_pathway_map(
    pathway_df: pd.DataFrame,
    gene_symbols: List[str],
    min_pathway_size: int = 100,
    max_pathway_size: int = 500
) -> Tuple[List[List[int]], List[str], Dict[str, int]]:
    
    gene_to_idx = {gene.upper(): idx for idx, gene in enumerate(gene_symbols)}
    
    pathway_groups = pathway_df.groupby('pathway_name')
    pathway_indices = []
    pathway_names = []
    
    for pathway_name, group in pathway_groups:
        genes_in_pathway = group['gene_symbol'].str.upper().unique()
        indices = [gene_to_idx[g] for g in genes_in_pathway if g in gene_to_idx]
        
        if min_pathway_size <= len(indices) <= max_pathway_size:
            pathway_indices.append(indices)
            pathway_names.append(pathway_name)
    
    print(f"Built pathway map: {len(pathway_names)} pathways covering {len(gene_symbols)} genes")
    print(f"  Pathway size range: [{min([len(p) for p in pathway_indices])}, {max([len(p) for p in pathway_indices])}]")
    
    return pathway_indices, pathway_names, gene_to_idx


def get_pathway_statistics(pathway_indices: List[List[int]], pathway_names: List[str]) -> pd.DataFrame:
    stats = pd.DataFrame({
        'pathway_name': pathway_names,
        'num_genes': [len(idx) for idx in pathway_indices]
    })
    return stats.sort_values('num_genes', ascending=False)
