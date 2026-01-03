from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    in_path = base_dir / "Cancer_Epi.discoODAres_CS_ANNOATION.outer.tsv"
    out_path = base_dir / "seed_genes.txt"

    genes: set[str] = set()

    with in_path.open("r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in, delimiter="\t")
        for row in reader:
            val = row.get("pvalue")
            if not val or val == "NA":
                continue
            try:
                if float(val) < 0.02:
                    gene_id = row.get("GeneID")
                    if gene_id:
                        genes.add(gene_id)
            except ValueError:
                continue

    with out_path.open("w", encoding="utf-8") as f_out:
        for gene in sorted(genes):
            f_out.write(gene + "\n")


if __name__ == "__main__":
    main()
