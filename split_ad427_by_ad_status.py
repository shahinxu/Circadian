import os
import time
import numpy as np
import anndata as ad

INPUT = "ad427_compressed.h5ad"
LABEL_COL = "ADdiag2types"
OUT_AD = "ad427_AD.h5ad"
OUT_NONAD = "ad427_nonAD.h5ad"


def main():
    t0 = time.time()
    if not os.path.exists(INPUT):
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    print(f"Loading (backed): {INPUT}")
    adata = ad.read_h5ad(INPUT, backed="r")

    if LABEL_COL not in adata.obs.columns:
        raise KeyError(f"Column not found in obs: {LABEL_COL}")

    labels = adata.obs[LABEL_COL].astype(str).to_numpy()
    is_ad = labels == "AD"
    is_nonad = labels == "nonAD"

    print("Total cells:", adata.n_obs)
    print("AD cells:", int(is_ad.sum()))
    print("nonAD cells:", int(is_nonad.sum()))
    print("Other/unknown cells:", int((~(is_ad | is_nonad)).sum()))

    tasks = [(OUT_AD, is_ad, "AD"), (OUT_NONAD, is_nonad, "nonAD")]

    for out_path, mask, name in tasks:
        n = int(mask.sum())
        if n == 0:
            print(f"Skip {name}: 0 cells")
            continue

        print(f"Writing {name} -> {out_path} (cells={n})")
        t1 = time.time()
        # Subset by rows (cells), keep all genes.
        sub = adata[mask, :]
        sub.write_h5ad(out_path)
        dt = time.time() - t1
        print(f"Done {name}: {out_path} in {dt/60:.1f} min")

    try:
        adata.file.close()
    except Exception:
        pass

    print(f"All done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
