import h5py
import numpy as np

p = "ad427_compressed.h5ad"
keywords = [
    "ad", "dx", "diagn", "braak", "cerad", "cog", "dement", "path",
    "pmad", "nrad", "nci", "mci", "control", "alz"
]

with h5py.File(p, "r") as f:
    obs = f["obs"]
    cols = list(obs.keys())
    print("obs_col_count", len(cols))

    hit = [c for c in cols if any(k in c.lower() for k in keywords)]
    print("candidate_cols", hit)

    def read_col(name, maxn=20):
        x = obs[name]

        if isinstance(x, h5py.Group) and "codes" in x and "categories" in x:
            codes = x["codes"][()]
            cats = x["categories"][()]
            cats = [
                c.decode() if isinstance(c, (bytes, bytearray, np.bytes_)) else str(c)
                for c in cats
            ]
            vals = []
            for i in np.unique(codes[:300000]):
                if 0 <= i < len(cats):
                    vals.append(cats[i])
                else:
                    vals.append(str(i))
            return vals[:maxn], "categorical"

        arr = x[()]
        if hasattr(arr, "dtype") and arr.dtype.kind in ("S", "O", "U"):
            arr = [
                a.decode() if isinstance(a, (bytes, bytearray, np.bytes_)) else str(a)
                for a in arr[:300000]
            ]
            u = sorted(set(arr))
            return u[:maxn], "string"

        a = np.asarray(arr)
        if np.issubdtype(a.dtype, np.number):
            vals = [f"min={np.nanmin(a):.4g}", f"max={np.nanmax(a):.4g}"]
            if np.issubdtype(a.dtype, np.floating):
                vals.append(f"nan={int(np.isnan(a).sum())}")
            return vals, "numeric"

        return [str(type(arr))], "other"

    for c in hit:
        try:
            vals, typ = read_col(c)
            print(f"{c} [{typ}] -> {vals}")
        except Exception as e:
            print(f"{c} [error] -> {e}")

    for col in ["ADdiag2types", "ADdiag3types"]:
        if col in obs and isinstance(obs[col], h5py.Group) and "codes" in obs[col] and "categories" in obs[col]:
            g = obs[col]
            codes = g["codes"][()]
            cats = [
                c.decode() if isinstance(c, (bytes, bytearray, np.bytes_)) else str(c)
                for c in g["categories"][()]
            ]
            print("\\ncounts", col)
            for i, cat in enumerate(cats):
                print(cat, int((codes == i).sum()))
            print("missing_code(-1)", int((codes == -1).sum()))
