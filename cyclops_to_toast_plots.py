import math
import itertools
import pandas as pd

def circular_correlation(angles1, angles2):
    """Jammalamadaka–Sarma 圆形相关系数，angles* 为弧度数组/序列"""
    angles1 = list(angles1)
    angles2 = list(angles2)
    if len(angles1) != len(angles2):
        raise ValueError("angles1 and angles2 must have the same length.")
    n = len(angles1)
    if n < 2:
        return float("nan")

    def circular_mean(arr):
        sum_sin = sum(math.sin(a) for a in arr)
        sum_cos = sum(math.cos(a) for a in arr)
        return math.atan2(sum_sin, sum_cos)

    mean1 = circular_mean(angles1)
    mean2 = circular_mean(angles2)

    num = 0.0
    den1 = 0.0
    den2 = 0.0
    for a1, a2 in zip(angles1, angles2):
        s1 = math.sin(a1 - mean1)
        s2 = math.sin(a2 - mean2)
        num += s1 * s2
        den1 += s1 * s1
        den2 += s2 * s2

    den = math.sqrt(den1 * den2)
    if den == 0:
        return float("nan")
    return num / den

# 1. 读入文件
path = "/playpen-shared/zhenx/Circadian/CYCLOPS-2.0/results/Circadian_genes/Zhang_CancerCell_2025_CD4_CD8_Myeloid/predicted_phase.csv"
df = pd.read_csv(path)

# 2. 提取 sample ID（去掉最后一个 . 后面的 celltype），以及 celltype
#    也可以直接用 Covariate_D 这一列作为 celltype
df["sample"] = df["ID"].str.rsplit(".", n=1).str[0]
df["celltype"] = df["Covariate_D"]

# 3. 用 Phase 列做角度（弧度）；如果想用 Phases_MA，把下面 "Phase" 改成 "Phases_MA"
pivot = df.pivot_table(
    index="sample",
    columns="celltype",
    values="Phase",
    aggfunc="mean"
)

celltypes = list(pivot.columns)

results = []

for ct1, ct2 in itertools.combinations(celltypes, 2):
    s1 = pivot[ct1]
    s2 = pivot[ct2]
    mask = s1.notna() & s2.notna()
    angles1 = s1[mask].to_numpy()
    angles2 = s2[mask].to_numpy()

    if len(angles1) < 2:
        corr = float("nan")
    else:
        corr = circular_correlation(angles1, angles2)

    results.append({
        "celltype1": ct1,
        "celltype2": ct2,
        "n_common_samples": len(angles1),
        "circular_corr": corr,
    })

res_df = pd.DataFrame(results)
print(res_df)