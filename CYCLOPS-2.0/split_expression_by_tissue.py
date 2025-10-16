import pandas as pd
import os

# 路径可根据实际情况修改
expression_path = "expression.csv"
metadata_path = "metadata.csv"

# 读取数据
expr_df = pd.read_csv(expression_path, index_col=0)
meta_df = pd.read_csv(metadata_path)

# 构建样本到Tissue的映射
tissue_map = dict(zip(meta_df['Sample'], meta_df['Tissue']))

# 按Tissue分组样本
tissue_samples = {}
for sample, tissue in tissue_map.items():
    tissue_samples.setdefault(tissue, []).append(sample)

# 为每个Tissue创建文件夹并保存对应的expression.csv
for tissue, samples in tissue_samples.items():
    # 只保留在表达矩阵中的样本
    valid_samples = [s for s in samples if s in expr_df.columns]
    if not valid_samples:
        continue
    sub_expr = expr_df[valid_samples]
    out_dir = tissue.replace(' ', '_')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'expression.csv')
    # 保留基因名为第一列
    sub_expr.to_csv(out_path)
    print(f"写入 {out_path}，包含 {len(valid_samples)} 个样本")

print("分割完成！")
