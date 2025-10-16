import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

base_path = "/home/xuzhen/data/GSE54652"
fit_output_path = "output/CYCLOPS_2025-10-10T10_15_00/Fits/Fit_Output_2025-10-10T10_15_00.csv"
metadata_path = base_path + "/metadata.csv"

# 读取数据
fit_df = pd.read_csv(fit_output_path)
meta_df = pd.read_csv(metadata_path)

# 自动查找 Phases_AG 列
phase_col = [col for col in fit_df.columns if col.startswith("Phases_AG")]
if not phase_col:
    raise ValueError("未找到 Phases_AG 列！")
phase_col = phase_col[0]

# 样本名列名自动判断
fit_samples = fit_df['Sample'] if 'Sample' in fit_df.columns else fit_df.iloc[:, 0]
meta_samples = meta_df['Sample'] if 'Sample' in meta_df.columns else meta_df.iloc[:, 0]

# 构建样本到 Phase/Time_Hours 的映射
fit_map = dict(zip(fit_samples, fit_df[phase_col]))
meta_map = dict(zip(meta_samples, meta_df['Time_Hours']))

# 取交集
common_samples = list(set(fit_map.keys()) & set(meta_map.keys()))
if not common_samples:
    raise ValueError("没有交集样本，无法画图！")

# 对齐数据
phases = [fit_map[s] for s in common_samples]
time_hours = [meta_map[s] for s in common_samples]
time_radian = (np.array(time_hours) % 24) * 2 * np.pi / 24

plt.figure(figsize=(8, 7))
plt.scatter(time_radian, phases, c='b', label='Phase vs. Time')
plt.xlabel('Collection Time', fontsize=24)
plt.ylabel('Predicted Phase', fontsize=24)
plt.tight_layout()
plt.savefig(os.path.join(base_path, "phase_vs_time.png"), dpi=150)
plt.show()
print('已保存图片 phase_vs_time.png')
