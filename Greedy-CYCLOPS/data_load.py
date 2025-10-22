import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
# 确保你的 PCA.py 路径正确，或者直接从 sklearn 导入
# from sklearn.decomposition import PCA
from PCA import create_eigengenes
import gc

# ExpressionDataset and blunt_percentile

class ExpressionDataset(Dataset):
    def __init__(self, expressions, times=None, celltypes=None):
        self.expressions = torch.FloatTensor(expressions)
        self.times = torch.FloatTensor(times) if times is not None else None
        self.celltypes = celltypes

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, idx):
        sample = {'expression': self.expressions[idx]}
        if self.times is not None:
            sample['time'] = self.times[idx]
        if self.celltypes is not None:
            sample['celltype'] = self.celltypes[idx]
        return sample

def blunt_percentile(data, percent=0.975):
    # Winsorize each gene across samples by clipping to percentile bounds
    n = data.shape[0]
    nfloor_idx = max(0, int(np.floor((1 - percent) * n)))
    nceiling_idx = min(n - 1, int(np.ceil(percent * n)) - 1)
    if nfloor_idx >= nceiling_idx or n <= 1:
        return data
    sorted_data = np.sort(data, axis=0)
    row_min = sorted_data[nfloor_idx, :]
    row_max = sorted_data[nceiling_idx, :]
    data = np.clip(data, row_min, row_max)
    return data

# --- mean_normalize 函数已被移除 ---

# --- 更新后的 load_and_preprocess_train_data ---
def load_and_preprocess_train_data(
        train_file,
        n_components=50,
        blunt_percent=0.975,
        min_cv=0.14,
        max_cv=0.7,
        min_mean_rank=10000,
        use_oscope_filter=True,
        min_pair_corr=0.2,
        max_pair_corr=0.9,
        min_gene_pairs=5
    ):
    print("=== Loading training data ===")
    df = pd.read_csv(train_file, low_memory=False)
    sample_columns = [col for col in df.columns if col != 'Gene_Symbol']
    gene_df = df[~df['Gene_Symbol'].isin(['celltype_D', 'time_C'])].copy()
    initial_gene_symbols = gene_df['Gene_Symbol'].values
    expression_data = gene_df[sample_columns].values.T  # (samples, genes)

    print(f"Initial data shape: {expression_data.shape}")

    # 1. Winsorize
    expression_data = blunt_percentile(expression_data, percent=blunt_percent)
    print(f"Shape after blunt_percentile: {expression_data.shape}")

    # 2. Initial CV and mean-rank filtering
    gene_means = np.mean(expression_data, axis=0)
    gene_stds = np.std(expression_data, axis=0)
    gene_cvs = gene_stds / (gene_means + 1e-8)
    mean_rank = np.argsort(np.argsort(-gene_means))
    initial_keep = (gene_cvs > min_cv) & (gene_cvs < max_cv) & (mean_rank < min_mean_rank)

    expression_data_initial_filtered = expression_data[:, initial_keep]
    initial_filtered_gene_symbols = initial_gene_symbols[initial_keep]
    print(f"Shape after initial CV/Mean filtering: {expression_data_initial_filtered.shape}")

    if expression_data_initial_filtered.shape[1] == 0:
        # fallback: keep all genes and continue
        print("Warning: Initial CV/Mean filtering removed all genes. Falling back to keep all genes.")
        final_keep_mask = np.ones_like(initial_keep, dtype=bool)
        final_gene_symbols = initial_gene_symbols
        expression_data_final_filtered = expression_data[:, final_keep_mask]
        fallback_initial_filter = True
    else:
        fallback_initial_filter = False

    # If we didn't fallback, use the initial filtered data as default
    if not fallback_initial_filter:
        final_keep_mask = initial_keep
        final_gene_symbols = initial_filtered_gene_symbols
        expression_data_final_filtered = expression_data_initial_filtered

    # --- Oscope-inspired filtering ---
    if use_oscope_filter:
        print("Applying Oscope-inspired pair filtering...")
        if expression_data_initial_filtered.shape[1] <= 1:
             print("Skipping Oscope filter: Not enough genes after initial filtering.")
        else:
            # 3. Standardize for correlation calculation
            scaler_filter = StandardScaler()
            expression_scaled_filter = scaler_filter.fit_transform(expression_data_initial_filtered)

            # 4. 计算相关性矩阵
            print("Calculating correlation matrix...")
            # 确保使用 float64 以提高相关性计算的稳定性
            try:
                corr_matrix = np.corrcoef(expression_scaled_filter.astype(np.float64), rowvar=False)
            except MemoryError:
                 print("MemoryError calculating full correlation matrix. Consider reducing genes or using incremental methods.")
                 raise
            # 处理因标准差为零（如果发生）导致的相关性 NaN
            corr_matrix = np.nan_to_num(corr_matrix)
            n_filtered_genes = corr_matrix.shape[0]
            print(f"Correlation matrix shape: {corr_matrix.shape}")

            # 5. Identify good gene pairs and count occurrences
            print("Identifying good pairs and counting gene occurrences...")
            gene_pair_counts = np.zeros(n_filtered_genes, dtype=int)
            abs_corr = np.abs(corr_matrix)
            triu_mask = np.triu(np.ones_like(abs_corr, dtype=bool), k=1)
            corr_range_mask = (abs_corr > min_pair_corr) & (abs_corr < max_pair_corr)
            valid_pair_mask = triu_mask & corr_range_mask
            row_indices, col_indices = np.where(valid_pair_mask)
            np.add.at(gene_pair_counts, row_indices, 1)
            np.add.at(gene_pair_counts, col_indices, 1)
            n_good_pairs = len(row_indices)
            print(f"Found {n_good_pairs} pairs with abs(corr) between {min_pair_corr} and {max_pair_corr}.")
            oscope_keep_mask_relative = gene_pair_counts >= min_gene_pairs
            n_oscope_genes = np.sum(oscope_keep_mask_relative)
            print(f"Found {n_oscope_genes} genes participating in at least {min_gene_pairs} good pairs.")

            if n_oscope_genes == 0:
                print("Warning: Oscope filter removed all initially filtered genes. Reverting to initial CV/Mean filter.")
                # final_keep_mask 保持为 initial_keep
            elif n_oscope_genes < n_components:
                print(f"Warning: Oscope filter resulted in only {n_oscope_genes} genes, less than n_components={n_components}.")
                # 决定如何处理: 这里选择继续使用这少量基因
                # 需要将 oscope_keep_mask_relative (相对于 initial_filtered) 映射回原始基因索引
                initial_indices = np.where(initial_keep)[0]
                oscope_kept_indices = initial_indices[oscope_keep_mask_relative]
                oscope_keep_mask_final = np.zeros_like(initial_keep, dtype=bool)
                oscope_keep_mask_final[oscope_kept_indices] = True
                final_keep_mask = oscope_keep_mask_final # 更新掩码
            else:
                # 同样需要映射回原始索引
                initial_indices = np.where(initial_keep)[0]
                oscope_kept_indices = initial_indices[oscope_keep_mask_relative]
                oscope_keep_mask_final = np.zeros_like(initial_keep, dtype=bool)
                oscope_keep_mask_final[oscope_kept_indices] = True
                final_keep_mask = oscope_keep_mask_final # 更新掩码

            # Update final gene symbols and expression data
            final_gene_symbols = initial_gene_symbols[final_keep_mask]
            expression_data_final_filtered = expression_data[:, final_keep_mask]
            print(f"Shape after Oscope filtering: {expression_data_final_filtered.shape}")

            # 清理中间变量以释放内存
            del expression_scaled_filter, corr_matrix, abs_corr, triu_mask, corr_range_mask, valid_pair_mask
            gc.collect()

    # --- 后续处理使用 final_keep_mask 和 expression_data_final_filtered ---

    if expression_data_final_filtered.shape[1] == 0:
        raise ValueError("All genes were removed after all filtering steps.")

    # 7. Final standardization
    final_scaler = StandardScaler()
    expression_scaled = final_scaler.fit_transform(expression_data_final_filtered)

    # 8. PCA
    print("Performing PCA...")
    actual_n_components = min(n_components, expression_scaled.shape[1], expression_scaled.shape[0])
    if actual_n_components < n_components:
        print(f"Warning: Reducing n_components from {n_components} to {actual_n_components} due to limited features ({expression_scaled.shape[1]}) or samples ({expression_scaled.shape[0]}) after filtering.")
    if actual_n_components <= 0:
         raise ValueError("Cannot perform PCA with 0 components.")

    # 假设 create_eigengenes 返回 components, model, variance_ratio_sum or array
    try:
        pca_components, pca_model, explained_variance_ratios = create_eigengenes(expression_scaled, actual_n_components)
        print(f"PCA output shape: {pca_components.shape}")
        if isinstance(explained_variance_ratios, np.ndarray):
             print(f"Explained variance by {actual_n_components} components: {np.sum(explained_variance_ratios):.4f}")
        else: # 如果只返回总和
             print(f"Explained variance by {actual_n_components} components: {explained_variance_ratios:.4f}")

    except Exception as e:
         print(f"Error during PCA: {e}")
         print(f"Data shape fed to PCA: {expression_scaled.shape}")
         raise

    # Wrap into Dataset
    train_dataset = ExpressionDataset(pca_components)

    preprocessing_info = {
        'scaler': final_scaler,
        'pca_model': pca_model,
        'sample_columns': sample_columns,
        'n_components': actual_n_components,
        'gene_keep_mask': final_keep_mask,
        'final_gene_symbols': final_gene_symbols,
        'blunt_percent': blunt_percent,
        'use_oscope_filter': use_oscope_filter,
        'min_pair_corr': min_pair_corr,
        'max_pair_corr': max_pair_corr,
        'min_gene_pairs': min_gene_pairs,
        'fallback_initial_filter': fallback_initial_filter
    }
    return train_dataset, preprocessing_info

# --- 更新后的 load_and_preprocess_test_data ---
def load_and_preprocess_test_data(test_file, preprocessing_info):
    print("\n=== Loading test data ===")
    df = pd.read_csv(test_file, low_memory=False)

    celltype_row = df[df['Gene_Symbol'] == 'celltype_D']
    time_row = df[df['Gene_Symbol'] == 'time_C']
    has_celltype = not celltype_row.empty
    has_time = not time_row.empty
    print(f"Test set has time info: {has_time}")
    print(f"Test set has celltype info: {has_celltype}")

    # 使用训练时确定的样本列
    sample_columns = preprocessing_info['sample_columns']
    # 确保测试文件包含这些列，否则可能出错
    available_sample_columns = [col for col in sample_columns if col in df.columns]
    if len(available_sample_columns) != len(sample_columns):
        print(f"Warning: Test file is missing some sample columns defined during training.")
        # 可能需要处理这种情况，取决于你的需求

    celltypes = None
    times = None
    if has_celltype:
        celltypes = celltype_row.iloc[0].get(available_sample_columns, pd.Series(index=available_sample_columns, dtype=object)).values # 使用 get 获取，避免 KeyError
        print(f"Test set celltypes: {np.unique(np.asarray(celltypes))}")
    if has_time:
        times = time_row.iloc[0].get(available_sample_columns, pd.Series(index=available_sample_columns, dtype=float)).values.astype(float)
        print(f"Test set time range: {np.nanmin(times):.2f} - {np.nanmax(times):.2f} hours") # 使用 nanmin/max

    gene_df = df[~df['Gene_Symbol'].isin(['celltype_D', 'time_C'])].copy()

    # --- 关键：确保使用与训练时完全相同的基因集和顺序 ---
    final_gene_symbols = preprocessing_info['final_gene_symbols']
    # 使用 reindex 保证顺序和基因集一致
    gene_df_indexed = gene_df.set_index('Gene_Symbol').reindex(final_gene_symbols)
    # 现在从 reindex 后的 DataFrame 中提取表达数据，注意样本列顺序也要对齐
    test_expression_data = gene_df_indexed[available_sample_columns].values.T # (samples, genes)

    # 处理 reindex 可能引入的 NaN (基因在测试集缺失)
    if np.isnan(test_expression_data).any():
        print("Warning: Missing gene values detected in test set after reindexing. Imputing with 0 before scaling.")
        # 在标准化之前用 0 填充可能是合理的，因为标准化会处理均值
        test_expression_data = np.nan_to_num(test_expression_data, nan=0.0)

    print(f"Aligned test data shape before processing: {test_expression_data.shape}")
    if test_expression_data.shape[1] != len(final_gene_symbols):
         print(f"Warning: Shape mismatch after gene alignment. Expected {len(final_gene_symbols)} genes, got {test_expression_data.shape[1]}")


    # --- 应用训练时确定的预处理步骤 ---

    # 1. 异常值处理 (使用训练时的百分比，但理想应使用训练时确定的阈值)
    # 注意: 对测试集应用 blunt_percentile 可能不是最佳做法，因为它基于测试集分布。
    # 更严格的方法是保存训练集的 row_min/row_max 并应用 clip。为保持一致性，暂时保留。
    test_expression_data = blunt_percentile(test_expression_data, percent=preprocessing_info['blunt_percent'])

    # 2. 标准化 (使用训练集的 scaler)
    scaler = preprocessing_info['scaler']
    try:
        test_expression_scaled = scaler.transform(test_expression_data)
    except ValueError as e:
         print(f"Error during scaling test data: {e}")
         print(f"Test data shape: {test_expression_data.shape}, Scaler expected features: {scaler.n_features_in_}")
         raise

    # 3. PCA 变换 (使用训练集的 pca_model)
    pca_model = preprocessing_info['pca_model']
    actual_n_components = preprocessing_info['n_components'] # 使用训练时确定的组件数

    # 再次检查特征数是否匹配 PCA 模型
    if test_expression_scaled.shape[1] != pca_model.n_features_in_:
         raise ValueError(f"Number of features in scaled test data ({test_expression_scaled.shape[1]}) does not match PCA model expected input features ({pca_model.n_features_in_})")

    try:
        test_pca_components = pca_model.transform(test_expression_scaled)
    except Exception as e:
        print(f"Error during PCA transform on test data: {e}")
        print(f"Scaled test data shape: {test_expression_scaled.shape}")
        raise

    print(f"Test data after PCA shape: {test_pca_components.shape}")
    print(f"Expected n_components: {actual_n_components}")

    # --- 封装成 Dataset ---
    test_dataset = ExpressionDataset(test_pca_components, times, celltypes)

    # 添加测试集相关信息 (可选)
    test_preprocessing_info = preprocessing_info.copy()
    test_preprocessing_info.update({
        'test_has_time': has_time,
        'test_has_celltype': has_celltype,
        'test_sample_columns': available_sample_columns, # 实际使用的样本列
        'test_pca_components': test_pca_components # 存储测试集的 PCA 结果
    })
    return test_dataset, test_preprocessing_info