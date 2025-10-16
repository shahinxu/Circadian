import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from PCA import create_eigengenes

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


def load_and_preprocess_train_data(train_file, n_components=50, max_samples=100, random_state=42,
                                   select_top_genes: int = None):
    print("=== 加载训练数据 ===")
    df = pd.read_csv(train_file, low_memory=False)
    
    celltype_row = df[df['Gene_Symbol'] == 'celltype_D']
    time_row = df[df['Gene_Symbol'] == 'time_C']
    
    has_celltype = not celltype_row.empty
    has_time = not time_row.empty
    
    print(f"训练集包含时间信息: {has_time}")
    print(f"训练集包含细胞类型信息: {has_celltype}")
    
    sample_columns = [col for col in df.columns if col != 'Gene_Symbol']
    n_samples = len(sample_columns)
    
    print(f"原始样本数量: {n_samples}")
    print(f"最大样本数量限制: {max_samples}")
    
    celltypes = None
    times = None
    
    if has_celltype:
        celltypes = celltype_row.iloc[0][sample_columns].values
    print(f"训练集细胞类型: {np.unique(np.asarray(celltypes))}")
    
    if has_time:
        times = time_row.iloc[0][sample_columns].values.astype(float)
        print(f"训练集时间范围: {times.min():.2f} - {times.max():.2f} 小时")
    
    gene_df = df[~df['Gene_Symbol'].isin(['celltype_D', 'time_C'])].copy()
    gene_names = gene_df['Gene_Symbol'].values
    expression_data = gene_df[sample_columns].values.T
    
    print(f"训练集原始基因数量: {len(gene_names)}")
    
    print("进行训练数据标准化...")
    # Optional: select top genes by variance before scaling/PCA
    selected_gene_names = gene_names
    if select_top_genes is not None and select_top_genes > 0:
        k = min(select_top_genes, expression_data.shape[1])
        gene_var = np.nanvar(expression_data, axis=0)
        top_idx = np.argsort(-gene_var)[:k]
        expression_data = expression_data[:, top_idx]
        selected_gene_names = gene_names[top_idx]

    scaler = StandardScaler()
    expression_scaled = scaler.fit_transform(expression_data)
    
    selected_expression, pca_model, explained_variance = create_eigengenes(
        expression_scaled, n_components
    )
    
    if n_samples > max_samples:
        print(f"样本数量 ({n_samples}) 超过最大限制 ({max_samples})，进行截断...")
        np.random.seed(random_state)
        selected_indices = np.random.choice(n_samples, max_samples, replace=False)
        selected_indices = np.sort(selected_indices)
        
        selected_expression = selected_expression[selected_indices]
        if times is not None:
            times = times[selected_indices]
        if celltypes is not None:
            celltypes = celltypes[selected_indices]
        
        actual_samples = max_samples
        print(f"截断后样本数量: {actual_samples}")
        
    elif n_samples < max_samples:
        print(f"样本数量 ({n_samples}) 少于最大限制 ({max_samples})，进行0填充...")
        pad_size = max_samples - n_samples
        
        padding = np.zeros((pad_size, n_components))
        selected_expression = np.vstack([selected_expression, padding])
        
        if times is not None:
            times = np.concatenate([times, np.zeros(pad_size)])
        if celltypes is not None:
            celltypes = np.concatenate([np.asarray(celltypes), np.asarray(['PADDING'] * pad_size)])
        
        actual_samples = max_samples
        print(f"填充后样本数量: {actual_samples}")
        
    else:
        actual_samples = n_samples
        print(f"样本数量正好等于最大限制: {actual_samples}")
        
    celltype_to_idx = {}
    if has_celltype:
        unique_celltypes = np.unique(np.asarray(celltypes))
        celltype_to_idx = {ct: idx for idx, ct in enumerate(unique_celltypes)}
        print(f"细胞类型映射: {celltype_to_idx}")
    
    train_dataset = ExpressionDataset(selected_expression, times, celltypes)
    
    preprocessing_info = {
        'scaler': scaler,
        'pca_model': pca_model,
        'spc_components': selected_expression,
        'explained_variance': explained_variance,
        'train_has_time': has_time,
        'train_has_celltype': has_celltype,
        'celltype_to_idx': celltype_to_idx,
        'n_celltypes': len(celltype_to_idx) if celltype_to_idx else 0,
        'n_components': n_components,
        'max_samples': max_samples,
        'actual_samples': actual_samples,
        'original_samples': n_samples,
        'all_gene_names': gene_names,
        'selected_gene_names': selected_gene_names,
        'period_hours': 24.0
    }
    
    return train_dataset, preprocessing_info

def load_and_preprocess_test_data(test_file, preprocessing_info):
    print("\n=== 加载测试数据 ===")
    df = pd.read_csv(test_file, low_memory=False)
    
    celltype_row = df[df['Gene_Symbol'] == 'celltype_D']
    time_row = df[df['Gene_Symbol'] == 'time_C']
    
    has_celltype = not celltype_row.empty
    has_time = not time_row.empty
    
    print(f"测试集包含时间信息: {has_time}")
    print(f"测试集包含细胞类型信息: {has_celltype}")
    
    sample_columns = [col for col in df.columns if col != 'Gene_Symbol']
    n_samples = len(sample_columns)
    
    celltypes = None
    times = None
    
    if has_celltype:
        celltypes = celltype_row.iloc[0][sample_columns].values
        print(f"测试集细胞类型: {np.unique(np.asarray(celltypes))}")
    
    if has_time:
        times = time_row.iloc[0][sample_columns].values.astype(float)
        print(f"测试集时间范围: {times.min():.2f} - {times.max():.2f} 小时")
    
    gene_df = df[~df['Gene_Symbol'].isin(['celltype_D', 'time_C'])].copy()
    test_gene_names = gene_df['Gene_Symbol'].values
    test_expression_data = gene_df[sample_columns].values.T
    
    print(f"测试集原始基因数量: {len(test_gene_names)}")
    print(f"测试集样本数量: {n_samples}")
    
    scaler = preprocessing_info['scaler']
    pca_model = preprocessing_info['pca_model']
    n_components = preprocessing_info['n_components']
    
    print("使用训练集的标准化参数处理测试数据...")
    test_expression_scaled = scaler.transform(test_expression_data)
    
    print("使用训练集的细胞类型感知变换器处理测试数据...")
    test_spc_components = pca_model.transform(test_expression_scaled)
    
    print(f"细胞类型感知变换后的测试数据维度: {test_spc_components.shape}")
    print(f"期望的组件数量: {n_components}")
    
    test_dataset = ExpressionDataset(test_spc_components, times, celltypes)
    
    test_preprocessing_info = preprocessing_info.copy()
    test_preprocessing_info.update({
        'test_has_time': has_time,
        'test_has_celltype': has_celltype,
        'test_sample_columns': sample_columns,
        'test_gene_names': test_gene_names,
        'test_spc_components': test_spc_components
    })
    
    return test_dataset, test_preprocessing_info