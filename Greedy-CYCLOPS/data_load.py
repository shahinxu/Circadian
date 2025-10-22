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

def blunt_percentile(data, percent=0.975):
    # data: (samples, genes)
    n = data.shape[0]
    nfloor = int(1 + np.floor((1 - percent) * n))
    nceiling = int(np.ceil(percent * n))
    sorted_data = np.sort(data, axis=0)
    row_min = sorted_data[nfloor, :]
    row_max = sorted_data[nceiling-1, :]
    data = np.where(data < row_min, row_min, data)
    data = np.where(data > row_max, row_max, data)
    return data

def mean_normalize(data):
    gene_means = np.mean(data, axis=0, keepdims=True)
    return (data - gene_means) / gene_means

def load_and_preprocess_train_data(
        train_file, 
        n_components=50, 
        blunt_percent=0.975, 
        do_mean_normalize=True, 
        min_cv=0.14, 
        max_cv=0.7, 
        min_mean_rank=10000
    ):
    print("=== Loading training data ===")
    df = pd.read_csv(train_file, low_memory=False)
    sample_columns = [col for col in df.columns if col != 'Gene_Symbol']
    gene_df = df[~df['Gene_Symbol'].isin(['celltype_D', 'time_C'])].copy()
    expression_data = gene_df[sample_columns].values.T  # (samples, genes)
    expression_data = blunt_percentile(expression_data, percent=blunt_percent)
    gene_means = np.mean(expression_data, axis=0)
    gene_stds = np.std(expression_data, axis=0)
    gene_cvs = gene_stds / (gene_means + 1e-8)
    mean_rank = np.argsort(-gene_means)  # descending
    keep = (gene_cvs > min_cv) & (gene_cvs < max_cv) & (mean_rank < min_mean_rank)
    expression_data = expression_data[:, keep]
    if do_mean_normalize:
        expression_data = mean_normalize(expression_data)
    scaler = StandardScaler()
    expression_scaled = scaler.fit_transform(expression_data)
    pca_components, pca_model, _ = create_eigengenes(expression_scaled, n_components)
    train_dataset = ExpressionDataset(pca_components)
    preprocessing_info = {
        'scaler': scaler,
        'pca_model': pca_model,
        'sample_columns': sample_columns,
        'n_components': n_components,
        'gene_keep_mask': keep,
        'do_mean_normalize': do_mean_normalize,
        'blunt_percent': blunt_percent
    }
    return train_dataset, preprocessing_info

def load_and_preprocess_test_data(test_file, preprocessing_info):
    print("\n=== Loading test data ===")
    df = pd.read_csv(test_file, low_memory=False)
    celltype_row = df[df['Gene_Symbol'] == 'celltype_D']
    time_row = df[df['Gene_Symbol'] == 'time_C']
    has_celltype = not celltype_row.empty
    has_time = not time_row.empty
    print(f"Test set has time info: {has_time}")
    print(f"Test set has celltype info: {has_celltype}")
    sample_columns = [col for col in df.columns if col != 'Gene_Symbol']
    celltypes = None
    times = None
    if has_celltype:
        celltypes = celltype_row.iloc[0][sample_columns].values
        print(f"Test set celltypes: {np.unique(np.asarray(celltypes))}")
    if has_time:
        times = time_row.iloc[0][sample_columns].values.astype(float)
        print(f"Test set time range: {times.min():.2f} - {times.max():.2f} hours")
    gene_df = df[~df['Gene_Symbol'].isin(['celltype_D', 'time_C'])].copy()
    test_gene_names = gene_df['Gene_Symbol'].values
    test_expression_data = gene_df[sample_columns].values.T
    # 1. Only keep genes selected in training
    keep = preprocessing_info['gene_keep_mask']
    test_expression_data = test_expression_data[:, keep]
    # 2. Outlier truncation
    test_expression_data = blunt_percentile(test_expression_data, percent=preprocessing_info['blunt_percent'])
    # 3. Mean normalization
    if preprocessing_info['do_mean_normalize']:
        test_expression_data = mean_normalize(test_expression_data)
    # 4. Standardization
    scaler = preprocessing_info['scaler']
    test_expression_scaled = scaler.transform(test_expression_data)
    # 5. PCA
    pca_model = preprocessing_info['pca_model']
    n_components = preprocessing_info['n_components']
    test_spc_components = pca_model.transform(test_expression_scaled)
    print(f"Test data after PCA shape: {test_spc_components.shape}")
    print(f"Expected n_components: {n_components}")
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