import os
from typing import Optional, cast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
from numpy.typing import ArrayLike, NDArray

def coords_to_phase(coords):
    x, y = coords[:, 0], coords[:, 1]
    phase = torch.atan2(y, x)
    phase = torch.where(phase < 0, phase + 2*np.pi, phase)
    return phase

def phase_to_coords(phase):
    x = torch.cos(phase)
    y = torch.sin(phase)
    return torch.stack([x, y], dim=1)

def time_to_phase(time_hours, period_hours=24.0):
    return 2 * np.pi * time_hours / period_hours

def plot_eigengenes_2d_with_phase_gradient(
        test_file, 
        preprocessing_info, 
        predicted_phases, 
        celltypes_data, save_dir='./results'):
    print("\n=== 绘制Eigengenes 2D关系图（相位渐变色）===")
    
    df = pd.read_csv(test_file, low_memory=False)
    
    celltype_row = df[df['Gene_Symbol'] == 'celltype_D']
    sample_columns = [col for col in df.columns if col != 'Gene_Symbol']
    
    gene_df = df[~df['Gene_Symbol'].isin(['celltype_D', 'time_C'])].copy()
    test_expression_data = gene_df[sample_columns].values.T
    
    scaler = preprocessing_info['scaler']
    pca_model = preprocessing_info['pca_model']
    
    test_expression_scaled = scaler.transform(test_expression_data)
    eigengenes = pca_model.transform(test_expression_scaled)
    
    print(f"Eigengenes维度: {eigengenes.shape}")
    print(f"预测相位范围: {predicted_phases.min():.3f} - {predicted_phases.max():.3f} 弧度")
    
    phase_normalized = (predicted_phases - predicted_phases.min()) / (predicted_phases.max() - predicted_phases.min())
    
    os.makedirs(save_dir, exist_ok=True)
    
    if celltypes_data is not None:
        unique_celltypes = np.unique(celltypes_data)
        unique_celltypes = [ct for ct in unique_celltypes if ct != 'PADDING']
        print(f"细胞类型: {unique_celltypes}")
    else:
        unique_celltypes = ['All_Samples']
        celltypes_data = np.array(['All_Samples'] * len(predicted_phases))
    
    n_components = min(5, eigengenes.shape[1])
    eigengene_pairs = []
    for i in range(n_components):
        for j in range(i+1, n_components):
            eigengene_pairs.append((i, j))
    
    print(f"将绘制 {len(eigengene_pairs)} 个eigengene对")
    
    for celltype in unique_celltypes:
        print(f"绘制细胞类型: {celltype}")
        
        celltype_mask = celltypes_data == celltype
        celltype_eigengenes = eigengenes[celltype_mask]
        celltype_phases = predicted_phases[celltype_mask]
        celltype_phase_norm = phase_normalized[celltype_mask]
        
        if len(celltype_eigengenes) < 3:
            print(f"  细胞类型 {celltype} 样本太少，跳过")
            continue
        
        n_cols = min(3, len(eigengene_pairs))
        n_rows = (len(eigengene_pairs) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if len(eigengene_pairs) == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten() if len(eigengene_pairs) > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for idx, (pc1, pc2) in enumerate(eigengene_pairs):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            x_data = celltype_eigengenes[:, pc1]
            y_data = celltype_eigengenes[:, pc2]
            
            scatter = ax.scatter(x_data, y_data, 
                               c=celltype_phase_norm, 
                               cmap='hsv',
                               alpha=0.8, 
                               s=50, 
                               edgecolors='black', 
                               linewidth=0.5)
            
            explained_var_1 = preprocessing_info['explained_variance'][pc1] * 100
            explained_var_2 = preprocessing_info['explained_variance'][pc2] * 100
            
            ax.set_xlabel(f'Eigengene {pc1+1} ({explained_var_1:.1f}% variance)', fontsize=12)
            ax.set_ylabel(f'Eigengene {pc2+1} ({explained_var_2:.1f}% variance)', fontsize=12)
            ax.set_title(f'Eigengene {pc1+1} vs {pc2+1}\n{celltype} (n={len(x_data)})', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Predicted Phase\n(gradient)', rotation=270, labelpad=15)
            
            phase_hours_min = celltype_phases.min() * 24 / (2 * np.pi)
            phase_hours_max = celltype_phases.max() * 24 / (2 * np.pi)
            cbar.set_ticks([0, 0.5, 1])
            cbar.set_ticklabels([f'{phase_hours_min:.1f}h', 
                               f'{(phase_hours_min + phase_hours_max)/2:.1f}h',
                               f'{phase_hours_max:.1f}h'])
        
        for idx in range(len(eigengene_pairs), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Eigengenes 2D Patterns - {celltype}\n(Colors represent predicted phase)', 
                     fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        filename = f'eigengenes_2d_phase_gradient_{celltype}.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  已保存: {filepath}")
    
    print("Eigengenes 2D相位渐变图绘制完成！")


def create_prediction_plots(results_df, save_dir):
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(results_df['Predicted_Phase_Hours'], bins=24, alpha=0.7, edgecolor='black')
    plt.xlabel('Predicted Phase (Hours)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Phases')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.scatter(results_df['Phase_X'], results_df['Phase_Y'], alpha=0.6)
    plt.xlabel('Phase X')
    plt.ylabel('Phase Y')
    plt.title('Phase Distribution in Unit Circle')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    circle = Circle((0, 0), 1, fill=False, color='red', linestyle='--', alpha=0.5)
    plt.gca().add_patch(circle)
    
    if 'True_Time_Hours' in results_df.columns:
        plt.subplot(2, 2, 3)
        plt.scatter(results_df['True_Time_Hours'], results_df['Predicted_Phase_Hours'], alpha=0.6)
        plt.plot([0, 24], [0, 24], 'r--', label='Perfect Prediction')
        plt.xlabel('True Time (Hours)')
        plt.ylabel('Predicted Phase (Hours)')
        plt.title('True Time vs Predicted Phase')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.hist(results_df['Phase_Error_Hours'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Error (Hours)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors')
        plt.grid(True, alpha=0.3)
    else:
        if 'Cell_Type' in results_df.columns:
            plt.subplot(2, 2, 3)
            unique_celltypes = results_df['Cell_Type'].unique()
            cmap = plt.get_cmap('tab10')
            colors = cmap(np.linspace(0, 1, len(unique_celltypes)))
            
            for i, celltype in enumerate(unique_celltypes):
                mask = results_df['Cell_Type'] == celltype
                plt.scatter(results_df.loc[mask, 'Phase_X'], 
                          results_df.loc[mask, 'Phase_Y'], 
                          c=[colors[i]], label=celltype, alpha=0.6)
            
            plt.xlabel('Phase X')
            plt.ylabel('Phase Y')
            plt.title('Phase Distribution by Cell Type')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            
            circle = Circle((0, 0), 1, fill=False, color='red', linestyle='--', alpha=0.5)
            plt.gca().add_patch(circle)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"预测分析图表保存到: {os.path.join(save_dir, 'prediction_analysis.png')}")


def get_original_gene_expressions(test_file, custom_genes, preprocessing_info, phase_hours_data, celltypes_data):
    try:
        print(f"从原始文件重新读取自定义基因: {custom_genes}")
        
        df = pd.read_csv(test_file, low_memory=False)
        
        gene_df = df[~df['Gene_Symbol'].isin(['celltype_D', 'time_C'])].copy()
        test_gene_names = gene_df['Gene_Symbol'].values
        
        sample_columns = [col for col in df.columns if col != 'Gene_Symbol']
        test_expression_data = gene_df[sample_columns].values.T
        
        scaler = preprocessing_info['scaler']
        test_expression_scaled = scaler.transform(test_expression_data)
        
        found_genes = []
        gene_expressions_list = []
        
        for gene in custom_genes:
            if gene in test_gene_names:
                gene_idx = np.where(test_gene_names == gene)[0][0]
                gene_expression = test_expression_scaled[:, gene_idx]
                gene_expressions_list.append(gene_expression)
                found_genes.append(gene)
                print(f"  ✓ 找到基因: {gene}")
            else:
                print(f"  ✗ 基因 {gene} 不在测试数据中")
        
        if len(found_genes) == 0:
            print("错误: 没有找到任何指定的基因")
            return None, None
        
        gene_expressions = np.column_stack(gene_expressions_list)
        
        print(f"成功获取 {len(found_genes)} 个基因的表达数据")
        print(f"表达数据维度: {gene_expressions.shape}")
        
        return gene_expressions, np.array(found_genes)
        
    except Exception as e:
        print(f"从原始文件读取基因表达数据时出错: {e}")
        return None, None

def plot_celltype_gene_expression_raw(expressions, phase_hours, gene_names, celltype, save_dir):
    n_genes = len(gene_names)
    
    print(f"为细胞类型 {celltype} 绘制基因表达图（原始数据）")
    print(f"样本数量: {len(expressions)}")
    print(f"基因数量: {n_genes}")
    print(f"表达数据维度: {expressions.shape}")
    
    n_cols = min(5, n_genes)
    n_rows = (n_genes + n_cols - 1) // n_cols
    
    _, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_genes == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten() if n_genes > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, gene_name in enumerate(gene_names):
        ax = axes[i]
        
        gene_expression = expressions[:, i]
        
        if len(phase_hours) == 0:
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{gene_name}', fontsize=10)
            continue
        
        ax.scatter(phase_hours, gene_expression, alpha=0.6, s=20, color='blue', label='Raw Data')
        
        if len(phase_hours) > 5:
            try:
                def sine_func(x, amplitude, phase_shift, offset):
                    return amplitude * np.sin(2 * np.pi * x / 24 + phase_shift) + offset
                
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(sine_func, phase_hours, gene_expression, max_nfev=2000)
                
                x_fit = np.linspace(0, 24, 100)
                y_fit = sine_func(x_fit, *popt)
                ax.plot(x_fit, y_fit, '--', color='green', alpha=0.7, linewidth=2)
                
                amplitude, phase_shift, _ = popt
                peak_time = (-phase_shift * 24 / (2 * np.pi)) % 24
                
                y_pred = sine_func(phase_hours, *popt)
                ss_res = np.sum((gene_expression - y_pred) ** 2)
                ss_tot = np.sum((gene_expression - np.mean(gene_expression)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                ax.text(0.02, 0.98, 
                       f'Peak: {peak_time:.1f}h\nAmp: {amplitude:.3f}\nR²: {r_squared:.3f}', 
                       transform=ax.transAxes, verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
            except Exception as e:
                print(f"拟合失败 {gene_name}: {e}")
        
        ax.set_title(f'{gene_name}', fontsize=10)
        ax.set_xlabel('Predicted Phase (Hours)')
        ax.set_ylabel('Expression Level')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        
        if i == 0:
            ax.legend(fontsize=8)
    
    for i in range(n_genes, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Gene Expression vs Predicted Phase - {celltype} (Raw Data)', fontsize=14)
    plt.tight_layout()
    
    filename = f'gene_expression_phase_{celltype}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"基因表达相位图（原始数据）已保存: {filepath}")

def plot_celltype_comparison_raw(
    expressions, 
    phase_hours, 
    celltypes, 
    gene_names, 
    valid_celltypes, 
    save_dir,
    n_genes_to_compare=4
):
    print("绘制细胞类型对比图（原始数据）...")
    print(f"有效细胞类型: {valid_celltypes}")
    print(f"选择的基因: {gene_names[:n_genes_to_compare]}")
    
    n_genes_to_compare = min(n_genes_to_compare, len(gene_names))
    top_genes = gene_names[:n_genes_to_compare]
    
    n_cols = min(2, n_genes_to_compare)
    n_rows = (n_genes_to_compare + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
    if n_genes_to_compare == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten() if n_genes_to_compare > 1 else [axes]
    else:
        axes = axes.flatten()
    
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(valid_celltypes)))
    
    for i, gene_name in enumerate(top_genes):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        for celltype_idx, celltype in enumerate(valid_celltypes):
            celltype_mask = celltypes == celltype
            celltype_expressions = expressions[celltype_mask, i]
            celltype_phases = phase_hours[celltype_mask]
            
            if len(celltype_expressions) < 3:
                continue
            
            ax.scatter(celltype_phases, celltype_expressions, 
                      color=colors[celltype_idx], alpha=0.6, s=15, 
                      label=f'{celltype} (n={len(celltype_expressions)})')
            
            if len(celltype_expressions) > 5:
                try:
                    sorted_indices = np.argsort(celltype_phases)
                    sorted_phases = celltype_phases[sorted_indices]
                    sorted_expressions = celltype_expressions[sorted_indices]
                    
                    window_size = max(3, len(sorted_phases) // 8)
                    if len(sorted_phases) >= window_size:
                        from scipy.ndimage import uniform_filter1d
                        smooth_expression = uniform_filter1d(sorted_expressions.astype(float), size=window_size)
                        ax.plot(sorted_phases, smooth_expression, 
                               color=colors[celltype_idx], linewidth=2, alpha=0.8)
                        
                except Exception as e:
                    print(f"平滑处理失败 {gene_name} - {celltype}: {e}")
        
        ax.set_title(f'{gene_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Phase (Hours)', fontsize=12)
        ax.set_ylabel('Expression Level', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) <= 8:
            ax.legend(fontsize=10, loc='best')
        else:
            ax.legend(handles[:8], labels[:8], fontsize=10, loc='best', 
                     title=f"Showing first 8/{len(handles)} cell types")
    
    for i in range(n_genes_to_compare, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Gene Expression Comparison Across Cell Types (Raw Data)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = 'gene_expression_celltype_comparison.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"细胞类型对比图（原始数据）已保存: {filepath}")

def plot_gene_expression_with_custom_data(
    model, 
    test_loader, 
    preprocessing_info, 
    custom_gene_expressions, 
    custom_gene_names, 
    device='cuda', 
    save_dir='./results'
):
    print("\n=== 使用自定义基因数据绘制基因表达相位图（原始数据）===")
    model.eval()
    
    celltype_to_idx = preprocessing_info.get('celltype_to_idx', {})
    
    all_phase_hours = []
    all_celltypes = []
    
    with torch.no_grad():
        for batch in test_loader:
            expressions = batch['expression'].to(device)
            celltypes = batch.get('celltype', None)
            
            # 准备细胞类型索引
            celltype_indices = None
            if celltypes is not None and celltype_to_idx:
                batch_celltype_indices = []
                for ct in celltypes:
                    batch_celltype_indices.append(celltype_to_idx.get(ct, 0))
                celltype_indices = torch.tensor(batch_celltype_indices, device=device)
            
            phase_coords, phase_angles, _ = model(expressions, celltype_indices)
            phase_hours = phase_angles.cpu().numpy() * preprocessing_info.get('period_hours', 24.0) / (2 * np.pi)
            
            all_phase_hours.append(phase_hours)
            
            if celltypes is not None:
                all_celltypes.extend(celltypes)
    
    phase_hours_data = np.concatenate(all_phase_hours)
    
    if all_celltypes:
        celltypes_data = np.array(all_celltypes)
        unique_celltypes = np.unique(celltypes_data)
        print(f"发现细胞类型: {unique_celltypes}")
    else:
        celltypes_data = None
        unique_celltypes = ['All_Samples']

    
    os.makedirs(save_dir, exist_ok=True)
    
    if celltypes_data is not None:
        for celltype in unique_celltypes:
            if celltype == 'PADDING':
                continue
                
            celltype_mask = celltypes_data == celltype
            celltype_expressions = custom_gene_expressions[celltype_mask]
            celltype_phases = phase_hours_data[celltype_mask]
            
            if len(celltype_expressions) < 5:
                continue
            
            plot_celltype_gene_expression_raw(
                celltype_expressions, celltype_phases, custom_gene_names,
                celltype, save_dir
            )
    else:
        plot_celltype_gene_expression_raw(
            custom_gene_expressions, phase_hours_data, custom_gene_names,
            'All_Samples', save_dir
        )
    
    if celltypes_data is not None and len(unique_celltypes) > 1:
        valid_celltypes = [ct for ct in unique_celltypes if ct != 'PADDING']
        if len(valid_celltypes) > 1:
            plot_celltype_comparison_raw(
                custom_gene_expressions, phase_hours_data, celltypes_data, custom_gene_names,
                valid_celltypes, save_dir
            )
    
    print("自定义基因表达相位图绘制完成！")

def predict_and_save_phases(model, test_loader, preprocessing_info, device='cuda', save_dir='./results'):
    print("\n=== 预测测试集相位 ===")
    model.eval()
    
    celltype_to_idx = preprocessing_info.get('celltype_to_idx', {})
    sample_names = preprocessing_info.get('test_sample_columns', [])
    
    all_phase_coords = []
    all_phases = []
    all_times = []
    all_celltypes = []
    
    batch_start_idx = 0
    
    with torch.no_grad():
        for batch in test_loader:
            expressions = batch['expression'].to(device)
            times = batch.get('time', None)
            celltypes = batch.get('celltype', None)
            
            celltype_indices = None
            if celltypes is not None and celltype_to_idx:
                batch_celltype_indices = []
                for ct in celltypes:
                    batch_celltype_indices.append(celltype_to_idx.get(ct, 0))
                celltype_indices = torch.tensor(batch_celltype_indices, device=device)
            
            phase_coords, phase_angles, _ = model(expressions, celltype_indices)
            
            all_phase_coords.append(phase_coords.cpu().numpy())
            all_phases.append(phase_angles.cpu().numpy())
            
            batch_size = expressions.shape[0]
            batch_start_idx += batch_size
            
            if times is not None:
                all_times.append(times.cpu().numpy())
            if celltypes is not None:
                all_celltypes.extend(celltypes)
    
    phase_coords = np.vstack(all_phase_coords)
    phases = np.concatenate(all_phases)
    
    if all_times:
        times = np.concatenate(all_times)
    else:
        times = None
        
    if all_celltypes:
        celltypes = np.array(all_celltypes)
    else:
        celltypes = None
    
    os.makedirs(save_dir, exist_ok=True)
    
    if sample_names and len(sample_names) == len(phases):
        sample_identifiers = sample_names
    else:
        sample_identifiers = [f"Sample_{i}" for i in range(len(phases))]
        print("警告: 无法获取样本名称，使用生成的索引")
    
    results_data = {
        'Sample_ID': sample_identifiers,
        'Phase_X': phase_coords[:, 0],
        'Phase_Y': phase_coords[:, 1],
        'Predicted_Phase_Radians': phases,
        'Predicted_Phase_Degrees': phases * 180 / np.pi,
        'Predicted_Phase_Hours': phases * preprocessing_info.get('period_hours', 24.0) / (2 * np.pi)
    }
    
    if times is not None:
        results_data['True_Time_Hours'] = times
        results_data['True_Phase_Radians'] = time_to_phase(times, preprocessing_info.get('period_hours', 24.0))
        results_data['Phase_Error_Radians'] = np.abs(phases - results_data['True_Phase_Radians'])
        results_data['Phase_Error_Radians'] = np.minimum(
            results_data['Phase_Error_Radians'], 
            2*np.pi - results_data['Phase_Error_Radians']
        )
        results_data['Phase_Error_Hours'] = results_data['Phase_Error_Radians'] * preprocessing_info.get('period_hours', 24.0) / (2 * np.pi)
    
    if celltypes is not None:
        results_data['Cell_Type'] = celltypes
    
    results_df = pd.DataFrame(results_data)
    
    # 不保存任何预测CSV，确保目录内不遗留历史预测文件
    for fname in ('phase_predictions.csv', 'phase_predictions_simple.csv'):
        fpath = os.path.join(save_dir, fname)
        if os.path.isfile(fpath):
            try:
                os.remove(fpath)
                print(f"已删除遗留文件: {fpath}")
            except Exception as e:
                print(f"[WARN] 无法删除遗留文件 {fpath}: {e}")
    
    print(f"\n=== 预测统计 ===")
    print(f"预测样本数量: {len(phases)}")
    print(f"预测相位范围: {phases.min():.3f} - {phases.max():.3f} 弧度")
    print(f"预测相位范围: {(phases * 180 / np.pi).min():.1f} - {(phases * 180 / np.pi).max():.1f} 度")
    print(f"预测时间范围: {results_data['Predicted_Phase_Hours'].min():.2f} - {results_data['Predicted_Phase_Hours'].max():.2f} 小时")
    
    if times is not None:
        mean_error_hours = np.mean(results_data['Phase_Error_Hours'])
        std_error_hours = np.std(results_data['Phase_Error_Hours'])
        print(f"平均预测误差: {mean_error_hours:.2f} ± {std_error_hours:.2f} 小时")
        
        for threshold in [1, 2, 3, 6]:
            accuracy = np.mean(results_data['Phase_Error_Hours'] <= threshold) * 100
            print(f"误差 ≤ {threshold}小时的样本比例: {accuracy:.1f}%")
    
    if celltypes is not None:
        print(f"\n按细胞类型统计:")
        celltype_stats = results_df.groupby('Cell_Type').agg({
            'Predicted_Phase_Hours': ['mean', 'std', 'count']
        }).round(2)
        print(celltype_stats)
    
    create_prediction_plots(results_df, save_dir)
    
    return results_df

def sanitize_filename(s: str) -> str:
    """Sanitize filename for safe file operations"""
    if s is None:
        return 'ALL'
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(s))


def load_metadata_for_phase_comparison(metadata_csv: str) -> pd.DataFrame:
    meta = pd.read_csv(metadata_csv, low_memory=False)
    cols = set(meta.columns.astype(str).str.strip())

    def to_float_series(s):
        return pd.to_numeric(s, errors='coerce')

    # Case 1: study + sample + time
    if {'study', 'sample', 'time'}.issubset(cols):
        m = meta.copy()
        m['study'] = m['study'].astype(str)
        m['sample'] = m['sample'].astype(str)
        m['study_sample'] = m['study'] + '_' + m['sample']
        m['time_mod24'] = to_float_series(m['time'])
        return m[['study_sample', 'time_mod24']]

    # Case 2: study_sample + time_mod24
    if {'study_sample', 'time_mod24'}.issubset(cols):
        m = meta.copy()
        m['study_sample'] = m['study_sample'].astype(str)
        m['time_mod24'] = to_float_series(m['time_mod24'])
        return m[['study_sample', 'time_mod24']]

    # Case 3: study_sample + time
    if {'study_sample', 'time'}.issubset(cols):
        m = meta.copy()
        m['study_sample'] = m['study_sample'].astype(str)
        m['time_mod24'] = to_float_series(m['time'])
        return m[['study_sample', 'time_mod24']]

    # Case 4: Sample + Time_Hours
    if {'Sample', 'Time_Hours'}.issubset(cols):
        m = meta.copy()
        m['study_sample'] = m['Sample'].astype(str)
        m['time_mod24'] = to_float_series(m['Time_Hours'])
        return m[['study_sample', 'time_mod24']]

    # Case 5: Sample + time
    if {'Sample', 'time'}.issubset(cols):
        m = meta.copy()
        m['study_sample'] = m['Sample'].astype(str)
        m['time_mod24'] = to_float_series(m['time'])
        return m[['study_sample', 'time_mod24']]

    raise ValueError("Unsupported metadata format. Expected one of: {'study','sample','time'}; {'study_sample','time_mod24'}; {'study_sample','time'}; {'Sample','Time_Hours'}; {'Sample','time'}")


def load_predictions_for_comparison(pred_csv: str) -> pd.DataFrame:
    """Load predictions CSV and standardize column names"""
    df = pd.read_csv(pred_csv)
    # Infer study_sample
    if 'study_sample' in df.columns:
        df['study_sample'] = df['study_sample'].astype(str)
    elif {'study', 'sample'}.issubset(df.columns):
        df['study'] = df['study'].astype(str)
        df['sample'] = df['sample'].astype(str)
        df['study_sample'] = df['study'] + '_' + df['sample']
    elif 'Sample' in df.columns:
        df['study_sample'] = df['Sample'].astype(str)
    elif 'Sample_ID' in df.columns:
        # my_cyclops phase_predictions_simple.csv uses Sample_ID
        df['study_sample'] = df['Sample_ID'].astype(str)
    elif 'sample' in df.columns:
        df['study_sample'] = df['sample'].astype(str)
    else:
        raise ValueError(f'{pred_csv}: cannot infer study_sample (expected study+sample, Sample, or study_sample)')

    # Infer predicted phase
    if 'Predicted_Phase_Hours' in df.columns:
        pred_hours = pd.to_numeric(df['Predicted_Phase_Hours'], errors='coerce')
    elif 'Predicted_Phase_Radians' in df.columns:
        rad = pd.to_numeric(df['Predicted_Phase_Radians'], errors='coerce')
        pred_hours = rad * 24.0 / (2 * np.pi)
    elif 'Predicted_Phase_Degrees' in df.columns:
        deg = pd.to_numeric(df['Predicted_Phase_Degrees'], errors='coerce')
        pred_hours = deg * (24.0 / 360.0)
    else:
        phase_candidates = [
            'phase', 'Phase', 'phase_hours', 'pred_phase', 'predicted_phase',
            'theta', 'Theta', 'predicted_time', 'pred_time', 'phase_prediction',
        ]
        phase_col = None
        for c in phase_candidates:
            if c in df.columns:
                phase_col = c
                break
        if phase_col is None:
            num_cols = [c for c in df.columns if np.issubdtype(np.asarray(df[c]).dtype, np.number)]
            if num_cols:
                phase_col = num_cols[-1]
            else:
                raise ValueError(f'{pred_csv}: cannot locate predicted phase column')
        pred_hours = pd.to_numeric(df[phase_col], errors='coerce')

    df['pred_phase'] = pred_hours % 24
    return df[['study_sample', 'pred_phase']]




def best_align_phase_for_comparison(
    x_hours: np.ndarray,
    y_hours: np.ndarray,
    step: float = 0.1,
) -> tuple[np.ndarray, float, float, float, float, bool]:
    from scipy.stats import pearsonr, spearmanr

    x_arr: np.ndarray = np.asarray(x_hours, dtype=float)
    y_arr: np.ndarray = np.asarray(y_hours, dtype=float)
    shifts: np.ndarray = np.arange(0.0, 24.0, step, dtype=float)

    best_r = -np.inf
    best = {
        'aligned': x_arr,
        'r': float('nan'),
        'r2': float('nan'),
        'spearman_R': float('nan'),
        'shift': 0.0,
        'flipped': False,
    }

    for flipped in (False, True):
        x0: np.ndarray = (24.0 - x_arr) % 24.0 if flipped else x_arr
        for s in shifts:
            xs = (x0 + s) % 24.0
            xs_np = np.asarray(xs, dtype=float)
            y_np = y_arr
            try:
                r = float(pearsonr(xs_np, y_np)[0])
            except Exception:
                r = np.nan
            if not np.isfinite(r):
                continue
            # Maximize positive correlation after allowing flip
            if r > best_r:
                best_r = r
                # Calculate Spearman correlation for the best alignment
                try:
                    spearman_R = float(spearmanr(xs_np, y_np)[0])
                except Exception:
                    spearman_R = float('nan')
                best.update(
                    aligned=xs,
                    r=r,
                    r2=r*r,
                    spearman_R=spearman_R,
                    shift=float(s),
                    flipped=flipped
                )

    return (
        best['aligned'],
        float(best['r']),
        float(best['r2']),
        float(best['spearman_R']),
        float(best['shift']),
        bool(best['flipped'])
    )


def plot_phase_vs_metadata_comparison(pred_csv: str, celltype: str, meta: pd.DataFrame, out_dir: str):
    try:
        preds = load_predictions_for_comparison(pred_csv)
        base_columns = [c for c in ('study_sample', 'time_mod24') if c in meta.columns]
        if {'study_sample', 'time_mod24'}.issubset(base_columns):
            meta_view = meta[['study_sample', 'time_mod24']].copy()
        elif 'sample' in meta.columns and 'time_mod24' in meta.columns:
            meta_view = meta[['sample', 'time_mod24']].copy()
            meta_view = meta_view.rename(columns={'sample': 'study_sample'})
        else:
            raise ValueError('metadata must contain columns for sample identifiers and time_mod24')

        meta_view['study_sample'] = meta_view['study_sample'].astype(str)
        preds['study_sample'] = preds['study_sample'].astype(str)

        joined = preds.merge(meta_view, on='study_sample', how='left').dropna(subset=['pred_phase', 'time_mod24'])
        if joined.empty:
            print(f"[WARN] No matches for {celltype} ({pred_csv})")
            return None

        phase_hours = np.asarray(joined['pred_phase'], dtype=float)
        metadata_hours = np.asarray(joined['time_mod24'], dtype=float)

        aligned, r, r2, spearman_R, best_shift, flipped = best_align_phase_for_comparison(phase_hours, metadata_hours, step=0.1)

        plt.figure(figsize=(8, 6))
        plt.scatter(aligned, metadata_hours, s=28, alpha=0.85, edgecolors='white', linewidths=0.4, color='tab:blue')
        plt.xlabel('Predicted Phase', fontsize=24)
        plt.ylabel('Collection Time', fontsize=24)

        subtitle = f'Shift={best_shift:.2f}h'
        if flipped:
            subtitle += ' (flipped)'

        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()

        os.makedirs(out_dir, exist_ok=True)
        safe_ct = sanitize_filename(celltype)
        out_path = os.path.join(out_dir, f'phase_vs_time_{safe_ct}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out_path}')
        return out_path, r, r2, spearman_R

    except Exception as e:
        print(f"[ERROR] Failed to plot phase vs metadata for {celltype}: {e}")
        return None


def generate_phase_metadata_comparison(results_df: pd.DataFrame, metadata_csv: str, save_dir: str):
    print(f"\n=== 生成 Phase vs Metadata 对比图 ===")
    print(f"Results DataFrame shape: {results_df.shape}")
    print(f"Results DataFrame columns: {list(results_df.columns)}")
    
    try:
        meta = load_metadata_for_phase_comparison(metadata_csv)
        print(f"Metadata shape: {meta.shape}")
        print(f"Metadata columns: {list(meta.columns)}")
        out_dir = os.path.join(save_dir, 'phase_vs_metadata')
        os.makedirs(out_dir, exist_ok=True)

        metrics_rows = []
        
        if 'Cell_Type' in results_df.columns:
            print("检测到多细胞类型数据，按细胞类型分别处理")
            celltypes = pd.unique(results_df['Cell_Type'].dropna())
            print(f"发现 {len(celltypes)} 个细胞类型: {list(celltypes)}")
            
            # 生成整体图（使用临时文件以复用比较函数，并在完成后删除）
            try:
                tmp_all = os.path.join(out_dir, 'preds_ALL.tmp.csv')
                results_df[['Sample_ID', 'Predicted_Phase_Hours']].to_csv(tmp_all, index=False)
                print(f"处理整体数据(临时): {tmp_all}")
                result = plot_phase_vs_metadata_comparison(tmp_all, 'ALL', meta, out_dir)
                if result:
                    _, r, r2, spearman_R = result
                    print(f"整体 phase-vs-metadata 图已生成: R={r:.3f}, R²={r2:.3f}, Spearman ρ={spearman_R:.3f}")
                if os.path.isfile(tmp_all):
                    os.remove(tmp_all)
            except Exception as e:
                print(f"[WARN] 为整体生成 phase-vs-metadata 图失败: {e}")
            
            # 为每个细胞类型生成图
            for ct in celltypes:
                try:
                    print(f"处理细胞类型: {ct}")
                    safe_name = sanitize_filename(ct)
                    tmp_path = os.path.join(out_dir, f'preds_{safe_name}.csv')
                    subset = results_df[results_df['Cell_Type'] == ct][['Sample_ID', 'Predicted_Phase_Hours']].copy()
                    print(f"  细胞类型 {ct} 样本数: {len(subset)}")
                    subset.to_csv(tmp_path, index=False)
                    
                    result = plot_phase_vs_metadata_comparison(tmp_path, ct, meta, out_dir)
                    if result is not None:
                        _, r, r2, spearman_R = result
                        metrics_rows.append({'celltype': ct, 'R_spuare': r2, 'Pearson_R': r, 'Spearman_R': spearman_R})
                        print(f"  {ct}: R={r:.3f}, R²={r2:.3f}, Spearman ρ={spearman_R:.3f}")
                    else:
                        print(f"  {ct}: 匹配失败")
                        
                    # 清理临时文件
                    if os.path.isfile(tmp_path):
                        os.remove(tmp_path)
                        
                except Exception as e:
                    print(f"[ERROR] 为 celltype={ct} 生成图失败: {e}")
                    
        else:
            print("未检测到 Cell_Type 列，作为单一数据集处理")
            
            celltype_name = 'ALL'
            try:
                save_dir_parts = os.path.normpath(save_dir).split(os.sep)
                if len(save_dir_parts) >= 2:
                    potential_celltype = save_dir_parts[-2]
                    if potential_celltype not in ['result', 'results', 'output', 'data', 'phase_vs_metadata']:
                        celltype_name = potential_celltype
                        print(f"从路径推断细胞类型: {celltype_name}")
            except Exception:
                pass
            
            # 为整个数据集生成图和 metrics（使用临时文件，不保留）
            try:
                tmp_all = os.path.join(out_dir, f'preds_{sanitize_filename(celltype_name)}.tmp.csv')
                results_df[['Sample_ID', 'Predicted_Phase_Hours']].to_csv(tmp_all, index=False)
                print(f"处理数据(临时): {tmp_all} (作为 {celltype_name})")
                result = plot_phase_vs_metadata_comparison(tmp_all, celltype_name, meta, out_dir)
                if result is not None:
                    _, r, r2, spearman_R = result
                    metrics_rows.append({'celltype': celltype_name, 'R_spuare': r2, 'Pearson_R': r, 'Spearman_R': spearman_R})
                    print(f"{celltype_name}: R={r:.3f}, R²={r2:.3f}, Spearman ρ={spearman_R:.3f}")
                else:
                    print(f"{celltype_name}: 匹配失败")
                # 清理临时文件
                if os.path.isfile(tmp_all):
                    os.remove(tmp_all)
            except Exception as e:
                print(f"[ERROR] 为 {celltype_name} 生成图失败: {e}")
                import traceback
                traceback.print_exc()

        # 保存 metrics.csv
        print(f"准备保存 metrics，行数: {len(metrics_rows)}")
        if metrics_rows:
            try:
                metrics_df = pd.DataFrame(metrics_rows)
                metrics_path = os.path.join(out_dir, 'metrics.csv')
                metrics_df.to_csv(metrics_path, index=False)
                print(f"Metrics 表保存到: {metrics_path}")
                
                # 验证文件是否真的保存了
                if os.path.isfile(metrics_path):
                    saved_df = pd.read_csv(metrics_path)
                    print(f"验证: metrics.csv 已保存，包含 {len(saved_df)} 行")
                    print("内容预览:")
                    print(saved_df.to_string())
                else:
                    print("错误: metrics.csv 文件未成功保存")
                
                # 打印简要统计
                print(f"各细胞类型统计:")
                for _, row in metrics_df.iterrows():
                    print(f"  {row['celltype']}: R={row['Pearson_R']:.3f}, R²={row['R_spuare']:.3f}, Spearman ρ={row['Spearman_R']:.3f}")
                    
            except Exception as e:
                print(f"[ERROR] 保存 metrics.csv 失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("警告: 没有 metrics 数据可保存")
            print("可能的原因:")
            print("1. 预测数据与 metadata 无法匹配")
            print("2. 数据格式问题或列缺失 (需要 Sample_ID 与 Predicted_Phase_Hours)")

        print(f"Phase-vs-metadata 输出保存在: {out_dir}")
        
    except Exception as e:
        print(f"[ERROR] 生成 phase-metadata 对比时出错: {e}")
        import traceback
        traceback.print_exc()