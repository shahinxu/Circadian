# Iterative TOAST Training Pipeline

## 概述

这是一个迭代式训练管道，通过逐步扩展节律基因集来改进TOAST模型的性能。

## 工作流程

```
Iteration 1: Seed genes (e.g., 100 genes)
    ↓ Train TOAST
    ↓ Predict phases for all samples
    ↓ Calculate rhythmicity for all genes
    ↓ Select top-k most rhythmic genes (e.g., 500)
    
Iteration 2: Seed + 500 new genes = 600 genes
    ↓ Train TOAST
    ↓ Predict phases
    ↓ Select another 500 genes
    
Iteration 3: 1100 genes
    ↓ ...
    
Iteration 5: Final model with ~2100 genes
```

## 核心思想

1. **渐进式特征选择**: 不是一次性使用所有基因，而是基于相位信息逐步扩展
2. **相位引导**: 使用预测的sample phase来识别具有节律性的基因
3. **多轮验证**: 每一轮都生成对比图，可以看到performance的变化

## 节律性计算方法

### Method 1: Circular-Linear Correlation
```python
# 计算基因表达与相位的相关性
r_sin = corr(expression, sin(phase))
r_cos = corr(expression, cos(phase))
rhythmicity_score = max(|r_sin|, |r_cos|)
```

### Method 2: ANOVA F-statistic
```python
# 将样本按相位分成8个bin，计算组间方差
# F值越大，说明基因表达在不同相位下差异越显著
```

## 使用方法

### 基本用法

```bash
cd /home/rzh/zhenx/Circadian/TOAST_inter

conda activate llama

python iterative_toast.py \
  --data_path ../data/GSE54651_all_tissues \
  --num_iterations 5 \
  --genes_per_iteration 500 \
  --num_epochs 500 \
  --device cuda
```

### 参数说明

- `--data_path`: 数据集路径（需要包含expression.csv, metadata.csv, seed_genes.txt）
- `--num_iterations`: 迭代次数（默认5次）
- `--genes_per_iteration`: 每次添加的基因数（默认500）
- `--num_epochs`: 每次训练的epoch数（默认500）
- `--lr`: 学习率（默认1e-3）
- `--device`: 使用的设备（cuda或cpu）
- `--output_dir`: 输出目录

### 数据格式要求

数据集文件夹需要包含：

1. **expression.csv**: 
   - Index: sample names
   - Columns: gene names
   - Values: normalized expression values

2. **metadata.csv**:
   - Index: sample names (must match expression.csv)
   - Columns: 
     - `phase`: true phase in radians (optional, for comparison)
     - `tissue`: tissue labels (optional, for coloring plots)

3. **seed_genes.txt**:
   - One gene name per line
   - Initial set of rhythmic genes to start with

## 输出结果

每次运行会在`output_dir`下创建一个时间戳文件夹，包含：

### 每个iteration的输出

1. **model_iter{N}.pt**: 训练好的模型权重
2. **predictions_iter{N}.csv**: 样本的phase预测结果
3. **comparison_iter{N}.png**: 预测vs真实相位的对比图
4. **genes_iter{N+1}.txt**: 下一轮要使用的基因列表

### 汇总文件

- **summary.csv**: 每个iteration的基因数量汇总

## 示例输出

```
ITERATION 1/5
Current gene set size: 100
[1.1] Training TOAST model...
    Epoch 50/500, Loss: 0.8532, Phase Std: 1.2341
    Epoch 100/500, Loss: 0.7123, Phase Std: 1.5678
    ...
  Model saved: results_iterative/iter_20251208_120000/model_iter1.pt
  Predictions saved: results_iterative/iter_20251208_120000/predictions_iter1.csv
[1.2] Generating comparison plot...
  Comparison plot saved: results_iterative/iter_20251208_120000/comparison_iter1.png
  Correlations - Pearson: 0.456, Spearman: 0.432, Circular: 0.512
[1.3] Selecting new rhythmic genes...
  Selected 500 new genes
  Rhythmicity range: 0.3245 - 0.7891

ITERATION 2/5
Current gene set size: 600
...
```

## 可视化分析

每个iteration的comparison图展示：
- X轴: 真实相位（小时）
- Y轴: 预测相位（小时）
- 颜色: 不同tissue
- 标题: 当前基因数和相关性指标

通过对比5张图，可以看出：
- 基因数量增加是否改善了预测
- 哪些tissue的预测效果最好
- 是否存在过拟合现象

## 优化建议

1. **调整genes_per_iteration**: 如果计算资源有限，可以减少每次添加的基因数
2. **调整num_epochs**: 如果看到loss plateau很快，可以减少epoch数
3. **调整rhythmicity method**: 可以尝试'correlation'或'anova'两种方法
4. **Early stopping**: 如果某个iteration的performance下降，可以提前停止

## 注意事项

- 确保有足够的GPU内存（基因数增加会增加内存需求）
- 每个iteration大约需要5-10分钟（取决于数据大小和epoch数）
- 完整的5次iteration可能需要30-60分钟
