# Zeitzeiger 自动化工具

## 核心文件

- `run_zeitzeiger_auto.py` - Python自动化脚本（leave-one-out）
- `run_zeitzeiger_with_config.sh` - Bash批量运行器
- `zeitzeiger_config.sh` - 配置文件
- `run_zeitzeiger_separate.R` - R核心脚本
- `plot_all_results.py` - 批量绘图工具

## 快速使用

```bash
# 单个数据集
python run_zeitzeiger_auto.py --dataset GSE54651

# 批量运行
bash run_zeitzeiger_with_config.sh GSE54651 GSE54652

# 绘图
python plot_all_results.py --results-dir ./results/GSE54651 --out-dir ./plots
```

详细说明见 `README_AUTO.md`
