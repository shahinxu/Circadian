# Comprehensive Pathway-Gene Dataset

## 📊 Dataset Overview

This is a comprehensive dataset integrating **multiple mainstream biological pathway databases**, containing complete pathway-gene mapping relationships.

### Core Statistics

- **Total Pathway-Gene Mappings**: 800,990 entries
- **Unique Pathways**: 11,525 
- **Unique Genes**: 18,888 
- **Data Sources**: 7 major databases
- **Species**: Human (Homo sapiens)
- **File Size**: 44 MB

---

## 🗂️ Data Sources

| Database | Pathways | Genes | Mappings | Avg Pathway Size | Description |
|----------|----------|-------|----------|------------------|-------------|
| **GO_BP** | 7,608 | 17,951 | 630,308 | 82.8 | Gene Ontology Biological Process - Most comprehensive |
| **Reactome** | 1,736 | 11,290 | 97,590 | 56.2 | Manually curated pathways - High quality |
| **KEGG** | 844 | 6,029 | 22,457 | 26.6 | Metabolic and signaling pathways - Authoritative |
| **WikiPathways** | 804 | 8,747 | 36,679 | 45.6 | Community maintained - Frequently updated |
| **BioCarta** | 292 | 1,509 | 4,814 | 16.5 | Classic signaling pathways |
| **PID** | 222 | 2,732 | 8,563 | 38.6 | NCI Pathway Interaction Database |
| **Other** | 19 | 349 | 579 | 30.5 | Other sources |

**Data Version**: MSigDB v2024.1 (Latest 2024 version)

---

## 📄 文件格式

### `pathway_gene_comprehensive_dataset.csv`

**列说明**:

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `pathway_id` | 字符串 | 通路唯一标识符 | `GOBP_CIRCADIAN_RHYTHM` |
| `pathway_name` | 字符串 | 通路名称 | `GOBP_CIRCADIAN_RHYTHM` |
| `gene_symbol` | 字符串 | 基因符号 (HGNC标准) | `ARNTL` |
| `source_database` | 字符串 | 数据库来源 | `GO_BP` |
| `pathway_size` | 整数 | 通路包含的基因总数 | `207` |

**数据示例**:
```csv
pathway_id,pathway_name,gene_symbol,source_database,pathway_size
GOBP_CIRCADIAN_RHYTHM,GOBP_CIRCADIAN_RHYTHM,ARNTL,GO_BP,207
GOBP_CIRCADIAN_RHYTHM,GOBP_CIRCADIAN_RHYTHM,CLOCK,GO_BP,207
REACTOME_CIRCADIAN_CLOCK,REACTOME_CIRCADIAN_CLOCK,PER1,Reactome,70
KEGG_CIRCADIAN_RHYTHM_MAMMAL,KEGG_CIRCADIAN_RHYTHM_MAMMAL,CRY1,KEGG,13
```

---

## 🚀 快速使用

### 1. Excel/LibreOffice 查看

```bash
# 用 LibreOffice 打开
libreoffice pathway_gene_comprehensive_dataset.csv

# 或用文本查看器查看前几行
head -20 pathway_gene_comprehensive_dataset.csv
```

### 2. 命令行查询

```bash
# 查找特定基因的所有通路
grep "ARNTL" pathway_gene_comprehensive_dataset.csv

# 查找昼夜节律相关通路
grep -i "circadian" pathway_gene_comprehensive_dataset.csv

# 统计各数据库的数据量
cut -d',' -f4 pathway_gene_comprehensive_dataset.csv | sort | uniq -c
```

### 3. Python 分析

```python
import pandas as pd

# 加载数据集
df = pd.read_csv('pathway_gene_comprehensive_dataset.csv')

print(f"总映射数: {len(df):,}")
print(f"唯一通路: {df['pathway_id'].nunique():,}")
print(f"唯一基因: {df['gene_symbol'].nunique():,}")

# 查询特定基因
arntl = df[df['gene_symbol'] == 'ARNTL']
print(f"\nARNTL 基因参与 {len(arntl)} 个通路:")
print(arntl[['pathway_name', 'source_database']].head(10))

# 搜索昼夜节律通路
circadian = df[df['pathway_name'].str.contains('circadian', case=False)]
print(f"\n找到 {circadian['pathway_id'].nunique()} 个昼夜节律相关通路")
```

### 4. R 语言分析

```r
library(tidyverse)

# 加载数据
df <- read_csv('pathway_gene_comprehensive_dataset.csv')

# 查看基本信息
df %>% 
  summarise(
    pathways = n_distinct(pathway_id),
    genes = n_distinct(gene_symbol),
    mappings = n()
  )

# 查询昼夜节律通路
circadian <- df %>%
  filter(str_detect(pathway_name, regex('circadian', ignore_case = TRUE)))

# 统计各数据库
df %>%
  count(source_database, sort = TRUE)
```

---

## 💡 常见应用场景

### 1. 基因功能注释

**目的**: 查找某个基因参与的所有通路

```python
import pandas as pd

df = pd.read_csv('pathway_gene_comprehensive_dataset.csv')

# 查询你感兴趣的基因
gene_name = 'CLOCK'
gene_pathways = df[df['gene_symbol'] == gene_name]

print(f"{gene_name} 参与的通路:")
for idx, row in gene_pathways.iterrows():
    print(f"- {row['pathway_name']} ({row['source_database']})")
```

### 2. 通路富集分析

**目的**: 找出基因列表富集的通路

```python
import pandas as pd
from scipy.stats import hypergeom

# 加载数据
df = pd.read_csv('pathway_gene_comprehensive_dataset.csv')

# 你的基因列表（例如差异表达基因）
my_genes = ['ARNTL', 'CLOCK', 'PER1', 'PER2', 'CRY1', 'CRY2', 
            'NR1D1', 'NR1D2', 'DBP', 'TEF', 'HLF']

# 总基因数（背景）
total_genes = df['gene_symbol'].nunique()
query_size = len(my_genes)

# 计算每个通路的重叠
results = []
for pathway_id in df['pathway_id'].unique():
    pathway_genes = df[df['pathway_id'] == pathway_id]['gene_symbol'].unique()
    pathway_size = len(pathway_genes)
    
    # 计算重叠
    overlap = set(my_genes) & set(pathway_genes)
    overlap_count = len(overlap)
    
    if overlap_count >= 2:  # 至少2个基因重叠
        # 超几何检验
        p_value = hypergeom.sf(overlap_count - 1, total_genes, 
                              pathway_size, query_size)
        
        results.append({
            'pathway_id': pathway_id,
            'pathway_name': df[df['pathway_id'] == pathway_id]['pathway_name'].iloc[0],
            'pathway_size': pathway_size,
            'overlap': overlap_count,
            'overlap_genes': ', '.join(overlap),
            'p_value': p_value
        })

# 转换为DataFrame并排序
result_df = pd.DataFrame(results).sort_values('p_value')
print("\n富集的通路 (p < 0.01):")
print(result_df[result_df['p_value'] < 0.01].head(20))
```

### 3. 查找昼夜节律相关通路和基因

```python
import pandas as pd

df = pd.read_csv('pathway_gene_comprehensive_dataset.csv')

# 搜索关键词
keywords = ['circadian', 'rhythm', 'clock', 'period', 'cryptochrome']
pattern = '|'.join(keywords)

# 查找匹配的通路
circadian_data = df[df['pathway_name'].str.contains(pattern, case=False)]

print(f"找到 {circadian_data['pathway_id'].nunique()} 个相关通路")
print(f"涉及 {circadian_data['gene_symbol'].nunique()} 个基因")

# 按数据库统计
print("\n各数据库的昼夜节律通路:")
print(circadian_data.groupby('source_database')['pathway_id'].nunique())

# 列出所有通路
print("\n通路列表:")
for pathway in circadian_data['pathway_name'].unique():
    gene_count = len(circadian_data[circadian_data['pathway_name'] == pathway])
    print(f"- {pathway}: {gene_count} genes")
```

### 4. 通路比较

```python
import pandas as pd

df = pd.read_csv('pathway_gene_comprehensive_dataset.csv')

# 比较两个通路的基因重叠
pathway1_name = 'GOBP_CIRCADIAN_RHYTHM'
pathway2_name = 'REACTOME_CIRCADIAN_CLOCK'

genes1 = set(df[df['pathway_name'] == pathway1_name]['gene_symbol'])
genes2 = set(df[df['pathway_name'] == pathway2_name]['gene_symbol'])

overlap = genes1 & genes2
unique1 = genes1 - genes2
unique2 = genes2 - genes1

print(f"{pathway1_name}: {len(genes1)} 个基因")
print(f"{pathway2_name}: {len(genes2)} 个基因")
print(f"重叠: {len(overlap)} 个基因")
print(f"重叠基因: {', '.join(sorted(overlap))}")
```

---

## 🔍 昼夜节律通路示例

数据集中包含丰富的昼夜节律相关通路：

### Gene Ontology (GO_BP)
- `GOBP_CIRCADIAN_RHYTHM` - 207 genes
- `GOBP_REGULATION_OF_CIRCADIAN_RHYTHM` - 114 genes
- `GOBP_CIRCADIAN_REGULATION_OF_GENE_EXPRESSION` - 70 genes
- `GOBP_ENTRAINMENT_OF_CIRCADIAN_CLOCK` - 30 genes
- `GOBP_CIRCADIAN_SLEEP_WAKE_CYCLE` - 22 genes
- `GOBP_POSITIVE_REGULATION_OF_CIRCADIAN_RHYTHM` - 18 genes
- `GOBP_CIRCADIAN_SLEEP_WAKE_CYCLE_SLEEP` - 16 genes
- `GOBP_NEGATIVE_REGULATION_OF_CIRCADIAN_RHYTHM` - 12 genes

### Reactome
- `REACTOME_CIRCADIAN_CLOCK` - 70 genes
- `REACTOME_BMAL1_CLOCK_NPAS2_ACTIVATES_CIRCADIAN_GENE_EXPRESSION` - 27 genes

### KEGG
- `KEGG_CIRCADIAN_RHYTHM_MAMMAL` - 13 genes

### WikiPathways
- `WP_CIRCADIAN_RHYTHM_GENES` - 201 genes
- `WP_MELATONIN_METABOLISM_AND_EFFECTS` - 37 genes
- `WP_EXERCISEINDUCED_CIRCADIAN_REGULATION` - 48 genes
- `WP_CLOCKCONTROLLED_AUTOPHAGY_IN_BONE_METABOLISM` - 80 genes

### PID & BioCarta
- `PID_CIRCADIAN_PATHWAY` - 16 genes
- `BIOCARTA_CIRCADIAN_PATHWAY` - 6 genes

---

## 📈 数据质量说明

### 优点
✅ **覆盖全面**: 整合7个主流数据库，18,888个基因  
✅ **数据最新**: 使用MSigDB 2024.1版本  
✅ **标准化**: 统一使用HGNC基因命名标准  
✅ **多数据源**: 可对比不同数据库的注释差异  
✅ **易于使用**: 标准CSV格式，兼容各种工具

### 注意事项
⚠️ **数据库差异**: 不同数据库对通路的定义可能不同  
⚠️ **物种特异性**: 主要针对人类，小鼠基因需转换  
⚠️ **更新频率**: 数据库更新速度不同，建议定期更新  
⚠️ **文件大小**: 74MB，建议使用编程语言处理而非Excel

---

## 🛠️ 技术细节

### 数据整合流程
1. 下载MSigDB标准通路集合
2. 解析GMT格式文件
3. 统一基因命名（HGNC标准）
4. 识别数据库来源
5. 去重并整合
6. 导出为CSV格式

### 基因命名标准
- 使用 **HGNC (HUGO Gene Nomenclature Committee)** 官方基因符号
- 所有基因名均为大写
- 符合国际标准

### 通路ID规则
- GO: `GOBP_` 前缀（Gene Ontology Biological Process）
- Reactome: `REACTOME_` 前缀
- KEGG: `KEGG_` 前缀
- WikiPathways: `WP_` 前缀
- BioCarta: `BIOCARTA_` 前缀
- PID: `PID_` 前缀

---

## 📚 推荐阅读

### 选择合适的数据库

**需要最全面覆盖？** → **GO_BP**
- 最多的基因和通路
- 适合全局功能分析

**需要精确的通路？** → **Reactome**
- 人工审核，质量高
- 通路定义清晰

**需要代谢通路？** → **KEGG**
- 代谢通路最权威
- 通路图详细

**需要最新研究？** → **WikiPathways**
- 社区更新快
- 包含新发现

**昼夜节律研究推荐组合**: GO_BP + Reactome + KEGG

---

## 📖 引用

使用本数据集请引用原始数据库：

**MSigDB**:
- Liberzon, A., et al. (2015). The Molecular Signatures Database Hallmark Gene Set Collection. *Cell Systems*, 1(6), 417-425.

**Gene Ontology**:
- Ashburner, M., et al. (2000). Gene Ontology: tool for the unification of biology. *Nature Genetics*, 25(1), 25-29.

**Reactome**:
- Jassal, B., et al. (2020). The Reactome Pathway Knowledgebase. *Nucleic Acids Research*, 48(D1), D498-D503.

**KEGG**:
- Kanehisa, M., et al. (2021). KEGG: integrating viruses and cellular organisms. *Nucleic Acids Research*, 49(D1), D545-D551.

**WikiPathways**:
- Martens, M., et al. (2021). WikiPathways: connecting communities. *Nucleic Acids Research*, 49(D1), D613-D621.

---

## 📝 版本信息

- **数据集版本**: 1.0
- **创建日期**: 2025-12-07
- **数据来源**: MSigDB v2024.1
- **文件格式**: CSV (UTF-8编码)

---

## 🔗 相关资源

- **MSigDB**: https://www.gsea-msigdb.org/
- **Gene Ontology**: http://geneontology.org/
- **Reactome**: https://reactome.org/
- **KEGG**: https://www.genome.jp/kegg/
- **WikiPathways**: https://www.wikipathways.org/

---

## 📧 联系方式

如有问题或建议，请参考各原始数据库的官方文档。

---

## ⚖️ 许可协议

本数据集整合自多个公开数据库，各有不同许可：
- **Gene Ontology**: CC BY 4.0
- **Reactome**: CC BY 4.0  
- **WikiPathways**: CC0 1.0
- **MSigDB**: 学术使用许可

**使用限制**: 仅供学术研究使用，请遵守各原始数据库的使用条款。

---

**最后更新**: 2025年12月7日
