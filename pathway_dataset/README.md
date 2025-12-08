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

## 📄 File Format

### `pathway_gene_comprehensive_dataset.csv`

**Column Descriptions**:

| Column | Type | Description | Example |
|--------|------|-------------|----------|
| `pathway_name` | String | Pathway name (unique identifier) | `GOBP_CIRCADIAN_RHYTHM` |
| `gene_symbol` | String | Gene symbol (HGNC standard) | `ARNTL` |
| `source_database` | String | Source database | `GO_BP` |
| `pathway_size` | Integer | Total number of genes in pathway | `207` |

**Data Example**:
```csv
pathway_name,gene_symbol,source_database,pathway_size
GOBP_CIRCADIAN_RHYTHM,ARNTL,GO_BP,207
GOBP_CIRCADIAN_RHYTHM,CLOCK,GO_BP,207
REACTOME_CIRCADIAN_CLOCK,PER1,Reactome,70
KEGG_CIRCADIAN_RHYTHM_MAMMAL,CRY1,KEGG,13
```

---

## 🚀 Quick Start

### 1. View with Excel/LibreOffice

```bash
# Open with LibreOffice
libreoffice pathway_gene_comprehensive_dataset.csv

# Or view first lines with text viewer
head -20 pathway_gene_comprehensive_dataset.csv
```

### 2. Command Line Queries

```bash
# Find all pathways for a specific gene
grep "ARNTL" pathway_gene_comprehensive_dataset.csv

# Find circadian-related pathways
grep -i "circadian" pathway_gene_comprehensive_dataset.csv

# Count data by database
cut -d',' -f3 pathway_gene_comprehensive_dataset.csv | sort | uniq -c
```

### 3. Python Analysis

```python
import pandas as pd

# Load dataset
df = pd.read_csv('pathway_gene_comprehensive_dataset.csv')

print(f"Total mappings: {len(df):,}")
print(f"Unique pathways: {df['pathway_name'].nunique():,}")
print(f"Unique genes: {df['gene_symbol'].nunique():,}")

# Query specific gene
arntl = df[df['gene_symbol'] == 'ARNTL']
print(f"\nARNTL participates in {len(arntl)} pathways:")
print(arntl[['pathway_name', 'source_database']].head(10))

# Search circadian pathways
circadian = df[df['pathway_name'].str.contains('circadian', case=False)]
print(f"\nFound {circadian['pathway_name'].nunique()} circadian-related pathways")
```

### 4. R Language Analysis

```r
library(tidyverse)

# Load data
df <- read_csv('pathway_gene_comprehensive_dataset.csv')

# View basic information
df %>% 
  summarise(
    pathways = n_distinct(pathway_name),
    genes = n_distinct(gene_symbol),
    mappings = n()
  )

# Query circadian pathways
circadian <- df %>%
  filter(str_detect(pathway_name, regex('circadian', ignore_case = TRUE)))

# Count by database
df %>%
  count(source_database, sort = TRUE)
```

---

## 💡 Common Use Cases

### 1. Gene Function Annotation

**Purpose**: Find all pathways a gene participates in

```python
import pandas as pd

df = pd.read_csv('pathway_gene_comprehensive_dataset.csv')

# Query your gene of interest
gene_name = 'CLOCK'
gene_pathways = df[df['gene_symbol'] == gene_name]

print(f"Pathways involving {gene_name}:")
for idx, row in gene_pathways.iterrows():
    print(f"- {row['pathway_name']} ({row['source_database']})")
```

### 2. Pathway Enrichment Analysis

**Purpose**: Identify enriched pathways in your gene list

```python
import pandas as pd
from scipy.stats import hypergeom

# Load data
df = pd.read_csv('pathway_gene_comprehensive_dataset.csv')

# Your gene list (e.g., differentially expressed genes)
my_genes = ['ARNTL', 'CLOCK', 'PER1', 'PER2', 'CRY1', 'CRY2', 
            'NR1D1', 'NR1D2', 'DBP', 'TEF', 'HLF']

# Total genes (background)
total_genes = df['gene_symbol'].nunique()
query_size = len(my_genes)

# Calculate overlap for each pathway
results = []
for pathway_name in df['pathway_name'].unique():
    pathway_genes = df[df['pathway_name'] == pathway_name]['gene_symbol'].unique()
    pathway_size = len(pathway_genes)
    
    # Calculate overlap
    overlap = set(my_genes) & set(pathway_genes)
    overlap_count = len(overlap)
    
    if overlap_count >= 2:  # At least 2 genes overlap
        # Hypergeometric test
        p_value = hypergeom.sf(overlap_count - 1, total_genes, 
                              pathway_size, query_size)
        
        results.append({
            'pathway_name': pathway_name,
            'pathway_size': pathway_size,
            'overlap': overlap_count,
            'overlap_genes': ', '.join(overlap),
            'p_value': p_value
        })

# Convert to DataFrame and sort
result_df = pd.DataFrame(results).sort_values('p_value')
print("\nEnriched pathways (p < 0.01):")
print(result_df[result_df['p_value'] < 0.01].head(20))
```

### 3. Find Circadian-Related Pathways and Genes

```python
import pandas as pd

df = pd.read_csv('pathway_gene_comprehensive_dataset.csv')

# Search keywords
keywords = ['circadian', 'rhythm', 'clock', 'period', 'cryptochrome']
pattern = '|'.join(keywords)

# Find matching pathways
circadian_data = df[df['pathway_name'].str.contains(pattern, case=False)]

print(f"Found {circadian_data['pathway_name'].nunique()} related pathways")
print(f"Involving {circadian_data['gene_symbol'].nunique()} genes")

# Statistics by database
print("\nCircadian pathways by database:")
print(circadian_data.groupby('source_database')['pathway_name'].nunique())

# List all pathways
print("\nPathway list:")
for pathway in circadian_data['pathway_name'].unique():
    gene_count = len(circadian_data[circadian_data['pathway_name'] == pathway])
    print(f"- {pathway}: {gene_count} genes")
```

### 4. Pathway Comparison

```python
import pandas as pd

df = pd.read_csv('pathway_gene_comprehensive_dataset.csv')

# Compare gene overlap between two pathways
pathway1_name = 'GOBP_CIRCADIAN_RHYTHM'
pathway2_name = 'REACTOME_CIRCADIAN_CLOCK'

genes1 = set(df[df['pathway_name'] == pathway1_name]['gene_symbol'])
genes2 = set(df[df['pathway_name'] == pathway2_name]['gene_symbol'])

overlap = genes1 & genes2
unique1 = genes1 - genes2
unique2 = genes2 - genes1

print(f"{pathway1_name}: {len(genes1)} genes")
print(f"{pathway2_name}: {len(genes2)} genes")
print(f"Overlap: {len(overlap)} genes")
print(f"Overlapping genes: {', '.join(sorted(overlap))}")
```

---

## 🔍 Circadian Pathway Examples

The dataset contains abundant circadian-related pathways:

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

## 📈 Data Quality Notes

### Advantages
✅ **Comprehensive Coverage**: Integrates 7 major databases, 18,888 genes  
✅ **Up-to-date**: Uses MSigDB 2024.1 version  
✅ **Standardized**: Unified HGNC gene nomenclature  
✅ **Multiple Sources**: Compare annotations across different databases  
✅ **Easy to Use**: Standard CSV format, compatible with various tools

### Considerations
⚠️ **Database Differences**: Pathway definitions may vary across databases  
⚠️ **Species Specificity**: Primarily human-focused, mouse genes require conversion  
⚠️ **Update Frequency**: Databases update at different rates, periodic updates recommended  
⚠️ **File Size**: 44MB, recommended to use programming languages rather than Excel

---

## 🛠️ Technical Details

### Data Integration Pipeline
1. Download MSigDB canonical pathway collections
2. Parse GMT format files
3. Standardize gene nomenclature (HGNC)
4. Identify database sources
5. Deduplicate and integrate
6. Export as CSV format

### Gene Naming Standard
- Uses **HGNC (HUGO Gene Nomenclature Committee)** official gene symbols
- All gene names in uppercase
- Complies with international standards

### Pathway Naming Rules
- GO: `GOBP_` prefix (Gene Ontology Biological Process)
- Reactome: `REACTOME_` prefix
- KEGG: `KEGG_` prefix
- WikiPathways: `WP_` prefix
- BioCarta: `BIOCARTA_` prefix
- PID: `PID_` prefix

**Note**: pathway_name serves as unique identifier, no separate pathway_id

---

## 📚 Recommended Reading

### Choosing the Right Database

**Need most comprehensive coverage?** → **GO_BP**
- Most genes and pathways
- Suitable for global functional analysis

**Need precise pathways?** → **Reactome**
- Manually curated, high quality
- Clear pathway definitions

**Need metabolic pathways?** → **KEGG**
- Most authoritative for metabolism
- Detailed pathway diagrams

**Need latest research?** → **WikiPathways**
- Frequently updated by community
- Includes new discoveries

**Recommended for circadian research**: GO_BP + Reactome + KEGG

---

## 📖 Citation

Please cite the original databases when using this dataset:

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

## 📝 Version Information

- **Dataset Version**: 1.0
- **Creation Date**: 2025-12-07
- **Data Source**: MSigDB v2024.1
- **File Format**: CSV (UTF-8 encoding)

---

## 🔗 Related Resources

- **MSigDB**: https://www.gsea-msigdb.org/
- **Gene Ontology**: http://geneontology.org/
- **Reactome**: https://reactome.org/
- **KEGG**: https://www.genome.jp/kegg/
- **WikiPathways**: https://www.wikipathways.org/

---

## 📧 Contact

For questions or suggestions, please refer to the official documentation of each original database.

---

## ⚖️ License

This dataset integrates multiple public databases, each with different licenses:
- **Gene Ontology**: CC BY 4.0
- **Reactome**: CC BY 4.0  
- **WikiPathways**: CC0 1.0
- **MSigDB**: Academic use license

**Usage Restrictions**: For academic research use only. Please comply with the terms of use of each original database.

---

**Last Updated**: December 7, 2025
