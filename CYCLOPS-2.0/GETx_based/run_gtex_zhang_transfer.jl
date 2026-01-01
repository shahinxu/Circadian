using DataFrames, Statistics, StatsBase, LinearAlgebra, MultivariateStats
using DataFrames: Not, ByRow
using PyPlot, Distributed, Random, CSV, Revise, Distributions, Dates, MultipleTesting

base_path = normpath(joinpath(@__DIR__, ".."))
data_path = joinpath(base_path, "data")
cyclops_path = joinpath(base_path, "CYCLOPS-2.0")
output_path = joinpath(cyclops_path, "results")
output_path_warmup = joinpath(output_path, "warmup")

gtex_dataset = "GTEx_spleen"
zhang_datasets = [
    "GSE176078"
]  # Tumor subsets to iterate through

gtex_path = joinpath(data_path, "GTEx", gtex_dataset)
zhang_base_path = joinpath(data_path, "Zhang_CancerCell_2025_sub")

function add_tissue_type_row(df::DataFrame, tissue_type::String)
    # 先将所有数值列转换为 Any 类型以支持混合类型
    for col_name in names(df)[2:end]
        col_type = eltype(df[!, col_name])
        if col_type <: Number
            # 转换为 Any 类型
            df[!, col_name] = Vector{Any}(df[!, col_name])
        end
    end
    
    # 创建包含所有列名的字典
    new_row = Dict{String, Any}()
    new_row[names(df)[1]] = "TissueType_D"
    
    for col_name in names(df)[2:end]
        new_row[col_name] = tissue_type
    end
    
    # 使用 pushfirst! 在开头插入新行
    pushfirst!(df, new_row)
    
    return df
end

function subset_common_genes(df::DataFrame, gene_set::Set{String})
    # 注意：df 的第一行可能是 TissueType_D，需要排除
    first_row_is_metadata = string(df[1, 1]) == "TissueType_D"
    
    if first_row_is_metadata
        # 提取 metadata 行的值，转换为 String
        metadata_values = Dict{String, String}()
        for col_name in names(df)
            metadata_values[col_name] = String(df[1, col_name])
        end
        
        gene_rows = df[2:end, :]
        
        # 更严格的过滤：只保留在 gene_set 中的基因，并且去重
        filtered_df = gene_rows[in.(String.(gene_rows.Gene_Symbol), Ref(gene_set)), :]
        
        # 确保没有重复基因
        filtered_df = unique(filtered_df, :Gene_Symbol)
        
        # 按基因名排序
        sort!(filtered_df, :Gene_Symbol)
        
        # 创建新的结果 DataFrame，保持 gene_rows 的列类型（Any）
        result = copy(filtered_df)
        
        # 在开头插入 metadata 行
        pushfirst!(result, metadata_values)
        
        # 现在将所有样本列（除 Gene_Symbol）的第一行转换为确保是 String 类型
        for col_name in names(result)[2:end]
            if result[1, col_name] isa String
                # 已经是 String，无需处理
            else
                result[1, col_name] = String(result[1, col_name])
            end
        end
        
        return result
    else
        # 更严格的过滤
        filtered_df = df[in.(String.(df.Gene_Symbol), Ref(gene_set)), :]
        filtered_df = unique(filtered_df, :Gene_Symbol)
        sort!(filtered_df, :Gene_Symbol)
        return filtered_df
    end
end


println("="^80)
println("CYCLOPS Transfer Learning: GTEx Normal -> Zhang Tumor")
println("="^80)
println("GTEx dataset: ", gtex_dataset)
println("Zhang datasets: ", join(zhang_datasets, ", "))
println("="^80)

println("\n=== Loading GTEx Normal Data ===")
gtex_raw_TPM = CSV.read(joinpath(gtex_path, "expression.csv"), DataFrame)
gtex_metadata = CSV.read(joinpath(gtex_path, "metadata.csv"), DataFrame)

# 临时添加 TissueType_D 行，标记为 "GTEx" (Normal)
# 此函数会自动将数值列转换为 Any 类型以支持混合内容
gtex_raw_TPM = add_tissue_type_row(gtex_raw_TPM, "GTEx")

println("GTEx samples: ", size(gtex_raw_TPM, 2) - 1)
println("GTEx genes: ", size(gtex_raw_TPM, 1) - 1, " (+ 1 TissueType_D row)")
println("GTEx TissueType_D: GTEx (Normal)")
sample_ids_with_collection_times = String[]
sample_collection_times = Float64[]

if "Time_Hours" in names(gtex_metadata)
    sample_ids_with_collection_times = String.(gtex_metadata[!, :Sample])
    sample_collection_times = Float64.(gtex_metadata[!, :Time_Hours])
    println("\nFound collection times for ", length(sample_collection_times), " GTEx samples")
end

training_parameters = Dict(
    :regex_cont => r"^$",
    :regex_disc => r".*_D",
    
    :blunt_percent => 0.975,
    
    :seed_min_CV => -Inf,
    :seed_max_CV => Inf,
    :seed_mth_Gene => 10000,
    
    :norm_gene_level => true,
    :norm_disc => false,
    :norm_disc_cov => 1,
    
    :eigen_reg => true,
    :eigen_reg_disc_cov => 1,
    :eigen_reg_exclude => false,
    :eigen_reg_r_squared_cutoff => 0.6,
    :eigen_reg_remove_correct => false,
    
    :eigen_first_var => false,
    :eigen_first_var_cutoff => 0.85,
    :eigen_total_var => 0.85,
    :eigen_contr_var => 0.05,
    :eigen_var_override => true,
    :eigen_max => 5,
    
    :out_covariates => true,
    :out_use_disc_cov => true,
    :out_all_disc_cov => true,
    :out_disc_cov => 1,
    :out_use_cont_cov => false,
    :out_all_cont_cov => true,
    :out_use_norm_cont_cov => false,
    :out_all_norm_cont_cov => true,
    :out_cont_cov => 1,
    :out_norm_cont_cov => 1,
    
    :init_scale_change => true,
    :init_scale_1 => false,
    
    :train_n_models => 40,
    :train_μA => 0.001,
    :train_β => (0.9, 0.999),
    :train_min_steps => 1500,
    :train_max_steps => 2050,
    :train_μA_scale_lim => 1000,
    
    :cosine_shift_iterations => 192,
    :cosine_covariate_offset => true,
    
    :align_p_cutoff => 0.05,
    :align_base => "radians",
    :align_disc => true,
    :align_disc_cov => 1,
    :align_other_covariates => false,
    :align_batch_only => false,
    
    :X_Val_k => 10,
    :X_Val_omit_size => 0.1,
    
    :plot_use_o_cov => true,
    :plot_correct_batches => true,
    :plot_disc => true,
    :plot_disc_cov => 1,
    :plot_separate => true,
    :plot_color => ["b", "orange", "g", "r", "m", "y", "k"],
    :plot_only_color => true,
    :plot_p_cutoff => 0.05
)

training_parameters[:align_genes] = ["ARNTL", "CLOCK", "NPAS2", "NR1D1", "BHLHE41", "NR1D2", 
                                     "DBP", "CIART", "PER1", "PER3", "TEF", "HLF", 
                                     "CRY2", "PER2", "CRY1", "NFIL3"]
training_parameters[:align_acrophases] = [0, 0.0790637050481884, 0.151440116812406, 2.29555301890004, 
                                          2.90900605826091, 2.98706493493206, 2.99149022777511, 
                                          3.00769248308471, 3.1219769314524, 3.3058682224604, 
                                          3.31357155959037, 3.42557704861225, 3.50078722833753, 
                                          3.88658015146741, 4.99480367551318, 6.00770260397838]

# 添加cosine shift迭代数以提高稳定性
training_parameters[:cosine_shift_iterations] = 48  # 降低迭代数，减少数值不稳定

println("\n=== Initializing Distributed Computing ===")
n_workers = min(80, length(Sys.cpu_info()))
Distributed.addprocs(n_workers)
println("Added ", n_workers, " workers")

@everywhere begin
    using DataFrames, Statistics, StatsBase, LinearAlgebra, MultivariateStats
    using PyPlot, Random, CSV, Revise, Distributions, Dates, MultipleTesting
    include(joinpath($cyclops_path, "CYCLOPS.jl"))
end
println("\n" * "="^80)
println("=== Transfer Learning: GTEx → GTEx+Zhang ===")
println("="^80)
for zhang_dataset in zhang_datasets
    println("\n" * "-"^80)
    println("Processing Zhang dataset: ", zhang_dataset)
    println("-"^80)

    # 根据数据集名称确定路径
    if startswith(zhang_dataset, "GSE")
        zhang_path = joinpath(data_path, zhang_dataset)
    else
        zhang_path = joinpath(zhang_base_path, zhang_dataset)
    end

    println("\n=== Loading Zhang Tumor Data ===")
    zhang_raw_TPM = CSV.read(joinpath(zhang_path, "expression.csv"), DataFrame)
    
    # metadata.csv 是可选的，只在存在时加载
    metadata_path = joinpath(zhang_path, "metadata.csv")
    if isfile(metadata_path)
        zhang_metadata = CSV.read(metadata_path, DataFrame)
        println("Loaded metadata for ", zhang_dataset)
    else
        println("No metadata file found for ", zhang_dataset, " (optional)")
    end

    # 临时添加 TissueType_D 行，标记为 "Zhang" (Tumor)
    # 此函数会自动将数值列转换为 Any 类型以支持混合内容
    zhang_raw_TPM = add_tissue_type_row(zhang_raw_TPM, "Zhang")

    println("Zhang samples: ", size(zhang_raw_TPM, 2) - 1)
    println("Zhang genes: ", size(zhang_raw_TPM, 1) - 1, " (+ 1 TissueType_D row)")
    println("Zhang TissueType_D: Zhang (Tumor)")

    println("\n=== Combining GTEx and Zhang Data ===")
    # 获取所有基因名，排除 TissueType_D 元数据行
    gtex_genes = Set(String.(gtex_raw_TPM[!, 1]))
    zhang_genes = Set(String.(zhang_raw_TPM[!, 1]))
    
    # 从基因集合中移除 TissueType_D（如果存在）
    delete!(gtex_genes, "TissueType_D")
    delete!(zhang_genes, "TissueType_D")
    
    common_genes = sort(collect(intersect(gtex_genes, zhang_genes)))
    println("Common genes: ", length(common_genes))

    gene_set = Set(common_genes)
    gtex_subset = subset_common_genes(gtex_raw_TPM, gene_set)
    zhang_subset = subset_common_genes(zhang_raw_TPM, gene_set)
    
    # 调试输出：检查基因列表是否一致
    gtex_genes_in_subset = String.(gtex_subset[2:end, 1])  # 跳过 TissueType_D
    zhang_genes_in_subset = String.(zhang_subset[2:end, 1])  # 跳过 TissueType_D
    
    if length(gtex_genes_in_subset) != length(zhang_genes_in_subset)
        println("\n警告: GTEx 和 Zhang subset 的基因数量不一致!")
        println("  GTEx genes: ", length(gtex_genes_in_subset))
        println("  Zhang genes: ", length(zhang_genes_in_subset))
        
        # 找出差异
        gtex_only = setdiff(Set(gtex_genes_in_subset), Set(zhang_genes_in_subset))
        zhang_only = setdiff(Set(zhang_genes_in_subset), Set(gtex_genes_in_subset))
        
        if length(gtex_only) > 0
            println("  只在 GTEx 中的基因: ", collect(gtex_only)[1:min(10, length(gtex_only))])
        end
        if length(zhang_only) > 0
            println("  只在 Zhang 中的基因: ", collect(zhang_only)[1:min(10, length(zhang_only))])
        end
        
        # 强制确保两者基因列表一致
        println("\n  强制重新过滤以确保基因列表完全一致...")
        final_gene_set = Set(intersect(gtex_genes_in_subset, zhang_genes_in_subset))
        gtex_subset = subset_common_genes(gtex_raw_TPM, final_gene_set)
        zhang_subset = subset_common_genes(zhang_raw_TPM, final_gene_set)
        println("  重新过滤后: GTEx=", nrow(gtex_subset), " Zhang=", nrow(zhang_subset))
    end
    
    zhang_subset_no_gene = select(zhang_subset, Not(:Gene_Symbol))

    println("\n=== 合并前数据框检查 ===")
    println("gtex_subset 结构:")
    println("  Size: ", size(gtex_subset))
    println("  Number of rows: ", nrow(gtex_subset))

    println("\nzhang_subset 结构:")
    println("  Size: ", size(zhang_subset))
    println("  Number of rows: ", nrow(zhang_subset))

    println("\nzhang_subset (无 Gene_Symbol 列) 结构:")
    println("  Size: ", size(zhang_subset_no_gene))
    println("  Number of columns: ", ncol(zhang_subset_no_gene))

    println("\n准备执行 hcat...")
    gtex_zhang_combined = hcat(gtex_subset, zhang_subset_no_gene)

    println("Combined data dimensions:")
    println("  GTEx subset: ", size(gtex_subset))
    println("  Zhang subset: ", size(zhang_subset))
    println("  GTEx+Zhang combined: ", size(gtex_zhang_combined))

    seed_genes_path = joinpath(gtex_path, "seed_genes.txt")
    seed_genes = String[]
    if isfile(seed_genes_path)
        seed_genes = readlines(seed_genes_path)
        println("\nLoaded ", length(seed_genes), " seed genes from GTEx")
    else
        zhang_seed_path = joinpath(zhang_path, "seed_genes.txt")
        if isfile(zhang_seed_path)
            seed_genes = readlines(zhang_seed_path)
            println("\nLoaded ", length(seed_genes), " seed genes from Zhang dataset")
        else
            println("\nWarning: No seed genes file found, will use all genes")
        end
    end

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    output_combined = joinpath(output_path, "GTEx_Zhang_Transfer_$(zhang_dataset)_$(timestamp)")
    mkpath(output_combined)

    println("\n=== Running TransferFit_d1 (GTEx → GTEx+Zhang) ===")
    println("Following official CYCLOPS pattern:")
    println("  dataFile1 (GTEx source): $(size(gtex_subset))")
    println("  dataFile2 (GTEx+Zhang combined): $(size(gtex_zhang_combined))")
    println("  Seed genes: $(length(seed_genes))")
    
    if length(sample_ids_with_collection_times) > 0
        training_parameters[:train_sample_id] = sample_ids_with_collection_times
        training_parameters[:train_sample_phase] = sample_collection_times
        println("  Using collection times for $(length(sample_collection_times)) GTEx samples")
    else
        println("  WARNING: No collection time data - model may not learn circadian patterns!")
    end
    
    function fix_covariate_types!(df::DataFrame)
        if string(df[1, 1]) == "TissueType_D"
            for col_name in names(df)[2:end]
                first_val = String(df[1, col_name])
                new_col = Vector{Any}(undef, nrow(df))
                new_col[1] = first_val
                for i in 2:nrow(df)
                    new_col[i] = df[i, col_name]
                end
                df[!, col_name] = new_col
            end
        end
        return df
    end
    
    fix_covariate_types!(gtex_subset)
    fix_covariate_types!(gtex_zhang_combined)

    eigendata_transfer, modeloutputs_transfer, correlations_transfer, bestmodel_transfer, parameters_transfer = 
        CYCLOPS.TransferFit_d1(gtex_subset, gtex_zhang_combined, seed_genes, training_parameters)

    combined_date, combined_paths = CYCLOPS.Align(gtex_zhang_combined, modeloutputs_transfer, correlations_transfer, 
                                                   bestmodel_transfer, parameters_transfer, output_combined)

    println("\nCombined model saved to: ", output_combined)
    println("\n" * "="^80)
    println("=== Dataset Completed ===")
    println("="^80)
    println("Transfer model: ", output_combined)
    println("="^80)

    summary_file = joinpath(output_combined, "transfer_learning_summary.txt")
    open(summary_file, "w") do f
        write(f, "CYCLOPS Transfer Learning Summary\n")
        write(f, "="^80 * "\n\n")
        write(f, "Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))\n\n")
        write(f, "Datasets:\n")
        write(f, "  GTEx (Normal): $(gtex_dataset)\n")
        write(f, "  Zhang (Tumor): $(zhang_dataset)\n")
        write(f, "  Common genes: $(length(common_genes))\n\n")
        write(f, "Sample counts:\n")
        write(f, "  GTEx samples: $(size(gtex_subset, 2) - 1)\n")
        write(f, "  Zhang samples: $(size(zhang_subset, 2) - 1)\n")
        write(f, "  Combined samples: $(size(gtex_zhang_combined, 2) - 1)\n\n")
        write(f, "Output directories:\n")
        write(f, "  Transfer model: $(output_combined)\n")
    end

    println("\nSummary saved to: ", summary_file)
end

println("\nAll datasets processed! 🎉")