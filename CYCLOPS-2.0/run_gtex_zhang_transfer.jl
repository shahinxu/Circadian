#!/home/rzh/zhenx/Circadian/CYCLOPS-2.0/julia-1.6.7/bin/julia
#=
CYCLOPS Transfer Learning Script: GTEx (Normal) -> Zhang (Tumor)
This script trains a model on GTEx normal tissue data and transfers to Zhang cancer cell data
=#

using DataFrames, Statistics, StatsBase, LinearAlgebra, MultivariateStats
using PyPlot, Distributed, Random, CSV, Revise, Distributions, Dates, MultipleTesting

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Path Configuration
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
base_path = joinpath(homedir(), "zhenx", "Circadian")
data_path = joinpath(base_path, "data")
cyclops_path = joinpath(base_path, "CYCLOPS-2.0")
output_path = joinpath(cyclops_path, "results")
output_path_warmup = joinpath(output_path, "warmup")

# Data paths for GTEx (Normal) and Zhang (Tumor)
gtex_dataset = "GTEx_adipose_subcutaneous"  # First GTEx tissue as normal reference
zhang_dataset = "Bcell"  # First Zhang cell type as tumor data

gtex_path = joinpath(data_path, "GTEx", gtex_dataset)
zhang_path = joinpath(data_path, "Zhang_CancerCell_2025_sub", zhang_dataset)

# Alternative: use all Zhang data
# zhang_all_path = joinpath(data_path, "Zhang_CancerCell_2025_all")

println("="^80)
println("CYCLOPS Transfer Learning: GTEx Normal -> Zhang Tumor")
println("="^80)
println("GTEx dataset: ", gtex_dataset)
println("Zhang dataset: ", zhang_dataset)
println("="^80)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Load Data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
println("\n=== Loading GTEx Normal Data ===")
gtex_raw_TPM = CSV.read(joinpath(gtex_path, "expression.csv"), DataFrame)
gtex_metadata = CSV.read(joinpath(gtex_path, "metadata.csv"), DataFrame)

# Convert expression columns to Float64
for col in names(gtex_raw_TPM)[2:end]
    try
        gtex_raw_TPM[!, col] = parse.(Float64, string.(gtex_raw_TPM[!, col]))
    catch e
        println("Warning: Could not convert column $col to Float64")
    end
end

println("GTEx samples: ", size(gtex_raw_TPM, 2) - 1)
println("GTEx genes: ", size(gtex_raw_TPM, 1))

# Add collection time row if available
if "Time_Hours" in names(gtex_metadata)
    time_row = Any["CollectionTime_C"]
    for sample_name in names(gtex_raw_TPM)[2:end]
        idx = findfirst(==(sample_name), gtex_metadata[!, :Sample])
        if idx !== nothing
            push!(time_row, gtex_metadata[idx, :Time_Hours])
        else
            push!(time_row, 0.0)  # Default to 0 if not found
        end
    end
    # Insert collection time row at the top (before genes)
    pushfirst!(gtex_raw_TPM, time_row)
    println("Added CollectionTime_C row to GTEx data")
end

println("\n=== Loading Zhang Tumor Data ===")
zhang_raw_TPM = CSV.read(joinpath(zhang_path, "expression.csv"), DataFrame)
zhang_metadata = CSV.read(joinpath(zhang_path, "metadata.csv"), DataFrame)

# Convert expression columns to Float64
for col in names(zhang_raw_TPM)[2:end]
    try
        zhang_raw_TPM[!, col] = parse.(Float64, string.(zhang_raw_TPM[!, col]))
    catch e
        println("Warning: Could not convert column $col to Float64")
    end
end

println("Zhang samples: ", size(zhang_raw_TPM, 2) - 1)
println("Zhang genes: ", size(zhang_raw_TPM, 1))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Combine GTEx and Zhang data for joint analysis
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
println("\n=== Combining GTEx and Zhang Data ===")

# Add tissue type discriminator column to metadata
gtex_metadata_ext = copy(gtex_metadata)
gtex_metadata_ext[!, :TissueType_D] .= "Normal"

zhang_metadata_ext = copy(zhang_metadata)
zhang_metadata_ext[!, :TissueType_D] .= "Tumor"

# Merge expression data (ensure gene symbols align)
# Get common genes
gtex_genes = Set(gtex_raw_TPM[!, 1])
zhang_genes = Set(zhang_raw_TPM[!, 1])
common_genes = sort(collect(intersect(gtex_genes, zhang_genes)))

println("Common genes: ", length(common_genes))

# Create combined expression dataframe
combined_expr = DataFrame(Gene_Symbol = common_genes)

# Add GTEx samples
gtex_subset = gtex_raw_TPM[in.(gtex_raw_TPM[!, 1], Ref(Set(common_genes))), :]
sort!(gtex_subset, :Gene_Symbol)
for col in names(gtex_subset)[2:end]
    combined_expr[!, col] = gtex_subset[!, col]
end

# Add Zhang samples
zhang_subset = zhang_raw_TPM[in.(zhang_raw_TPM[!, 1], Ref(Set(common_genes))), :]
sort!(zhang_subset, :Gene_Symbol)
for col in names(zhang_subset)[2:end]
    combined_expr[!, col] = zhang_subset[!, col]
end

println("Combined expression shape: ", size(combined_expr))
println("Total samples: ", size(combined_expr, 2) - 1)

# Create combined metadata
# Note: Need to align metadata columns - keep common ones
# For simplicity, create minimal combined metadata with TissueType_D
combined_metadata = DataFrame(
    Sample = vcat(names(gtex_subset)[2:end], names(zhang_subset)[2:end]),
    TissueType_D = vcat(fill("Normal", length(names(gtex_subset)) - 1), 
                        fill("Tumor", length(names(zhang_subset)) - 1))
)

# Save combined data for reference
mkpath(joinpath(output_path, "combined_data"))
CSV.write(joinpath(output_path, "combined_data", "combined_expression.csv"), combined_expr)
CSV.write(joinpath(output_path, "combined_data", "combined_metadata.csv"), combined_metadata)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Load Seed Genes
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Use seed genes from GTEx or Zhang (or mouse core genes)
seed_genes_path = joinpath(gtex_path, "seed_genes.txt")
if isfile(seed_genes_path)
    seed_genes = readlines(seed_genes_path)
    println("\nLoaded ", length(seed_genes), " seed genes from GTEx")
else
    # Fallback to Zhang seed genes
    seed_genes_path = joinpath(zhang_path, "seed_genes.txt")
    if isfile(seed_genes_path)
        seed_genes = readlines(seed_genes_path)
        println("\nLoaded ", length(seed_genes), " seed genes from Zhang")
    else
        println("\nWarning: No seed genes file found, will use all genes")
        seed_genes = String[]
    end
end

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Optional: Collection Times (if available in GTEx)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
sample_ids_with_collection_times = String[]
sample_collection_times = Float64[]

if "Time_Hours" in names(gtex_metadata)
    sample_ids_with_collection_times = String.(gtex_metadata[!, :Sample])
    sample_collection_times = Float64.(gtex_metadata[!, :Time_Hours])
    println("\nFound collection times for ", length(sample_collection_times), " GTEx samples")
end

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Training Parameters
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
training_parameters = Dict(
    :regex_cont => r".*_C",
    :regex_disc => r".*_D",
    
    :blunt_percent => 0.975,
    
    :seed_min_CV => -Inf,
    :seed_max_CV => Inf,
    :seed_mth_Gene => 15000,
    
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
    
    :train_n_models => 80,
    :train_μA => 0.001,
    :train_β => (0.9, 0.999),
    :train_min_steps => 1500,
    :train_max_steps => 2050,
    :train_μA_scale_lim => 1000,
    
    :cosine_shift_iterations => 192,
    :cosine_covariate_offset => true,
    
    :align_p_cutoff => 0.05,
    :align_base => "radians",
    :align_disc => true,  # Align by tissue type
    :align_disc_cov => 1,
    :align_other_covariates => false,
    :align_batch_only => false,
    
    :X_Val_k => 10,
    :X_Val_omit_size => 0.1,
    
    :plot_use_o_cov => true,
    :plot_correct_batches => true,
    :plot_disc => true,
    :plot_disc_cov => 1,
    :plot_separate => true,  # Separate plots for Normal vs Tumor
    :plot_color => ["b", "orange", "g", "r", "m", "y", "k"],
    :plot_only_color => true,
    :plot_p_cutoff => 0.05
)

# If collection times are available, add them to parameters
if length(sample_ids_with_collection_times) > 0
    training_parameters[:train_collection_times] = true
    training_parameters[:train_sample_id] = sample_ids_with_collection_times
    training_parameters[:train_sample_phase] = sample_collection_times
    training_parameters[:train_collection_time_balance] = 1.0
    
    training_parameters[:align_samples] = sample_ids_with_collection_times
    training_parameters[:align_phases] = sample_collection_times
end

# Add mouse core gene alignment
training_parameters[:align_genes] = ["ARNTL", "CLOCK", "NPAS2", "NR1D1", "BHLHE41", "NR1D2", 
                                     "DBP", "CIART", "PER1", "PER3", "TEF", "HLF", 
                                     "CRY2", "PER2", "CRY1", "NFIL3"]
training_parameters[:align_acrophases] = [0, 0.0790637050481884, 0.151440116812406, 2.29555301890004, 
                                          2.90900605826091, 2.98706493493206, 2.99149022777511, 
                                          3.00769248308471, 3.1219769314524, 3.3058682224604, 
                                          3.31357155959037, 3.42557704861225, 3.50078722833753, 
                                          3.88658015146741, 4.99480367551318, 6.00770260397838]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Warmup Parameters
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Initialize Distributed Computing
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
println("\n=== Initializing Distributed Computing ===")
n_workers = min(16, length(Sys.cpu_info()))
Distributed.addprocs(n_workers)
println("Added ", n_workers, " workers")

@everywhere begin
    using DataFrames, Statistics, StatsBase, LinearAlgebra, MultivariateStats
    using PyPlot, Random, CSV, Revise, Distributions, Dates, MultipleTesting
    include(joinpath($cyclops_path, "CYCLOPS.jl"))
end

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Warmup Run
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
println("\n=== Running Warmup ===")
mkpath(output_path_warmup)

# Use full training parameters for warmup with minimal training steps
warmup_params = copy(training_parameters)
warmup_params[:train_min_steps] = 2
warmup_params[:train_max_steps] = 2
warmup_params[:train_n_models] = length(Sys.cpu_info())
# GTEx warmup has no discontinuous covariates (no TissueType_D)
warmup_params[:out_use_disc_cov] = false
warmup_params[:out_all_disc_cov] = false

eigendata_warmup, modeloutputs_warmup, correlations_warmup, bestmodel_warmup, parameters_warmup = 
    CYCLOPS.Fit(gtex_raw_TPM, seed_genes, warmup_params)

CYCLOPS.Align(gtex_raw_TPM, modeloutputs_warmup, correlations_warmup, bestmodel_warmup, 
              parameters_warmup, output_path_warmup)

println("Warmup completed")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Training Run 1: GTEx Only (Normal Tissue)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
println("\n" * "="^80)
println("=== Training Run 1: GTEx Normal Tissue Only ===")
println("="^80)

output_gtex = joinpath(output_path, "GTEx_$(gtex_dataset)_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")
mkpath(output_gtex)

eigendata_gtex, modeloutputs_gtex, correlations_gtex, bestmodel_gtex, parameters_gtex = 
    CYCLOPS.Fit(gtex_raw_TPM, seed_genes, training_parameters)

fit_output_gtex = CYCLOPS.Align(gtex_raw_TPM, modeloutputs_gtex, correlations_gtex, 
                                 bestmodel_gtex, parameters_gtex, output_gtex)

println("\nGTEx model saved to: ", output_gtex)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Training Run 2: Combined GTEx + Zhang (Transfer Learning Setup)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
println("\n" * "="^80)
println("=== Training Run 2: GTEx + Zhang Combined ===")
println("="^80)

output_combined = joinpath(output_path, "GTEx_Zhang_Combined_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")
mkpath(output_combined)

# Prepare GTEx-only data for TransferFit_d1
gtex_raw_TPM_F = gtex_raw_TPM

# Prepare combined data
# Need to add TissueType_D column indicator
# This follows the pattern: first row indicates tissue type, rest is expression
combined_raw_TPM_F = deepcopy(combined_expr)

# Insert a row at the top with tissue type annotations
tissue_type_row = DataFrame()
tissue_type_row[!, :Gene_Symbol] = ["TissueType_D"]
for col in names(combined_expr)[2:end]
    if col in names(gtex_subset)[2:end]
        tissue_type_row[!, col] = ["Normal"]
    else
        tissue_type_row[!, col] = ["Tumor"]
    end
end

combined_raw_TPM_F = vcat(tissue_type_row, combined_raw_TPM_F)

# Prepare Zhang-only data
zhang_raw_TPM_F = deepcopy(zhang_subset)
zhang_tissue_row = DataFrame()
zhang_tissue_row[!, :Gene_Symbol] = ["TissueType_D"]
for col in names(zhang_subset)[2:end]
    zhang_tissue_row[!, col] = ["Tumor"]
end
zhang_raw_TPM_F = vcat(zhang_tissue_row, zhang_raw_TPM_F)

println("\n=== Running Transfer Learning: GTEx -> GTEx+Zhang ===")
eigendata_transfer, modeloutputs_transfer, correlations_transfer, bestmodel_transfer, parameters_transfer = 
    CYCLOPS.TransferFit_d1(gtex_raw_TPM_F, combined_raw_TPM_F, seed_genes, training_parameters)

fit_output_combined = CYCLOPS.Align(combined_raw_TPM_F, modeloutputs_transfer, correlations_transfer, 
                                     bestmodel_transfer, parameters_transfer, output_combined)

println("\nCombined model saved to: ", output_combined)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Apply Trained Model to Zhang Tumor Data Only
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
println("\n" * "="^80)
println("=== Applying Model to Zhang Tumor Data ===")
println("="^80)

output_zhang = joinpath(output_combined, "Zhang_$(zhang_dataset)_predictions")
mkpath(output_zhang)

# Apply the combined model specifically to Zhang data
# This uses ReApplyFit_d1 to apply the trained model to new data
println("\n=== Re-applying fit to Zhang tumor samples ===")

zhang_transform, zhang_metricDataframe, zhang_correlations, ~, zhang_ops = 
    CYCLOPS.ReApplyFit_d1(bestmodel_transfer, gtex_raw_TPM_F, combined_raw_TPM_F, 
                          zhang_raw_TPM_F, seed_genes, parameters_transfer)

# Align and plot Zhang predictions
~, (zhang_plot_path, ~, ~, ~) = CYCLOPS.Align(combined_raw_TPM_F, zhang_raw_TPM_F, 
                                               fit_output_combined, zhang_metricDataframe, 
                                               zhang_correlations, bestmodel_transfer, 
                                               parameters_transfer, zhang_ops, output_zhang)

println("\nZhang predictions saved to: ", output_zhang)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Summary
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
println("\n" * "="^80)
println("=== CYCLOPS Transfer Learning Completed ===")
println("="^80)
println("GTEx normal model: ", output_gtex)
println("Combined GTEx+Zhang model: ", output_combined)
println("Zhang tumor predictions: ", output_zhang)
println("="^80)

# Save a summary report
summary_file = joinpath(output_path, "transfer_learning_summary.txt")
open(summary_file, "w") do f
    write(f, "CYCLOPS Transfer Learning Summary\n")
    write(f, "="^80 * "\n\n")
    write(f, "Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))\n\n")
    write(f, "Datasets:\n")
    write(f, "  GTEx (Normal): $(gtex_dataset)\n")
    write(f, "  Zhang (Tumor): $(zhang_dataset)\n")
    write(f, "  Common genes: $(length(common_genes))\n\n")
    write(f, "Sample counts:\n")
    write(f, "  GTEx samples: $(size(gtex_raw_TPM, 2) - 1)\n")
    write(f, "  Zhang samples: $(size(zhang_raw_TPM, 2) - 1)\n")
    write(f, "  Combined samples: $(size(combined_expr, 2) - 1)\n\n")
    write(f, "Output directories:\n")
    write(f, "  GTEx model: $(output_gtex)\n")
    write(f, "  Combined model: $(output_combined)\n")
    write(f, "  Zhang predictions: $(output_zhang)\n")
end

println("\nSummary saved to: ", summary_file)
println("\nAll done! 🎉")
