using DataFrames, Statistics, StatsBase, LinearAlgebra, MultivariateStats
using PyPlot, Distributed, Random, CSV, Revise, Distributions, Dates, MultipleTesting

data_path = "DATA_PATH_PLACEHOLDER"
output_path = "OUTPUT_PATH_PLACEHOLDER"
path_to_cyclops = "./CYCLOPS.jl"

expression_data = CSV.read(joinpath(data_path, "expression.csv"), DataFrame)
# seed_genes = readlines(joinpath(data_path, "seed_genes.txt"))
# seed_genes = String.(expression_data[!, 1])
seed_genes = [
    "Arntl", "ARNTL",
    "Clock", "CLOCK",
    "Npas2", "NPAS2",
    "Nr1d1", "NR1D1",
    "Bhlhe41", "BHLHE41",
    "Nr1d2", "NR1D2",
    "Dbp", "DBP",
    "Ciart", "CIART",
    "Per1", "PER1",
    "Per3", "PER3",
    "Tef", "TEF",
    "Hlf", "HLF",
    "Cry2", "CRY2",
    "Per2", "PER2",
    "Cry1", "CRY1",
    "Rorc", "RORC",
    "Nfil3", "NFIL3",
]


sample_ids_with_collection_times = []
sample_collection_times = []

if ((length(sample_ids_with_collection_times)+length(sample_collection_times))>0) && 
   (length(sample_ids_with_collection_times) != length(sample_collection_times))
    error("Number of sample ids must match number of collection times.")
end

training_parameters = Dict(
    :regex_cont => r".*_C",
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
    :train_n_models => 80,
    :train_μA => 0.001,
    :train_β => (0.9, 0.999),
    :train_min_steps => 1500,
    :train_max_steps => 2050,
    :train_μA_scale_lim => 1000,
    :train_circular => false,
    :train_collection_times => true,
    :train_collection_time_balance => 1.0,
    :cosine_shift_iterations => 192,
    :cosine_covariate_offset => true,
    :align_p_cutoff => 0.05,
    :align_base => "radians",
    :align_disc => false,
    :align_disc_cov => 1,
    :align_other_covariates => false,
    :align_batch_only => false,
    :X_Val_k => 10,
    :X_Val_omit_size => 0.1,
    :plot_use_o_cov => true,
    :plot_correct_batches => true,
    :plot_disc => false,
    :plot_disc_cov => 1,
    :plot_separate => false,
    :plot_color => ["b", "orange", "g", "r", "m", "y", "k"],
    :plot_only_color => true,
    :plot_p_cutoff => 0.05
)

Distributed.addprocs(length(Sys.cpu_info()))
@everywhere begin
    using DataFrames, Statistics, StatsBase, LinearAlgebra, MultivariateStats
    using PyPlot, Random, CSV, Revise, Distributions, Dates, MultipleTesting
    include($path_to_cyclops)
end

println("Starting CYCLOPS fit...")
eigendata, modeloutputs, correlations, bestmodel, parameters = 
    CYCLOPS.Fit(expression_data, seed_genes, training_parameters)

println("Aligning and saving results...")
CYCLOPS.Align(expression_data, modeloutputs, correlations, bestmodel, parameters, output_path)

println("Generating visualizations...")
clock_genes = ["ARNTL", "CLOCK", "PER1", "PER2", "PER3", "CRY1", "CRY2", "NR1D1", "NR1D2", "DBP", "TEF", "HLF"]

try
    fit_file = joinpath(output_path, filter(x -> startswith(x, "Fit_Output_"), readdir(output_path))[1])
    pred_df = CSV.read(fit_file, DataFrame)
    sample_to_phase = Dict(zip(pred_df.Sample_ID, pred_df.Predicted_Phase_Hours))
    
    meta_file = joinpath(data_path, "metadata.csv")
    cell_types = [nothing]
    if isfile(meta_file)
        meta_df = CSV.read(meta_file, DataFrame)
        if :CellType_D in names(meta_df)
            cell_types = unique(meta_df.CellType_D)
        end
    end
    
    gene_col = findfirst(x -> occursin("gene", lowercase(string(x))) || x == :Gene_Symbol, names(expression_data))
    isnothing(gene_col) && (gene_col = 1)
    
    for cell_type in cell_types
        samples = isnothing(cell_type) ? collect(keys(sample_to_phase)) : meta_df[meta_df.CellType_D .== cell_type, :Sample]
        suffix = isnothing(cell_type) ? "" : "_$(cell_type)"
        
        available_genes = filter(g -> g in expression_data[!, gene_col], clock_genes)
        length(available_genes) == 0 && continue
        
        n_cols = min(3, length(available_genes))
        n_rows = ceil(Int, length(available_genes) / n_cols)
        fig, axes = subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        axes = length(available_genes) == 1 ? [axes] : vec(axes)
        
        for (idx, gene) in enumerate(available_genes)
            ax = axes[idx]
            gene_idx = findfirst(==(gene), expression_data[!, gene_col])
            isnothing(gene_idx) && continue
            
            phases, expressions = Float64[], Float64[]
            for sample in samples
                if haskey(sample_to_phase, sample) && sample in names(expression_data)
                    push!(phases, sample_to_phase[sample])
                    push!(expressions, expression_data[gene_idx, sample])
                end
            end
            
            length(phases) == 0 && continue
            ax.scatter(phases, expressions, alpha=0.6, s=30, edgecolors="black", linewidth=0.5)
            
            try
                x_fit = range(0, 24, length=1000)
                baseline, amp = mean(expressions), (maximum(expressions) - minimum(expressions)) / 2
                phase_guess = phases[argmax(expressions)]
                y_fit = baseline .+ amp .* cos.(2π .* (x_fit .- phase_guess) ./ 24)
                ax.plot(x_fit, y_fit, "-", linewidth=2, alpha=0.8)
                ax.set_title("$(gene)\nPhase: $(round(phase_guess, digits=1))h, Amp: $(round(amp, digits=1))", fontsize=10)
            catch
                ax.set_title(gene, fontsize=10)
            end
            
            ax.set_xlabel("Phase (hours)", fontsize=9)
            ax.set_ylabel("Expression", fontsize=9)
            ax.set_xlim(0, 24)
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.grid(true, alpha=0.3)
        end
        
        for idx in (length(available_genes)+1):length(axes)
            axes[idx].axis("off")
        end
        
        suptitle("Clock Genes$(suffix)", fontsize=14, fontweight="bold")
        tight_layout()
        savefig(joinpath(output_path, "clock_genes$(suffix).png"), dpi=300, bbox_inches="tight")
        close()
    end
    
    figure(figsize=(10, 6))
    if isfile(meta_file) && :CellType_D in names(meta_df)
        merged = leftjoin(pred_df, meta_df, on=:Sample_ID => :Sample)
        for ct in sort(unique(merged.CellType_D))
            hist(merged[merged.CellType_D .== ct, :Predicted_Phase_Hours], bins=24, alpha=0.6, label=string(ct), edgecolor="black")
        end
        legend()
    else
        hist(pred_df.Predicted_Phase_Hours, bins=24, edgecolor="black")
    end
    xlabel("Predicted Phase (hours)", fontsize=12)
    ylabel("Count", fontsize=12)
    title("Phase Distribution", fontsize=14, fontweight="bold")
    xlim(0, 24)
    grid(true, alpha=0.3)
    savefig(joinpath(output_path, "phase_distribution.png"), dpi=300, bbox_inches="tight")
    close()
    
    println("Visualization complete!")
catch e
    println("Visualization skipped: $e")
end

println("CYCLOPS completed!")