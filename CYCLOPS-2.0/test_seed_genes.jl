using DataFrames, CSV, Statistics

# Load GTEx data
gtex_path = "/home/rzh/zhenx/Circadian/data/GTEx/GTEx_adipose_subcutaneous"
gtex_raw_TPM = CSV.read(joinpath(gtex_path, "expression.csv"), DataFrame)
gtex_metadata = CSV.read(joinpath(gtex_path, "metadata.csv"), DataFrame)

# Convert to Float64
for col in names(gtex_raw_TPM)[2:end]
    try
        gtex_raw_TPM[!, col] = parse.(Float64, string.(gtex_raw_TPM[!, col]))
    catch
    end
end

println("Original GTEx: ", size(gtex_raw_TPM))
println("First 5 genes: ", gtex_raw_TPM[1:5, 1])

# Add CollectionTime_C row
if "Time_Hours" in names(gtex_metadata)
    time_row = Any["CollectionTime_C"]
    for sample_name in names(gtex_raw_TPM)[2:end]
        idx = findfirst(==(sample_name), gtex_metadata[!, :Sample])
        if idx !== nothing
            push!(time_row, gtex_metadata[idx, :Time_Hours])
        else
            push!(time_row, 0.0)
        end
    end
    pushfirst!(gtex_raw_TPM, time_row)
    println("\nAfter adding CollectionTime_C: ", size(gtex_raw_TPM))
    println("First 5 rows: ", gtex_raw_TPM[1:5, 1])
end

# Load seed genes
seed_genes = readlines(joinpath(gtex_path, "seed_genes.txt"))
println("\nSeed genes: ", seed_genes)
println("Number of seed genes: ", length(seed_genes))

# Check which seed genes are found
gene_column = uppercase.(string.(gtex_raw_TPM[2:end, 1]))  # Skip first row (CollectionTime_C)
println("\nFirst 10 genes after uppercase: ", gene_column[1:10])

found_indices = Int[]
for (i, gene) in enumerate(uppercase.(string.(seed_genes)))
    idx = findfirst(==(gene), gene_column)
    if idx !== nothing
        push!(found_indices, idx)
        println("Found $gene at index $idx")
    else
        println("NOT FOUND: $gene")
    end
end

println("\nTotal seed genes found: ", length(found_indices))

# Check CV and mean for found genes
if length(found_indices) > 0
    expression_data = Matrix{Float64}(gtex_raw_TPM[found_indices .+ 1, 2:end])  # +1 to account for CollectionTime_C
    println("\nExpression data shape: ", size(expression_data))
    
    means = mean(expression_data, dims=2)
    stds = std(expression_data, dims=2)
    cvs = stds ./ means
    
    println("\nSeed gene statistics:")
    for (i, idx) in enumerate(found_indices)
        gene = gene_column[idx]
        println("$gene: mean=$(round(means[i], digits=4)), CV=$(round(cvs[i], digits=4))")
    end
    
    # Check filters
    println("\nFilter results:")
    println("CV > -Inf: ", sum(cvs .> -Inf))
    println("CV < Inf: ", sum(cvs .< Inf))
    
    # Check mean filter
    all_expression = Matrix{Float64}(gtex_raw_TPM[2:end, 2:end])
    all_means = mean(all_expression, dims=2)
    sorted_means = sort(vec(all_means), rev=true)
    mth_cutoff = sorted_means[min(15000, length(sorted_means))]
    println("Mean cutoff (15000th gene): ", round(mth_cutoff, digits=4))
    println("Genes above cutoff: ", sum(means .> mth_cutoff))
end
