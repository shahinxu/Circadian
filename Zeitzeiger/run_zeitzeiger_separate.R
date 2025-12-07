#!/usr/bin/env Rscript
# Simple ZeitZeiger runner for the common case: one training dataset and one test dataset
# Usage example:
# Rscript scripts/run_zeitzeiger_separate.R --expr-train train_expression.csv --meta-train train_metadata.csv \
#   --expr-test test_expression.csv --meta-test test_metadata.csv --sample-col Sample --time-col Time_Phase \
#   --time-format radians --out-prefix out/zz_run --save-model --verbose

suppressPackageStartupMessages({
  if (!requireNamespace('optparse', quietly = TRUE)) stop('Please install optparse')
  if (!requireNamespace('zeitzeiger', quietly = TRUE)) stop('Please install zeitzeiger')
  library(optparse)
  library(zeitzeiger)
})

opt_list <- list(
  make_option(c('--expr-train'), type='character', help='Training expression CSV (genes x samples or table with gene column)'),
  make_option(c('--meta-train'), type='character', help='Training metadata CSV'),
  make_option(c('--expr-test'), type='character', help='Test expression CSV'),
  make_option(c('--meta-test'), type='character', help='Test metadata CSV'),
  make_option(c('--sample-col'), type='character', default='Sample', help='Sample ID column name in metadata'),
  make_option(c('--time-col'), type='character', default='Time_Phase', help='Time column name in metadata'),
  make_option(c('--time-format'), type='character', default='auto', help='hours|radians|normalized|auto'),
  make_option(c('--out-prefix'), type='character', default='zeitzeiger_output', help='Output prefix for predictions and optional model'),
  make_option(c('--save-model'), action='store_true', default=FALSE, help='Save fitted model to RDS'),
  make_option(c('--verbose'), action='store_true', default=FALSE, help='Verbose')
)

opt <- parse_args(OptionParser(option_list = opt_list))
vcat <- function(...) if (opt$verbose) cat(..., '\n')

read_expression <- function(path) {
  stopifnot(file.exists(path))
  df <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  first_col <- colnames(df)[1]
  if (grepl('gene|symbol|probe', tolower(first_col))) {
    genes <- df[[1]]
    mat <- as.matrix(df[,-1, drop=FALSE])
    rownames(mat) <- genes
  } else {
    # try to coerce: if numeric values present in columns, treat rows as genes
    mat <- as.matrix(df)
  }
  storage.mode(mat) <- 'numeric'
  return(mat)
}

read_metadata <- function(path) {
  stopifnot(file.exists(path))
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

normalize_time <- function(vec, format='auto') {
  if (is.character(vec)) vec <- as.numeric(vec)
  if (format == 'auto') {
    mx <- max(vec, na.rm = TRUE)
    if (mx <= 1) format <- 'normalized' else if (mx <= (2*pi + 0.1)) format <- 'radians' else format <- 'hours'
  }
  if (format == 'normalized') return(vec)
  if (format == 'radians') return((vec %% (2*pi)) / (2*pi))
  if (format == 'hours') return((vec %% 24) / 24)
  stop('Unknown time format')
}

align_and_prepare <- function(expr_mat, meta, sample_col, time_col, time_format) {
  # expr_mat: genes x samples (rownames genes, colnames samples)
  sc <- if (sample_col %in% colnames(meta)) sample_col else if ('Sample' %in% colnames(meta)) 'Sample' else stop('No sample column')
  meta$.__sample_id <- as.character(meta[[sc]])
  # drop expression samples missing in metadata
  expr_samples <- colnames(expr_mat)
  keep <- expr_samples[expr_samples %in% meta$.__sample_id]
  if (length(keep) < length(expr_samples)) vcat('Dropping', length(expr_samples) - length(keep), 'expression samples not present in metadata')
  expr_mat2 <- expr_mat[, keep, drop=FALSE]
  # subset metadata to those samples and order to match columns
  meta2 <- meta[meta$.__sample_id %in% colnames(expr_mat2), , drop=FALSE]
  meta2 <- meta2[match(colnames(expr_mat2), meta2$.__sample_id), , drop=FALSE]
  if (!all(colnames(expr_mat2) == meta2$.__sample_id)) stop('Failed to align expression columns and metadata sample order')
  time_norm <- normalize_time(suppressWarnings(as.numeric(meta2[[time_col]])), time_format)
  list(mat = t(expr_mat2), time = time_norm, meta = meta2)
}

if (is.null(opt$`expr-train`) || is.null(opt$`meta-train`) || is.null(opt$`expr-test`) || is.null(opt$`meta-test`)) {
  stop('Provide --expr-train, --meta-train, --expr-test and --meta-test')
}

vcat('Reading training data...')
expr_train <- read_expression(opt$`expr-train`)
meta_train <- read_metadata(opt$`meta-train`)
vcat('Reading test data...')
expr_test <- read_expression(opt$`expr-test`)
meta_test <- read_metadata(opt$`meta-test`)

# intersect genes between train and test for safe fit
common_genes <- intersect(rownames(expr_train), rownames(expr_test))
if (length(common_genes) == 0) stop('No common genes between train and test')
expr_train <- expr_train[common_genes, , drop=FALSE]
expr_test <- expr_test[common_genes, , drop=FALSE]

prep_train <- align_and_prepare(expr_train, meta_train, opt$`sample-col`, opt$`time-col`, opt$`time-format`)
prep_test <- align_and_prepare(expr_test, meta_test, opt$`sample-col`, opt$`time-col`, opt$`time-format`)

mat_train <- prep_train$mat  # samples x genes
time_train <- prep_train$time
mat_test <- prep_test$mat
time_test <- prep_test$time

vcat('Train samples:', nrow(mat_train), 'Test samples:', nrow(mat_test), 'Genes:', ncol(mat_train))

# Fit ZeitZeiger with default, small settings (user can change the script later)
res <- zeitzeiger(xTrain = mat_train, timeTrain = time_train, xTest = mat_test,
                  nKnots = 3, nTime = 10, useSpc = TRUE, sumabsv = 2, nSpc = 2)

timePred_norm <- res$predResult$timePred[,1]
timePred_hours <- timePred_norm * 24
true_hours <- time_test * 24

errs_hours <- (timePred_hours - true_hours) %% 24
errs_hours <- ifelse(errs_hours > 12, errs_hours - 24, errs_hours)

pred_df <- data.frame(sample = rownames(mat_test), pred_time_norm = timePred_norm,
                      pred_time_hours = timePred_hours, true_time_hours = true_hours,
                      error_hours = errs_hours, abs_error_hours = abs(errs_hours), stringsAsFactors = FALSE)

out_csv <- paste0(opt$`out-prefix`, '.predictions.csv')
write.csv(pred_df, out_csv, row.names = FALSE)
vcat('Wrote predictions to', out_csv)

if (opt$`save-model`) {
  saveRDS(list(res = res), paste0(opt$`out-prefix`, '.model.rds'))
  vcat('Saved model RDS')
}

cat('Mean absolute error (hours):', mean(pred_df$abs_error_hours, na.rm = TRUE), '\n')
invisible(NULL)
