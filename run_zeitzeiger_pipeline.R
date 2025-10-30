#!/usr/bin/env Rscript
# Run a full ZeitZeiger pipeline on arbitrary datasets.
# Supports: single dataset split into train/test, or separate train/test files.
# Outputs predictions CSV and optional model RDS and summary metrics.

suppressPackageStartupMessages({
  if (!requireNamespace('optparse', quietly = TRUE)) stop('Please install the R package optparse to run this script')
  if (!requireNamespace('zeitzeiger', quietly = TRUE)) stop('Please install the R package zeitzeiger before running this script')
  library(optparse)
  library(zeitzeiger)
})

option_list = list(
  make_option(c('-e','--expr'), type='character', default=NULL,
              help='Expression CSV file. Rows = genes (first column gene symbol), cols = samples OR vice versa. Required.', metavar='file'),
  make_option(c('--meta'), type='character', default=NULL,
              help='Metadata CSV file. Must contain sample names and time column. Required.', metavar='file'),
  make_option(c('--expr-test'), type='character', default=NULL,
              help='Optional: expression CSV for test set (if not provided, script splits single dataset).', metavar='file'),
  make_option(c('--meta-test'), type='character', default=NULL,
              help='Optional: metadata CSV for test set (if not provided, script splits single dataset).', metavar='file'),
  make_option(c('-s','--sample-col'), type='character', default='sample',
              help='Column name in metadata with sample names [default "%default"]'),
  make_option(c('-t','--time-col'), type='character', default='time',
              help='Column name in metadata with time values [default "%default"]'),
  make_option(c('--time-format'), type='character', default='auto',
              help='How time is encoded: "hours" (0-24), "radians" (0-2*pi), "normalized" (0-1), or "auto" [default].',
              metavar='format'),
  make_option(c('--split-fraction'), type='double', default=0.7,
              help='When single dataset provided, fraction used for training [default %default]'),
  make_option(c('--seed'), type='integer', default=123,
              help='Random seed for splitting [default %default]'),
  make_option(c('--nknots'), type='integer', default=3,
              help='nKnots for zeitzeigerFit [default %default]'),
  make_option(c('--ntime'), type='integer', default=10,
              help='nTime for zeitzeigerSpc [default %default]'),
  make_option(c('--nspc'), type='integer', default=2,
              help='Number of SPCs to use for prediction [default %default]'),
  make_option(c('--sumabsv'), type='double', default=2,
              help='sumabsv (L1 constraint) passed to SPC [default %default]'),
  make_option(c('--top-genes'), type='integer', default=0,
              help='Optional: select top N most variable genes before training (0 = use all) [default %default]'),
  make_option(c('--out-prefix'), type='character', default='zeitzeiger_output',
              help='Prefix for output files (CSV, RDS) [default %default]'),
  make_option(c('--save-model'), action='store_true', default=FALSE,
              help='Save fitted objects (fitResult, spcResult, predResult) as RDS [default off]'),
  make_option(c('--verbose'), action='store_true', default=FALSE,
              help='Verbose messages')
)

opt_parser = OptionParser(option_list = option_list)
opt = parse_args(opt_parser)

if (is.null(opt$expr) || is.null(opt$meta)) {
  print_help(opt_parser)
  stop('At minimum --expr and --meta must be provided')
}

vcat <- function(...) if (opt$verbose) cat(..., '\n')

read_expression <- function(path) {
  stopifnot(file.exists(path))
  df <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  # assume first column is gene symbol if name not like sample
  # if first column header contains 'Gene' or 'gene' treat as gene symbol
  first_col <- colnames(df)[1]
  if (grepl('gene', tolower(first_col)) || grepl('symbol', tolower(first_col))) {
    genes <- df[[1]]
    mat <- as.matrix(df[,-1, drop=FALSE])
    rownames(mat) <- genes
  } else {
    # ambiguous: decide by checking if other columns look numeric or sample-like
    # If majority of columns (except first) are numeric, treat first as gene
    numeric_counts <- sapply(df[,-1, drop=FALSE], function(x) sum(!is.na(as.numeric(as.character(x)))))
    if (mean(numeric_counts) > 0) {
      genes <- df[[1]]
      mat <- as.matrix(df[,-1, drop=FALSE])
      rownames(mat) <- genes
    } else {
      # otherwise assume rows are genes with rownames provided
      mat <- as.matrix(df)
    }
  }
  storage.mode(mat) <- 'numeric'
  return(mat)
}

read_metadata <- function(path) {
  stopifnot(file.exists(path))
  m <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  return(m)
}

align_expr_meta <- function(mat, meta, sample_col) {
  if (sample_col %in% colnames(meta)) {
    samples <- meta[[sample_col]]
  } else {
    stop(sprintf('metadata does not contain sample column "%s". Found: %s', sample_col, paste(colnames(meta), collapse=',')))
  }
  # matrix may be genes x samples (rownames genes) or samples x genes
  if (all(samples %in% colnames(mat))) {
    # genes x samples -> transpose
    mat2 <- t(mat)
  } else if (all(samples %in% rownames(mat))) {
    mat2 <- mat
  } else if (all(colnames(mat) %in% samples)) {
    mat2 <- t(mat)
  } else {
    stop('Could not align samples between expression and metadata (names mismatch).')
  }
  # subset/order according to metadata
  if (!all(samples %in% rownames(mat2))) stop('Not all metadata samples found in expression after alignment')
  mat2 <- mat2[match(samples, rownames(mat2)), , drop = FALSE]
  rownames(mat2) <- samples
  return(mat2)
}

normalize_time <- function(vec, format = 'auto') {
  # vec could be numeric or character
  if (is.character(vec)) vec <- as.numeric(vec)
  if (format == 'auto') {
    if (max(vec, na.rm = TRUE) <= 1) format <- 'normalized'
    else if (max(vec, na.rm = TRUE) > 2*pi - 0.1) format <- 'radians'
    else format <- 'hours'
  }
  if (format == 'normalized') return(vec)
  if (format == 'radians') return((vec %% (2*pi)) / (2*pi))
  if (format == 'hours') {
    # map to 0-24 then divide by 24
    v <- vec %% 24
    return(v/24)
  }
  stop('Unknown time format')
}

circle_diff_hours <- function(pred_hours, true_hours) {
  # both in 0-24
  d <- (pred_hours - true_hours) %% 24
  d <- ifelse(d > 12, d - 24, d)
  return(d)
}

## Load data
vcat('Reading expression:', opt$expr)
expr_mat <- read_expression(opt$expr)
meta <- read_metadata(opt$meta)

if (!is.null(opt$`expr-test`) && !is.null(opt$`meta-test`)) {
  vcat('Using separate train/test files')
  expr_mat_train <- read_expression(opt$expr)
  meta_train <- meta
  expr_mat_test <- read_expression(opt$`expr-test`)
  meta_test <- read_metadata(opt$`meta-test`)
  # align separately
  mat_train <- align_expr_meta(expr_mat_train, meta_train, opt$`sample-col`)
  mat_test <- align_expr_meta(expr_mat_test, meta_test, opt$`sample-col`)
  time_train_raw <- meta_train[[opt$`time-col"]]
  time_test_raw <- meta_test[[opt$`time-col"]]
  time_format <- opt$`time-format`
  timeTrain <- normalize_time(time_train_raw, time_format)
  timeTest <- normalize_time(time_test_raw, time_format)
} else {
  vcat('Single dataset provided; will split into train/test')
  # align full matrix to metadata
  mat_all <- align_expr_meta(expr_mat, meta, opt$`sample-col`)
  set.seed(opt$seed)
  n <- nrow(mat_all)
  ntrain <- max(1, floor(opt$`split-fraction` * n))
  train_idx <- sample(seq_len(n), size = ntrain)
  mat_train <- mat_all[train_idx, , drop = FALSE]
  mat_test <- mat_all[-train_idx, , drop = FALSE]
  meta_train <- meta[train_idx, , drop = FALSE]
  meta_test <- meta[-train_idx, , drop = FALSE]
  time_format <- opt$`time-format`
  timeTrain <- normalize_time(meta_train[[opt$`time-col`]], time_format)
  timeTest <- normalize_time(meta_test[[opt$`time-col`]], time_format)
}

vcat('Train samples:', nrow(mat_train), 'Test samples:', nrow(mat_test))

## optional feature selection: top variable genes
if (opt$`top-genes` > 0) {
  v <- apply(t(mat_train), 2, var, na.rm = TRUE)
  topg <- names(sort(v, decreasing = TRUE))[seq_len(min(opt$`top-genes`, length(v)))]
  vcat('Selecting top', length(topg), 'genes by variance')
  mat_train <- mat_train[, topg, drop = FALSE]
  mat_test <- mat_test[, topg, drop = FALSE]
}

## Fit and predict
vcat('Fitting zeitzeiger...')
vcat('Using wrapper zeitzeiger()')
res <- zeitzeiger(xTrain = mat_train, timeTrain = timeTrain, xTest = mat_test,
                  nKnots = opt$nknots, nTime = opt$ntime, useSpc = TRUE,
                  sumabsv = opt$sumabsv, nSpc = opt$nspc)

timePred_norm <- res$predResult$timePred[,1]
timePred_hours <- timePred_norm * 24

# true times in hours
trueTest_norm <- normalize_time(meta_test[[opt$`time-col`]], time_format)
trueTest_hours <- trueTest_norm * 24

errs <- circle_diff_hours(timePred_hours, trueTest_hours)

pred_df <- data.frame(sample = rownames(mat_test),
                      pred_time_norm = timePred_norm,
                      pred_time_hours = timePred_hours,
                      true_time_norm = trueTest_norm,
                      true_time_hours = trueTest_hours,
                      error_hours = errs,
                      abs_error_hours = abs(errs), stringsAsFactors = FALSE)

out_csv <- paste0(opt$`out-prefix`, '.predictions.csv')
write.csv(pred_df, out_csv, row.names = FALSE)
vcat('Wrote predictions to', out_csv)

if (opt$`save-model`) {
  out_rds <- paste0(opt$`out-prefix`, '.model.rds')
  saveRDS(list(res = res), out_rds)
  vcat('Saved model to', out_rds)
}

# summary metrics
mean_abs <- mean(pred_df$abs_error_hours, na.rm = TRUE)
median_abs <- median(pred_df$abs_error_hours, na.rm = TRUE)
vcat(sprintf('Mean absolute circular error (hours): %.3f', mean_abs))
vcat(sprintf('Median absolute circular error (hours): %.3f', median_abs))

cat('Done. Outputs:\n')
cat(' - Predictions:', out_csv, '\n')
if (opt$`save-model`) cat(' - Model RDS:', out_rds, '\n')

invisible(NULL)
