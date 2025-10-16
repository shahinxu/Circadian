library(Oscope)
library(BiocParallel)

# ------------------------------
# 性能与并行配置（可调节）
# ------------------------------
# 如需更快，请将 QUICK_MODE <- TRUE（默认已开启）；若追求更全面结果可改为 FALSE
QUICK_MODE <- TRUE

# 关键开关与上限（在 QUICK_MODE=TRUE 时生效）
MAX_GENES_OSCOPE <- 800      # 进入 OscopeSine 的最大基因数上限（DataInput）
MAX_GENES_LOG1P  <- 1200     # log1p 回退时的最大候选基因数
MAX_CLUSTERS     <- 20       # ENI/绘图处理的最大簇数（按簇大小排序取前 N）
KM_MAX_K         <- if (QUICK_MODE) 6 else 10  # KMeans 最大 K
PLOT_GENES_PER_CLUSTER <- if (QUICK_MODE) 4 else 6

# OscopeSine 的并行开关与线程数（为稳定起见，Windows 下默认关闭并行）
SINE_PARALLEL <- FALSE
SINE_WORKERS  <- max(1L, parallel::detectCores(logical = TRUE) - 1L)

# ENI 阶段为稳定起见仍默认串行（之前并行容易报错）；若要尝试可改为 TRUE
ENI_PARALLEL <- FALSE

# 打印每步骤耗时
TIME_LOG <- TRUE

tic <- function() assign(".tic_time", Sys.time(), envir = .GlobalEnv)
toc <- function(msg = "") {
  if (!TIME_LOG) return(invisible(NULL))
  if (exists(".tic_time", envir = .GlobalEnv)) {
    dt <- difftime(Sys.time(), get(".tic_time", envir = .GlobalEnv), units = "mins")
    cat(sprintf("[耗时] %s: %.2f 分钟\n", msg, as.numeric(dt)))
  }
  tic()
}

# 注册并行后端（仅对部分内部并行有效；OscopeSine 的 parallel=TRUE 也会提速）
if (isTRUE(SINE_PARALLEL)) {
  # Windows 没有 MulticoreParam，使用 SnowParam
  bp <- BiocParallel::SnowParam(workers = SINE_WORKERS, type = "SOCK")
  BiocParallel::register(bp, default = TRUE)
}

# 将 CalcMV 的核心逻辑“写开”到脚本里，便于调试和打印中间结果（中文）
CalcMV_open <- function(Data,
                        Sizes = NULL,
                        NormData = FALSE,
                        MeanCutLow = 100,
                        MeanCutHigh = NULL,
                        ApproxVal = 1e-6,
                        Plot = FALSE,
                        verbose = TRUE) {
  if (is.null(rownames(Data))) stop("Data 必须有行名(基因名)")
  EmpData <- Data
  EmpSizes <- Sizes
  # 生成或修复 Sizes（避免 NA/非正值）
  if (is.null(EmpSizes) || length(EmpSizes) != ncol(EmpData)) {
    EmpSizes <- MedianNorm(EmpData)
  }
  bad_sz <- which(!is.finite(EmpSizes) | EmpSizes <= 0)
  if (length(bad_sz) > 0) {
    if (verbose) cat(sprintf("[CalcMV] 检测到 %d 个无效 size，尝试重新估计\n", length(bad_sz)))
    EmpSizes2 <- try(MedianNorm(EmpData), silent = TRUE)
    if (!inherits(EmpSizes2, "try-error") && length(EmpSizes2) == ncol(EmpData)) {
      EmpSizes <- EmpSizes2
    }
    # 若仍有无效值，用各列总和的中位数替代，最后兜底设为1
    bad_sz <- which(!is.finite(EmpSizes) | EmpSizes <= 0)
    if (length(bad_sz) > 0) {
      col_sum <- colSums(EmpData, na.rm = TRUE)
      med_sum <- stats::median(col_sum[col_sum > 0])
      if (!is.finite(med_sum) || med_sum <= 0) med_sum <- 1
      EmpSizes[bad_sz] <- med_sum
    }
  }
  if (verbose) {
    cat(sprintf("[CalcMV] 细胞数: %d, 基因数: %d\n", ncol(EmpData), nrow(EmpData)))
    cat(sprintf("[CalcMV] Sizes 范围: [%.4g, %.4g]\n", suppressWarnings(min(EmpSizes)), suppressWarnings(max(EmpSizes))))
    cat(sprintf("[CalcMV] MeanCutLow=%s, MeanCutHigh=%s\n",
                as.character(MeanCutLow), as.character(MeanCutHigh)))
  }

  if (!isTRUE(NormData)) EmpData.norm <- t(t(EmpData) / EmpSizes) else EmpData.norm <- EmpData
  # 归一化后清洗，避免 NA/Inf 影响均值计算
  EmpData.norm[!is.finite(EmpData.norm)] <- 0
  MeansC1 <- rowMeans(EmpData.norm, na.rm = TRUE)
  MedC1   <- apply(EmpData.norm, 1, median)

  Sig_tmp  <- (EmpData - MeansC1 %*% t(EmpSizes))^2
  Sig_tmp2 <- t(t(Sig_tmp) / EmpSizes)
  VarC1    <- rowMeans(Sig_tmp2)

  QC1 <- MeansC1 / VarC1
  QNB <- QC1
  QNB[which(QNB >= 1)] <- ApproxVal
  PhiNB <- (1 - QNB) / (MeansC1 * QNB)

  SampleMean <- MeansC1
  SampleVar  <- VarC1
  Which <- seq_along(SampleMean)
  if (!is.null(MeanCutLow)) {
    Which <- which(SampleMean > MeanCutLow)
    if (!is.null(MeanCutHigh)) {
      Which <- intersect(Which, which(SampleMean < MeanCutHigh))
    }
  }
  if (verbose) cat(sprintf("[CalcMV] 参与拟合的基因数(均值阈值后): %d\n", length(Which)))
  if (length(Which) < 3) {
    stop(sprintf("CalcMV: 用于拟合的基因数过少(<3)。MeanCutLow=%s, MeanCutHigh=%s, 合格数=%d",
                 as.character(MeanCutLow), as.character(MeanCutHigh), length(Which)))
  }

  Meanfit <- log10(SampleMean[Which])
  Varfit  <- log10(SampleVar[Which])
  lm1 <- stats::lm(Varfit ~ Meanfit)
  Coef <- stats::coef(lm1)
  if (verbose) {
    cat(sprintf("[CalcMV] 拟合系数: intercept=%.4f, slope=%.4f\n",
                as.numeric(Coef[1]), as.numeric(Coef[2])))
  }

  Gt10    <- names(SampleMean)[Which]
  MeanUse <- SampleMean[Gt10]
  VarUse  <- SampleVar[Gt10]
  Fit     <- 10^(Coef[1] + Coef[2] * log10(MeanUse))
  Diff    <- VarUse - Fit
  SamplePickGenes <- names(MeanUse)[which(Diff > 0)]
  if (verbose) cat(sprintf("[CalcMV] 残差>0 的基因数: %d\n", length(SamplePickGenes)))

  if (isTRUE(Plot)) {
    plot(SampleMean, SampleVar, col = "gray", pch = 21, xlab = "Mean",
         ylab = "Variance", log = "xy")
    lines(MeanUse[order(MeanUse)], Fit[order(MeanUse)])
    points(MeanUse[SamplePickGenes], VarUse[SamplePickGenes], pch = 21, col = "green")
    abline(v = c(MeanCutLow, MeanCutHigh))
  }

  return(list(
    Mean = MeansC1,
    Var = VarC1,
    Median = MedC1,
    GeneToUse = SamplePickGenes,
    Q = QC1,
    Q_mdf = QNB,
    Phi_mdf = PhiNB,
    Which = Which,
    coef = Coef
  ))
}

# 从CSV读取表达矩阵（行=基因，列=细胞），保留原始列名
if (TIME_LOG) tic()
my_data <- read.csv("../data/GSE233242/expression.csv", row.names = 1, check.names = FALSE)
my_data <- as.matrix(my_data)

# 基础校验与清洗，排查“读数据错了”的可能
if (is.null(rownames(my_data))) {
  stop("数据第一列应为基因名（row.names=1）。当前未检测到行名。")
}
# 强制为数值类型
suppressWarnings(storage.mode(my_data) <- "double")
non_finite <- sum(!is.finite(my_data))
if (non_finite > 0) {
  warning("检测到非有限值(NA/NaN/Inf)，已置为0: ", non_finite)
  my_data[!is.finite(my_data)] <- 0
}
negatives <- sum(my_data < 0, na.rm = TRUE)
if (negatives > 0) {
  warning("检测到负值，已截断为0: ", negatives)
  my_data[my_data < 0] <- 0
}
# 合并重复基因名（如有）
dup_genes <- sum(duplicated(rownames(my_data)))
if (dup_genes > 0) {
  message("检测到重复基因名，按行求和合并: ", dup_genes)
  my_data <- rowsum(my_data, group = rownames(my_data))
}
# 概览：维度与全零行列
cat(sprintf("基因数: %d，细胞数: %d\n", nrow(my_data), ncol(my_data)))
cat(sprintf("全零基因: %d，全零细胞: %d\n", sum(rowSums(my_data) == 0), sum(colSums(my_data) == 0)))

Sizes <- MedianNorm(my_data)
# 修复无效的 size 因子，避免 DataNorm 中出现 NA
if (length(Sizes) != ncol(my_data) || any(!is.finite(Sizes)) || any(Sizes <= 0)) {
  Sizes2 <- try(MedianNorm(my_data), silent = TRUE)
  if (!inherits(Sizes2, "try-error") && length(Sizes2) == ncol(my_data)) {
    Sizes <- Sizes2
  }
  bad <- which(!is.finite(Sizes) | Sizes <= 0)
  if (length(bad) > 0) {
    col_sum <- colSums(my_data, na.rm = TRUE)
    med_sum <- stats::median(col_sum[col_sum > 0])
    if (!is.finite(med_sum) || med_sum <= 0) med_sum <- 1
    Sizes[bad] <- med_sum
  }
}
cat(sprintf("Sizes 范围(修复后): [%.4g, %.4g]\n", suppressWarnings(min(Sizes)), suppressWarnings(max(Sizes))))
toc("读取与预处理原始数据")

if (TIME_LOG) tic()
DataNorm <- GetNormalizedMat(my_data, Sizes)
# 归一化矩阵清洗，防止 NA/NaN/Inf 传入后续
suppressWarnings(storage.mode(DataNorm) <- "double")
DataNorm[!is.finite(DataNorm)] <- 0
# 额外构造 log1p 版本，作为后续回退可用
DataLogNorm <- log1p(DataNorm)
suppresWarnings <- function(x) try(suppressWarnings(x), silent = TRUE)
suppresWarnings(storage.mode(DataLogNorm) <- "double")
DataLogNorm[!is.finite(DataLogNorm)] <- 0
toc("生成归一化/对数矩阵")

if (TIME_LOG) tic()
MV <- CalcMV_open(Data = my_data, Sizes = Sizes, MeanCutLow = 0, MeanCutHigh = 100000, Plot = FALSE, verbose = TRUE)
DataSubset <- DataNorm[MV$GeneToUse, , drop = FALSE]
# 清理子集，避免 NormForSine 的分位数计算报 NA 错
DataSubset[!is.finite(DataSubset)] <- 0
row_sd <- apply(DataSubset, 1, stats::sd)
keep <- is.finite(row_sd) & row_sd > 0
DataSubset <- DataSubset[keep, , drop = FALSE]
if (nrow(DataSubset) < 3L) {
  # 回退：从 DataNorm 选方差最高的基因补足
  vars_norm <- apply(DataNorm, 1, stats::var, na.rm = TRUE)
  vars_norm[!is.finite(vars_norm)] <- 0
  ord <- order(vars_norm, decreasing = TRUE)
  top_n <- min(length(ord), if (QUICK_MODE) MAX_GENES_OSCOPE else 1000L)
  DataSubset <- DataNorm[ord[seq_len(top_n)], , drop = FALSE]
  DataSubset[!is.finite(DataSubset)] <- 0
  row_sd <- apply(DataSubset, 1, stats::sd)
  keep <- is.finite(row_sd) & row_sd > 0
  DataSubset <- DataSubset[keep, , drop = FALSE]
}

# 进一步在 QUICK_MODE 下裁剪候选基因数，避免 OscopeSine 过慢
if (QUICK_MODE && nrow(DataSubset) > MAX_GENES_OSCOPE) {
  vars_ds <- apply(DataSubset, 1, stats::var, na.rm = TRUE)
  vars_ds[!is.finite(vars_ds)] <- 0
  ord_ds <- order(vars_ds, decreasing = TRUE)
  keep_n <- min(length(ord_ds), MAX_GENES_OSCOPE)
  DataSubset <- DataSubset[ord_ds[seq_len(keep_n)], , drop = FALSE]
}
cat(sprintf("用于正弦模型的候选基因数: %d\n", nrow(DataSubset)))
toc("CalcMV + 候选基因筛选")

safe_norm_for_sine <- function(mat) {
  # 调用 NormForSine，并清理输出，确保无非有限/常量行
  din <- NormForSine(mat)
  mode(din) <- "numeric"
  din[!is.finite(din)] <- NA_real_
  keep_rows <- rowSums(!is.finite(din)) == 0
  din <- din[keep_rows, , drop = FALSE]
  rsd <- apply(din, 1, stats::sd)
  keep_var <- is.finite(rsd) & rsd > 0
  din <- din[keep_var, , drop = FALSE]
  din
}

# 尝试 NormForSine，失败则回退到 log1p，再失败用 z-score
if (TIME_LOG) tic()
DataInput <- try(safe_norm_for_sine(DataSubset), silent = TRUE)
if (inherits(DataInput, "try-error") || nrow(DataInput) < 3L) {
  message("NormForSine 失败或基因过少，改用 log1p 矩阵作为输入")
  # 先在 log1p 矩阵上选择高方差基因
  vars_log <- apply(DataLogNorm, 1, stats::var, na.rm = TRUE)
  vars_log[!is.finite(vars_log)] <- 0
  ordL <- order(vars_log, decreasing = TRUE)
  top_nL <- min(length(ordL), if (QUICK_MODE) MAX_GENES_LOG1P else 2000L)
  DataSubsetL <- DataLogNorm[ordL[seq_len(top_nL)], , drop = FALSE]
  DataSubsetL[!is.finite(DataSubsetL)] <- 0
  DataInput <- try(safe_norm_for_sine(DataSubsetL), silent = TRUE)
  if (inherits(DataInput, "try-error") || nrow(DataInput) < 3L) {
    message("log1p + NormForSine 仍失败，改用 z-score 兜底")
    zscore <- function(v) {
      s <- stats::sd(v)
      if (!is.finite(s) || s == 0) return(rep(0, length(v)))
      (v - mean(v, na.rm = TRUE)) / s
    }
    DataInput <- t(apply(DataSubsetL, 1, zscore))
    mode(DataInput) <- "numeric"
    DataInput[!is.finite(DataInput)] <- 0
    rsd3 <- apply(DataInput, 1, stats::sd)
    keep3 <- is.finite(rsd3) & rsd3 > 0
    DataInput <- DataInput[keep3, , drop = FALSE]
    if (nrow(DataInput) < 3L) stop("兜底后用于 OscopeSine 的基因仍少于3个。")
  }
}
toc("NormForSine / 回退预处理")

# QUICK_MODE 下对 DataInput 再次裁剪，避免 OscopeSine 阶段过慢
if (QUICK_MODE && nrow(DataInput) > MAX_GENES_OSCOPE) {
  vars_in <- apply(DataInput, 1, stats::var, na.rm = TRUE)
  vars_in[!is.finite(vars_in)] <- 0
  ord_in <- order(vars_in, decreasing = TRUE)
  keep_in <- ord_in[seq_len(min(length(ord_in), MAX_GENES_OSCOPE))]
  DataInput <- DataInput[keep_in, , drop = FALSE]
}
cat(sprintf("进入 OscopeSine 的基因数: %d\n", nrow(DataInput)))
if (TIME_LOG) tic()
SineRes <- OscopeSine(DataInput, parallel = SINE_PARALLEL)
toc("OscopeSine")

if (TIME_LOG) tic()
KMRes <- OscopeKM(SineRes, maxK = KM_MAX_K)
toc("OscopeKM")

ToRM <- FlagCluster(SineRes, KMRes, DataInput)
SOFT_FLAG_FRACTION <- 0.5
flag_ids <- integer(0)
if (!is.null(ToRM$FlagID)) {
  flag_ids <- unique(as.integer(ToRM$FlagID))
  flag_ids <- flag_ids[is.finite(flag_ids) & flag_ids >= 1]
  flag_ids <- intersect(flag_ids, seq_along(KMRes))
}

# 先按索引剔除，再确保每个簇的基因都存在于 DataInput 的行名中
KMResUse <- if (length(flag_ids) > 0) KMRes[-flag_ids] else KMRes

# 软回退：若剔除后为空且标记比例过高，则忽略 FlagCluster，直接使用 KMRes
if (length(KMResUse) == 0 && length(KMRes) > 0) {
  frac_flagged <- if (length(KMRes) > 0) length(flag_ids) / length(KMRes) else 0
  if (is.finite(frac_flagged) && frac_flagged >= SOFT_FLAG_FRACTION) {
    message("[回退] FlagCluster 过于严格，已启用软回退，暂不剔除任何簇进行后续步骤")
    KMResUse <- KMRes
  }
}

# 过滤空/过小簇，并与 DataInput 行名求交集
rn <- rownames(DataInput)
# 最小基因数阈值，必要时可降到2以避免全部被过滤
MIN_GENES_PER_CLUSTER <- 3L
KMResUse2 <- list()
if (length(KMResUse) > 0) {
  for (i in seq_along(KMResUse)) {
    g <- KMResUse[[i]]
    g <- as.character(g)
    g_use <- intersect(g, rn)
    if (length(g_use) >= MIN_GENES_PER_CLUSTER) {
      nm <- names(KMResUse)[i]
      if (is.null(nm) || nm == "") nm <- paste0("cluster_", i)
      KMResUse2[[nm]] <- g_use
    }
  }
}
if (QUICK_MODE && length(KMResUse2) > MAX_CLUSTERS) {
  # 仅保留前 MAX_CLUSTERS 个基因数最多的簇
  ord_cl <- order(vapply(KMResUse2, length, integer(1L)), decreasing = TRUE)
  KMResUse2 <- KMResUse2[ord_cl[seq_len(MAX_CLUSTERS)]]
}
cat(sprintf("簇总数: %d, 剔除后: %d, 与数据交集并过滤<%d基因后: %d\n", length(KMRes), length(KMResUse), MIN_GENES_PER_CLUSTER, length(KMResUse2)))
if (length(KMResUse2) == 0) {
  # 再次回退：若严格阈值下为空，则放宽到至少2个基因
  KMResUse2 <- list()
  for (i in seq_along(KMResUse)) {
    g <- as.character(KMResUse[[i]])
    g_use <- intersect(g, rn)
    if (length(g_use) >= 2) {
      nm <- names(KMResUse)[i]
      if (is.null(nm) || nm == "") nm <- paste0("cluster_", i)
      KMResUse2[[nm]] <- g_use
    }
  }
  if (length(KMResUse2) == 0) {
    stop("过滤后没有可用的基因簇。请放宽筛选阈值或增加候选基因数量。")
  } else {
    message(sprintf("[回退] 放宽阈值到>=2基因，得到 %d 个簇", length(KMResUse2)))
  }
}
if (TIME_LOG) tic()
# 为稳定起见，默认串行；如需尝试并行可将 ENI_PARALLEL <- TRUE
ENIRes <- OscopeENI(KMRes = KMResUse2, Data = DataInput, NCThre = 100, parallel = ENI_PARALLEL)
toc("OscopeENI")

# ---------------------------------------------------------------
# 导出每个细胞的顺序/伪时间（按簇 & 共识）
# ---------------------------------------------------------------
dir.create("cell_order", showWarnings = FALSE, recursive = TRUE)
sample_ids <- colnames(DataInput)

# 每个簇单独导出：样本、顺序索引、归一化伪时间、角度、24小时尺度
for (cluster_name in names(ENIRes)) {
  recovered_order <- ENIRes[[cluster_name]]
  # recovered_order 多为列索引，映射到样本名
  recovered_order <- as.integer(recovered_order)
  recovered_order <- recovered_order[recovered_order >= 1 & recovered_order <= length(sample_ids)]
  if (length(recovered_order) == 0) next
  ord_idx <- seq_along(recovered_order)
  n <- length(recovered_order)
  pseudo_norm <- if (n > 1) (ord_idx - 1) / (n - 1) else rep(0, n)
  df <- data.frame(
    cluster = cluster_name,
    sample  = sample_ids[recovered_order],
    order_index = ord_idx,
    pseudotime_norm = pseudo_norm,
    pseudotime_deg  = pseudo_norm * 360,
    pseudotime_24h  = pseudo_norm * 24
  )
  utils::write.csv(df, file = file.path("cell_order", paste0("cluster_", cluster_name, "_cell_order.csv")), row.names = FALSE)
}

# 计算共识伪时间：对所有簇中出现的同一样本，取其归一化伪时间的平均
all_maps <- list()
for (cluster_name in names(ENIRes)) {
  recovered_order <- ENIRes[[cluster_name]]
  recovered_order <- as.integer(recovered_order)
  recovered_order <- recovered_order[recovered_order >= 1 & recovered_order <= length(sample_ids)]
  if (length(recovered_order) == 0) next
  ord_idx <- seq_along(recovered_order)
  n <- length(recovered_order)
  pseudo_norm <- if (n > 1) (ord_idx - 1) / (n - 1) else rep(0, n)
  df <- data.frame(
    sample  = sample_ids[recovered_order],
    pseudotime_norm = pseudo_norm
  )
  all_maps[[length(all_maps) + 1]] <- df
}
if (length(all_maps) > 0) {
  all_df <- do.call(rbind, all_maps)
  # 按样本聚合求均值
  consensus <- aggregate(pseudotime_norm ~ sample, data = all_df, FUN = mean)
  consensus$pseudotime_deg <- consensus$pseudotime_norm * 360
  consensus$pseudotime_24h <- consensus$pseudotime_norm * 24
  utils::write.csv(consensus, file = file.path("cell_order", "consensus_cell_order.csv"), row.names = FALSE)
}

saveRDS(ENIRes, file = "oscope_ENI_results.rds")
saveRDS(KMResUse, file = "oscope_KM_results.rds")

pdf("oscillating_gene_plots.pdf")

for (cluster_name in names(ENIRes)) {
  recovered_order <- ENIRes[[cluster_name]]
  # 使用经过过滤与命名的簇集合以避免名称不一致
  genes_in_cluster <- KMResUse2[[cluster_name]]
  if (is.null(genes_in_cluster) || length(genes_in_cluster) == 0) next
  
  # 设置画板为3x2
  par(mfrow = c(3, 2), mar = c(4, 4, 2, 1))
  
  # 最多画出每个簇的前若干基因
  num_to_plot <- min(length(genes_in_cluster), PLOT_GENES_PER_CLUSTER)
  
  if (num_to_plot > 0) {
    for (i in 1:num_to_plot) {
      plot(DataNorm[genes_in_cluster[i], recovered_order],
           xlab = "恢复出的细胞顺序", ylab = "基因表达量",
           main = paste(cluster_name, "-", genes_in_cluster[i]),
           type = "l",
           col = "blue")
    }
  }
}
dev.off()

print("===================================")
print("分析完成！结果已保存到 .rds 文件和 .pdf 文件中。")
print("===================================")