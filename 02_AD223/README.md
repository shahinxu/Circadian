Cyclops2 folder is the results shared from the author of the CYCLOPS2 for the prediction of circadian time in AD409 datasets, which has 223 samples overlapping with our AD427 dataset. The index of the table is "projid" in our dataset. So, ff you want to check whether your predict results are matching to this prediction, you could use "projid" in our .obs. 

AD223_circadian_genes_full_results are the circadian genes for each major cell class using 223 overlapping samples.

AD223_circadian_genes_AD_Only_results
AD223_circadian_genes_nonAD_Only_results are the circadian genes seperating for either AD or nonAD samples.

All circadian genes are not filter out by Q valuer or amplitute.
You could test which threshould is better.

QVAL = [0.05, 1e-2, 1e-3, 1e-4, 1e-5]
AMP  = 0.08

cir_count_Df=pd.DataFrame(columns=["celltype", "qvalue_threshold", "gene_count"])
for q in QVAL:
    for ct, grp in df.groupby("celltype"):
        sig = grp[(grp["qvalue"] < q)]
        cir_count_Df = cir_count_Df.append({"celltype": ct, "qvalue_threshold": q, "gene_count": len(sig)}, ignore_index=True)



