#!/usr/bin/env Rscript
# Run Seurat CCA integration on MouseEmbryo and evaluate ARI/NMI
suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
  library(hdf5r)
  library(mclust)
  library(aricode)
})

options(future.globals.maxSize = 4 * 1024^3)  # 4 GB

# --- Helper: read h5ad into Seurat object ---
read_h5ad_to_seurat <- function(path, slice_name="slice") {
  h5 <- H5File$new(path, mode="r")

  # Read expression matrix (X) - could be dense or sparse
  if (h5$exists("X")) {
    x_obj <- h5[["X"]]
    if (inherits(x_obj, "H5Group")) {
      # Sparse CSR/CSC
      data <- x_obj[["data"]]$read()
      indices <- x_obj[["indices"]]$read()
      indptr <- x_obj[["indptr"]]$read()
      shape <- x_obj$attr_open("shape")$read()
      # CSR format: shape = (n_obs, n_var)
      X_sparse <- sparseMatrix(
        j = indices + 1L,
        p = indptr,
        x = data,
        dims = shape
      )
    } else {
      X_dense <- x_obj$read()
      X_sparse <- as(X_dense, "dgCMatrix")
    }
  } else {
    stop("No X found in h5ad")
  }

  # Read var names (genes)
  if (h5$exists("var/_index")) {
    genes <- h5[["var/_index"]]$read()
  } else if (h5$exists("var/index")) {
    genes <- h5[["var/index"]]$read()
  } else {
    genes <- paste0("gene_", seq_len(ncol(X_sparse)))
  }

  # Read obs names (cells)
  if (h5$exists("obs/_index")) {
    cells <- h5[["obs/_index"]]$read()
  } else if (h5$exists("obs/index")) {
    cells <- h5[["obs/index"]]$read()
  } else {
    cells <- paste0("cell_", seq_len(nrow(X_sparse)))
  }

  # Read labels
  labels <- NULL
  label_key <- "cellbin_SpatialDomain"
  obs_group <- h5[["obs"]]
  if (obs_group$exists(label_key)) {
    lab_obj <- obs_group[[label_key]]
    if (inherits(lab_obj, "H5Group")) {
      # Categorical encoding
      codes <- lab_obj[["codes"]]$read()
      categories <- lab_obj[["categories"]]$read()
      labels <- categories[codes + 1L]
    } else {
      labels <- lab_obj$read()
    }
  }

  # Read spatial coords
  coords <- NULL
  if (h5$exists("obsm/spatial")) {
    coords <- h5[["obsm/spatial"]]$read()
    # hdf5r reads in column-major, transpose if needed
    if (nrow(coords) == 2 && ncol(coords) != 2) {
      coords <- t(coords)
    }
  }

  h5$close_all()

  # Transpose: Seurat wants genes x cells
  rownames(X_sparse) <- cells
  colnames(X_sparse) <- genes
  X_t <- t(X_sparse)  # genes x cells

  obj <- CreateSeuratObject(counts = X_t, project = slice_name)

  if (!is.null(labels)) {
    obj$label <- labels
  }
  if (!is.null(coords)) {
    rownames(coords) <- colnames(obj)
    obj[["spatial_coords"]] <- coords
  }

  return(obj)
}

# --- Load data ---
cat("Loading MouseEmbryo slices...\n")
s1 <- read_h5ad_to_seurat("./Data/MouseEmbyro/slices1.h5ad", "slice1")
cat(sprintf("  Slice 1: %d cells, %d genes\n", ncol(s1), nrow(s1)))
s2 <- read_h5ad_to_seurat("./Data/MouseEmbyro/slices2.h5ad", "slice2")
cat(sprintf("  Slice 2: %d cells, %d genes\n", ncol(s2), nrow(s2)))

# --- Get labels and coords before integration ---
labels1 <- s1$label
labels2 <- s2$label
coords1 <- s1[["spatial_coords"]]
coords2 <- s2[["spatial_coords"]]

# Count clusters
all_labels <- unique(c(labels1, labels2))
all_labels <- all_labels[!is.na(all_labels) & all_labels != "" & all_labels != "nan"]
n_clusters <- length(all_labels)
cat(sprintf("  n_clusters = %d\n", n_clusters))

# --- Seurat CCA Integration ---
cat("\nRunning Seurat CCA integration...\n")
n_hvg <- 3000
n_pcs <- 30

s1 <- NormalizeData(s1, verbose = FALSE)
s1 <- FindVariableFeatures(s1, nfeatures = n_hvg, verbose = FALSE)
s2 <- NormalizeData(s2, verbose = FALSE)
s2 <- FindVariableFeatures(s2, nfeatures = n_hvg, verbose = FALSE)

cat("  Finding integration anchors...\n")
obj.list <- list(s1, s2)
anchors <- FindIntegrationAnchors(object.list = obj.list, dims = 1:n_pcs, verbose = FALSE)

cat("  Integrating data...\n")
integrated <- IntegrateData(anchorset = anchors, dims = 1:n_pcs, verbose = FALSE)

DefaultAssay(integrated) <- "integrated"
integrated <- ScaleData(integrated, verbose = FALSE)
integrated <- RunPCA(integrated, npcs = n_pcs, verbose = FALSE)

cat("  Integration done.\n")

# Extract PCA embeddings
pca_emb <- Embeddings(integrated, "pca")

# Split back
cells1 <- colnames(s1)
cells2 <- colnames(s2)
emb1 <- pca_emb[cells1, ]
emb2 <- pca_emb[cells2, ]

# --- Evaluate: mclust ---
cat("\n=== Evaluation ===\n")

evaluate_mclust <- function(emb, labels, n_k, slice_name) {
  valid <- !is.na(labels) & labels != "" & labels != "nan"
  emb_v <- emb[valid, ]
  lab_v <- labels[valid]

  set.seed(666)
  fit <- Mclust(emb_v, G = n_k, modelNames = "EEE")
  pred <- fit$classification

  ari_val <- ARI(lab_v, pred)
  nmi_val <- NMI(lab_v, pred)
  cat(sprintf("  Seurat | %s [mclust]: ARI=%.4f  NMI=%.4f\n", slice_name, ari_val, nmi_val))
  return(c(ari_val, nmi_val))
}

cat("Running mclust clustering...\n")
res1 <- evaluate_mclust(emb1, labels1, n_clusters, "embryo_s1")
res2 <- evaluate_mclust(emb2, labels2, n_clusters, "embryo_s2")

cat(sprintf("\n  Seurat (mclust mean): ARI=%.4f  NMI=%.4f\n",
            mean(res1[1], res2[1]), mean(res1[2], res2[2])))

cat("\nDone!\n")
