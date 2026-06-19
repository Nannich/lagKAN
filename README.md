# lagKAN

lagKAN is a tool for inferring signed Gene Regulatory Networks (GRNs) by training a separate Kolmogorov-Arnold Network (KAN) for each gene. It predicts a target gene's expression, shifted by a relative time lag, using the expression of all other genes as input. The complete network is then constructed by extracting the feature importance scores and gradients from each trained KAN to determine the strength and regulatory effect (activation or repression) of every edge.

---

## Data Requirements

To run the inference loop, you need to prepare three input matrices:
* **Log-transformed expression counts**: Normalized (e.g. log1p) counts matrix formatted as (Cells x Genes).
* **Pseudotime values**: Matrix formatted as (Cells x Lineages).
* **Lineage assignment**: A boolean mask formatted as (Cells x Lineages), where True signifies a cell belongs to that specific lineage branch.

If only the gene expression counts matrix is available, you must first run a trajectory inference algorithm to obtain the pseudotime values for each cell.

---

## Usage

Use the `infer_grn` function to execute the inference algorithm.

```python
import pandas as pd
import lagkan

# 1. Format inputs from your dataframes
raw_counts = expression_df.values.T
log_counts = np.log1p(raw_counts)       
pseudotime = pt_df.fillna(0.0).values           
lineage_assignment = pt_df.notna().values       
gene_names = expression_df.index.values

# 2. Run inference with default parameters
ranked_edges_df = lagkan.infer_grn(
    log_counts=log_counts,
    pseudotime=pseudotime,
    lineage_assignment=lineage_assignment,
    gene_names=gene_names,
    dt=0.08,
    epochs=400,     
    lr=0.01,
    lamb_l1=0.02,
    edge_threshold=0.0
)
```

---

## Parameters

* `log_counts` (n_cells, n_genes): Log-transformed or normalized gene expression matrix (e.g., log1p format).
* `pseudotime` (n_cells, n_lineages): Pseudotime values of each cell along the lineage.
* `lineage_assignment` (n_cells, n_lineages): Boolean mask where True signifies a cell belongs to that specific lineage branch.
* `gene_names`: String identifiers for the genes. If None, features are named 'Gene_0', 'Gene_1', etc.
* `dt`: Relative time lag to apply on the target expression. Must be bound within the range [0, 1).
* `epochs`: Number of training epochs.
* `lr`: Learning rate for the optimizer.
* `lamb_l1`: L1 regularization penalty coefficient used to enforce sparsity in the network.
* `edge_threshold`: Minimum absolute weight cutoff required to retain an inferred regulatory edge in the final output.

---

## Output

The algorithm returns a pandas DataFrame detailing the inferred regulatory edges.
- Edges are sorted in descending order based on their absolute edge weight.
- The DataFrame includes the columns: Gene1 (the predictor gene), Gene2 (the target gene), EdgeWeight.
- An activating edge is represented by a positive weight, a repressing edge is represented by a negative weight.

---

## Examples

A step-by-step tutorial is provided in `examples/run_lagkan.ipynb`. This notebook demonstrates how to load a dataset from the `data/` directory, prepare the inputs, run the inference loop, and evaluate the resulting network.