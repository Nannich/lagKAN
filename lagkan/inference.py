import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from kan import KAN

def get_lagged_expression(log_counts, pseudotime, lineage_assignment, target_idx, dt=0.08):
    """
    Shifts each cell on each lineage based on dt.
    """
    n_lineages = lineage_assignment.shape[1]

    X_lagged = []
    Y_lagged = []

    for l in range(n_lineages):
        mask = lineage_assignment[:, l]
        if not np.any(mask): 
            continue
        
        lin_counts = log_counts[mask]
        lin_pt = pseudotime[mask, l]
        
        # Sort cells by pseudotime
        sort_idx = np.argsort(lin_pt)
        lin_counts_sorted = lin_counts[sort_idx]
        lin_pt_sorted = lin_pt[sort_idx]
        
        X_branch = []
        Y_branch = []
        
        # Calculate absolute time lag for this lineage
        pt_min = lin_pt_sorted[0]
        pt_max = lin_pt_sorted[-1]
        branch_duration = pt_max - pt_min

        if branch_duration > 0:
            branch_dt = branch_duration * dt
        else:
            branch_dt = dt
        
        # Shift each cells expression by the calculated lag
        for i in range(len(lin_pt_sorted)):
            t = lin_pt_sorted[i]
            t_future = t + branch_dt

            if t_future > pt_max:
                break
                
            x_t = lin_counts_sorted[i, :]
            # Interpolate target gene expression at the future time
            y_future = np.interp(t_future, lin_pt_sorted, lin_counts_sorted[:, target_idx])
            
            X_branch.append(x_t)
            Y_branch.append([y_future])
            
        if len(X_branch) > 0:
            X_lagged.append(np.array(X_branch))
            Y_lagged.append(np.array(Y_branch))

    X_final = np.vstack(X_lagged)
    Y_final = np.vstack(Y_lagged)
    X_final = np.delete(X_final, target_idx, axis=1)

    return X_final, Y_final

def train_n_to_1_kan(X_tensor, Y_tensor, device, epochs=400, lr=0.01, lamb_l1=0.02):
    """
    Trains a shallow 2-layer Kolmogorov-Arnold Network.
    """
    in_dim = X_tensor.shape[1]
    model = KAN(width=[in_dim, 1], grid=3, k=3, device=device, auto_save=False)
    model.update_grid_from_samples(X_tensor)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_tensor)
        mse_loss = criterion(predictions, Y_tensor)
        l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
        loss = mse_loss + (lamb_l1 * l1_loss)
        loss.backward()
        optimizer.step()

    model.eval()
    return model

def extract_kan_gradients(kan_model, X_tensor):
    """
    Determines edge signs using model gradients.
    """
    kan_model.eval()
    
    X_eval = X_tensor.clone().detach().requires_grad_(True)
    predictions = kan_model(X_eval)
    
    grads = torch.autograd.grad(outputs=predictions.sum(), inputs=X_eval)[0]
    mean_grads = grads.mean(dim=0).cpu().numpy()
    
    edge_signs = np.where(mean_grads >= 0, 1, -1)
    return edge_signs

def extract_kan_weights(kan_model):
    """
    Extracts feature attribution importance scores from the trained KAN model.
    """
    kan_model.attribute()
    edge_weights = kan_model.edge_scores[0].detach().cpu().numpy().flatten()
    return edge_weights

def infer_grn(raw_counts, pseudotime, lineage_assignment, gene_names=None, dt=0.08, epochs=400, lr=0.01, lamb_l1=0.02, edge_threshold=0.0):
    """
    Infers a Gene Regulatory Network by training a Kolmogorov-Arnold Network (KAN) for each gene.

    Parameters
    ----------
    log_counts : array-like of shape (n_cells, n_genes)
        Log-transformed or normalized gene expression matrix (e.g., log1p format).
    pseudotime : array-like of shape (n_cells, n_lineages)
        Pseudotime values of each cell along the lineage.
    lineage_assignment : array-like of shape (n_cells, n_lineages)
        Boolean mask where True signifies a cell belongs to that specific lineage branch.
    gene_names : list or array-like of str, optional
        String identifiers for the genes. If None, features are named 'Gene_0', 'Gene_1', etc.
    dt : float, default 0.08
        Relative time lag to apply on the target expression. Must be bound within the range [0, 1).
    epochs : int, default 400
        Number of training epochs.
    lr : float, default 0.01
        Learning rate.
    lamb_l1 : float, default 0.02
        L1 regularization penalty coefficient to enforce sparsity.
    edge_threshold : float, default 0.0
        Minimum absolute weight cutoff required to retain an inferred regulatory interaction.

    Returns
    -------
    pd.DataFrame
        Structured dataframe containing the inferred regulatory interactions with columns:
        - 'Gene1': Predictor gene name.
        - 'Gene2': Target gene name.
        - 'EdgeWeight': Signed edge weight (positive for activation, negative for repression).

        The dataframe is returned sorted in descending order by absolute edge weight.
    """
    if not (0 <= dt < 1):
        raise ValueError("The lag horizon parameter 'dt' must be a relative fraction between [0, 1).")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    log_counts = np.asarray(log_counts, dtype=np.float32)
    pseudotime = np.asarray(pseudotime, dtype=np.float32)
    lineage_assignment = np.asarray(lineage_assignment, dtype=bool)

    n_genes = raw_counts.shape[1]
    
    if gene_names is None:
        gene_names = [f"Gene_{i}" for i in range(n_genes)]
    else:
        gene_names = list(gene_names)

    edges = []
    
    # Trains a KAN for each gene where the input are all other genes
    for target_idx in range(n_genes):
        target_gene = gene_names[target_idx]
        
        print(f"[{target_idx + 1}/{n_genes}] Processing gene: {target_gene}...")

        X_numpy, Y_numpy = get_lagged_expression(
            log_counts, pseudotime, lineage_assignment, target_idx, dt=dt
        )
        
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32).to(device)
        Y_tensor = torch.tensor(Y_numpy, dtype=torch.float32).to(device)
        
        kan_model = train_n_to_1_kan(
            X_tensor, Y_tensor, device=device, epochs=epochs, lr=lr, lamb_l1=lamb_l1
        )
        
        edge_signs = extract_kan_gradients(kan_model, X_tensor)
        edge_weights = extract_kan_weights(kan_model)
        
        predictor_genes = [name for idx, name in enumerate(gene_names) if idx != target_idx]
        
        for i, source_gene in enumerate(predictor_genes):
            abs_weight = abs(edge_weights[i])
            
            if abs_weight > edge_threshold:
                final_signed_weight = abs_weight * edge_signs[i]
                
                edges.append({
                    "Gene1": source_gene,
                    "Gene2": target_gene,
                    "EdgeWeight": final_signed_weight
                })
                
    df_edges = pd.DataFrame(edges)
    
    if not df_edges.empty:
        df_edges['AbsWeight'] = df_edges['EdgeWeight'].abs()
        df_edges = df_edges.sort_values(by="AbsWeight", ascending=False).drop(columns=['AbsWeight']).reset_index(drop=True)
        
    return df_edges