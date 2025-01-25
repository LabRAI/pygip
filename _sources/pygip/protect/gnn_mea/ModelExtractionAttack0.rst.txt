ModelExtractionAttack0
======================

Basic extraction attack that queries a subset of nodes and synthesizes features based on multi-hop neighbors.

**Parameters**

dataset : object
    Dataset containing graph structures.
attack_node_fraction : float
    Fraction of nodes to use in attack.
model_path : str, optional
    Path to load pre-trained model.
alpha : float, optional
    Weight between first-order and second-order neighbors (default: 0.8).

**Methods**

get_nonzero_indices(matrix_row: np.ndarray) -> np.ndarray
    Gets indices of nonzero entries in adjacency matrix row.

    Returns:
        np.ndarray: Indices of nonzero entries.

attack()
    Executes attack procedure:
    1. Samples subset of nodes for querying
    2. Synthesizes neighbor features 
    3. Builds sub-graph and trains extraction model