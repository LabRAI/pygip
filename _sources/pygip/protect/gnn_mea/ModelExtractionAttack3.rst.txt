ModelExtractionAttack3
======================

Shadow model attack using subgraph indices to build partial graphs.

**Parameters**

dataset : object
    Dataset containing graph structures.
attack_node_fraction : float
    Fraction of nodes for attack.
model_path : str, optional
    Path to pre-trained model.

**Methods**

attack()
    Executes attack procedure:
    1. Loads subgraph indices
    2. Merges adjacency matrices
    3. Trains attack model on combined graph