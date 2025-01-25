ModelExtractionAttack2
======================

Structure-based extraction using randomly sampled nodes and identity features.

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
    1. Randomly selects training nodes
    2. Creates identity features
    3. Trains Net_attack model