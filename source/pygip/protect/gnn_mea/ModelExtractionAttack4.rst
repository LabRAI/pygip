ModelExtractionAttack4
======================

Enhanced shadow model attack with feature-based edge linking.

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
    1. Reads node indices
    2. Links edges based on feature similarity
    3. Trains model on augmented graph