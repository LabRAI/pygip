ModelExtractionAttack1
======================

Query-based extraction that reads selected nodes from file and builds shadow graph.

**Parameters**

dataset : object
    Dataset containing graph structures.
attack_node_fraction : float
    Fraction of nodes for attack.
selected_node_file : str
    File containing selected node IDs.
query_label_file : str
    File containing query labels.
shadow_graph_file : str, optional
    File describing shadow graph structure.

**Methods**

attack()
    Executes attack procedure:
    1. Reads nodes & labels from files
    2. Constructs shadow graph
    3. Trains extraction model