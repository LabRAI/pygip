Watermark_sage
==============

Implementation of GraphSAGE-based watermarking as proposed in "Watermarking Graph Neural Networks by Random Graphs".

**Parameters**

dataset : graph_to_dataset
    Dataset wrapper containing the graph and its attributes.
attack_node_fraction : float
    Fraction of nodes to be used for attack purposes.
wm_node : int, optional
    Number of nodes in watermark graph, default is 50.
pr : float, optional
    Probability for generating features, default is 0.1.
pg : float, optional
    Probability for generating edges, default is 0.
device : torch.device, optional
    Device to place the model on.

**Methods**

erdos_renyi_graph(n: int, p: float, directed: bool = False) -> torch.Tensor
    Generates Erdős-Rényi random graph.
    
    Parameters:
        n (int): Number of nodes.
        p (float): Edge probability.
        directed (bool): Whether to create directed graph.
    
    Returns:
        torch.Tensor: Edge index tensor.

attack()
    Performs watermark injection process through two-stage training.