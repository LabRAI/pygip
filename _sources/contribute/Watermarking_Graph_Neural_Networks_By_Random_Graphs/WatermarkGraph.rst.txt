WatermarkGraph
==============

Implementation of a watermark graph generation mechanism based on the Erdős-Rényi random graph model.

**Parameters**

n : int
    Number of nodes in the watermark graph.
num_features : int
    Dimension of node features.
num_classes : int
    Number of possible node classes.
pr : float, optional
    Probability for generating 1s in node features (via binomial distribution), default is 0.1.
pg : float, optional
    Probability for edge creation in Erdős-Rényi graph, default is 0.
device : torch.device, optional
    Device to place the generated graph on.

**Attributes**

pr : float
    Stored probability for feature generation.
pg : float
    Stored probability for edge generation.
device : torch.device
    Device where the graph is stored.
graph_wm : dgl.DGLGraph
    The generated watermark graph.

**Methods**

_generate_wm(n: int, num_features: int, num_classes: int) -> dgl.DGLGraph
    Internal method to generate the watermark graph.