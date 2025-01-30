graph_to_datasets
=================

A utility class that wraps a DGLGraph into a dataset-like structure.

**Parameters**

graph : dgl.DGLGraph
    The input graph to be wrapped.
attack_node_fraction : float
    The fraction of nodes to be used for attack purposes. Used to compute self.attack_node_number.
name : str, optional
    Name identifier for the dataset.

**Attributes**

graph : dgl.DGLGraph
    The wrapped graph with added self-loops.
dataset_name : str
    Name identifier of the dataset.
node_number : int
    Total number of nodes in the graph.
feature_number : int
    Dimension of node features.
label_number : int
    Number of unique labels in the graph.
attack_node_number : int
    Number of nodes designated for attacks.
features : torch.Tensor
    Node feature matrix.
labels : torch.Tensor
    Node labels.
train_mask : torch.Tensor
    Boolean mask indicating training nodes.
test_mask : torch.Tensor
    Boolean mask indicating test nodes.