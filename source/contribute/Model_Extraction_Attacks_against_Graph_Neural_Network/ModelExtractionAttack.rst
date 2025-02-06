ModelExtractionAttack
=====================

Base class for model extraction attacks on GNNs.

**Parameters**

dataset : object
    Dataset containing graph, features, labels, etc.
attack_node_fraction : float
    Fraction of nodes to use in attack.
model_path : str, optional
    Path to load pre-trained target model.

**Attributes**

dataset : object
    The dataset object.
graph : DGLGraph
    The graph structure.
node_number : int
    Total number of nodes.
feature_number : int
    Number of node features.
label_number : int
    Number of label classes.
attack_node_number : int
    Number of nodes for attack.
features : torch.Tensor
    Node features.
labels : torch.Tensor
    Node labels.
train_mask : torch.Tensor
    Training node mask.
test_mask : torch.Tensor
    Testing node mask.
net1 : nn.Module
    Target GCN model.

**Methods**

train_target_model()
    Trains target GCN model if not loaded from path.