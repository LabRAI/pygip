Gcn_Net
=======

A simple Graph Convolutional Network (GCN) implementation.

**Parameters**

feature_number : int
    Number of input features.
label_number : int  
    Number of output classes.

**Attributes**

layers : nn.ModuleList
    Contains GraphConv layers [feature_number->16, 16->label_number].
dropout : nn.Dropout
    Dropout layer with p=0.5.

**Methods**

forward(g: DGLGraph, features: torch.Tensor) -> torch.Tensor
    Forward pass computation.

    Parameters:
        g (DGLGraph): Input graph.
        features (torch.Tensor): Node features.
    
    Returns:
        torch.Tensor: Node logits.