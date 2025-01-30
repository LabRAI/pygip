Net_shadow
==========

A shadow model GCN used for model extraction or membership inference.

**Parameters**

feature_number : int
    Number of input features.
label_number : int
    Number of output classes.

**Attributes**

layer1 : GraphConv
    First graph convolution layer.
layer2 : GraphConv
    Second graph convolution layer.

**Methods**

forward(g: DGLGraph, features: torch.Tensor) -> torch.Tensor
    Forward pass of shadow model.

    Parameters:
        g (DGLGraph): Input graph.
        features (torch.Tensor): Node features.
    
    Returns:
        torch.Tensor: Node predictions.