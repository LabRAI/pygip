Net_attack
==========

An attack model GCN for model extraction.

**Parameters**

feature_number : int
    Number of input features.
label_number : int
    Number of output classes.

**Attributes**

layers : nn.ModuleList
    Contains GraphConv layers for attack.
dropout : nn.Dropout
    Dropout layer with p=0.5.

**Methods**

forward(g: DGLGraph, features: torch.Tensor) -> torch.Tensor
    Forward pass of attack model.

    Parameters:
        g (DGLGraph): Input graph.
        features (torch.Tensor): Node features.
    
    Returns:
        torch.Tensor: Attack predictions.