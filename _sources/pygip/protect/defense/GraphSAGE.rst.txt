GraphSAGE
=========

Implementation of the GraphSAGE model as described in the paper "Inductive Representation Learning on Large Graphs".

**Parameters**

in_channels : int
    Size of input feature dimensions.
hidden_channels : int
    Size of hidden layer dimensions.
out_channels : int
    Size of output dimensions (number of classes).

**Methods**

forward(blocks: List[dgl.DGLGraph], x: torch.Tensor) -> torch.Tensor
    Forward pass of the model.
    
    Parameters:
        blocks (List[dgl.DGLGraph]): List of sampled blocks for multi-layer GNN.
        x (torch.Tensor): Input node features.
    
    Returns:
        torch.Tensor: Output node representations.