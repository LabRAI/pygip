GAT
===========

Base class for Graph Attention Network model implementation using DGL.

**Parameters**

in_feats : int
    Input feature dimensions.
n_hidden : int
    Number of hidden units.
n_classes : int
    Number of output classes.
n_layers : int
    Number of GAT layers.
num_heads : int
    Number of attention heads.
num_workers : int
    Number of worker processes.
activation : callable
    Activation function.
dropout : float
    Dropout rate for regularization.
model_path : str, optional
    Path to load pre-trained model.

**Attributes**

layers : nn.ModuleList
    The GAT layer stack.
n_layers : int
    Number of GAT layers.
n_hidden : int
    Size of hidden dimensions.
n_classes : int
    Number of output classes.
num_workers : int
    Number of data loading workers.
device : torch.device
    Device for computation.
batch_size : int
    Size of mini-batches.
features : torch.Tensor
    Node feature matrix.
labels : torch.Tensor
    Node label tensor.
train_mask : torch.Tensor
    Training node mask.
val_mask : torch.Tensor
    Validation node mask.
test_mask : torch.Tensor
    Testing node mask.

**Methods**

forward(blocks, x)
    Forward computation through GAT layers.

inference(g, x, batch_size, num_heads, device)
    Performs full-graph inference.

train_model()
    Trains GAT model if not loaded from path.

evaluate(g, features, labels, mask)
    Evaluates model performance on given graph.