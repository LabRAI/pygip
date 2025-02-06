SAGE
=====================

Base class for GraphSAGE implementation using DGL.

**Parameters**

in_feats : int
    Input feature dimensions.
n_hidden : int
    Number of hidden units.
n_classes : int
    Number of output classes.
n_layers : int
    Number of SAGE layers.
activation : callable
    Activation function.
batch_size : int
    Size of mini-batches.
num_workers : int
    Number of worker processes.
dropout : float
    Dropout rate for regularization.

**Attributes**

layers : nn.ModuleList
    List of SAGE convolutional layers and final linear layer.
n_layers : int
    Number of layers in the model.
n_hidden : int
    Size of hidden dimensions.
n_classes : int
    Number of output classes.
dropout : nn.Dropout
    Dropout layer.
activation : callable
    Activation function.
batch_size : int
    Size of training batches.
num_workers : int
    Number of data loading workers.

**Methods**

forward(blocks, x)
    Forward computation through SAGE layers.
    Returns predictions and embeddings.

inference(g, x, batch_size, device)
    Performs full-graph inference without neighbor sampling.
    Returns predictions and embeddings.

evaluate_sage_target(model, g, inputs, labels, val_nid, batch_size, device)
    Evaluates model performance on validation set.

run_sage_target(args, device, data)
    Runs training loop with distributed data loading.