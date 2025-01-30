gin
=====================

Base class for Graph Isomorphism Network implementation using DGL.

**Parameters**

in_feats : int
    Input feature dimensions.
n_hidden : int
    Number of hidden units.
n_classes : int
    Number of output classes.
n_layers : int
    Number of GIN layers.
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
    List of GIN layers.
n_layers : int
    Number of GIN layers.
n_hidden : int
    Size of hidden dimensions.
n_classes : int
    Number of output classes.
dropout : nn.Dropout
    Dropout layer.
activation : callable
    Activation function.
batch_size : int
    Size of mini-batches.
num_workers : int
    Number of data loading workers.

**Methods**

forward(blocks, x)
    Forward computation through GIN layers.

inference(g, x, batch_size, device)
    Performs full-graph inference.

evaluate_gin_target(model, g, inputs, labels, val_nid, batch_size, device)
    Evaluates model performance on validation set.

run_gin_target(args, device, data)
    Runs GIN model training with distributed data loading.