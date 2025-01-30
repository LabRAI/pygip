SAGE_SURROGATE
=====================

Base class for GraphSAGE surrogate model implementation using DGL.

**Parameters**

in_feats : int
    Input feature dimensions.
n_hidden : int
    Number of hidden units.
n_output_dim : int
    Output embedding dimensions.
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
    List of SAGE convolutional layers.
n_layers : int
    Number of model layers.
n_hidden : int
    Size of hidden dimensions.
n_output_dim : int
    Size of output embedding.
n_classes : int
    Number of classification classes.
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
    Returns node embeddings.

inference(g, x, batch_size, device)
    Performs full-graph inference with neighbor sampling.
    Returns node embeddings.

evaluate_sage_surrogate(model, clf, g, inputs, labels, val_nid, batch_size, device)
    Evaluates surrogate model performance with classifier.

run_sage_surrogate(device, data, fan_out, batch_size, num_workers, num_hidden, num_layers, dropout, lr, num_epochs, log_every, eval_every)
    Runs surrogate model training with embedding learning and classification.