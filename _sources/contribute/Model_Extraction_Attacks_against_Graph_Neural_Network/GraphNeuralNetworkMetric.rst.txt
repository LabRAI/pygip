GraphNeuralNetworkMetric
========================

Evaluation metrics class for Graph Neural Networks.

**Parameters**

fidelity : float, optional
    Initial fidelity score, defaults to 0.
accuracy : float, optional
    Initial accuracy score, defaults to 0.
model : nn.Module, optional
    PyTorch model to evaluate.
graph : DGLGraph, optional
    The graph structure to evaluate on.
features : torch.Tensor, optional
    Node features of the graph.
mask : torch.Tensor, optional
    Boolean mask for selecting nodes.
labels : torch.Tensor, optional
    Ground truth labels.
query_labels : torch.Tensor, optional
    Labels obtained from model queries.

**Attributes**

model : nn.Module or None
    The model being evaluated.
graph : DGLGraph or None
    The graph structure.
features : torch.Tensor or None
    Node feature matrix.
mask : torch.Tensor or None
    Evaluation mask.
labels : torch.Tensor or None
    Ground truth labels.
query_labels : torch.Tensor or None
    Query result labels.
accuracy : float
    Current accuracy score.
fidelity : float
    Current fidelity score.

**Methods**

evaluate_helper(model, graph, features, labels, mask) -> float or None
    Helper method to compute model accuracy.
    
    Parameters:
        model (nn.Module): The model to evaluate.
        graph (DGLGraph): Input graph.
        features (torch.Tensor): Node features.
        labels (torch.Tensor): True labels.
        mask (torch.Tensor): Node mask.
    
    Returns:
        float or None: Evaluation accuracy if inputs valid, None otherwise.

evaluate()
    Updates fidelity and accuracy metrics through evaluation.

__str__() -> str
    Returns string representation of metrics.