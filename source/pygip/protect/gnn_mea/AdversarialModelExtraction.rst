AdversarialModelExtraction
==========================

A class implementing adversarial model extraction attacks on graph neural networks using DGL.

**Parameters**
dataset : Dataset
    The target graph dataset to perform extraction on.
attack_node_fraction : float
    Fraction of nodes to use in the attack.
model_path : str, optional
    Path to load pre-trained target model from.

**Attributes**
graph : DGLGraph
    The target graph to attack.
features : torch.Tensor
    Node feature matrix.
labels : torch.Tensor  
    Node label tensor.
label_number : int
    Number of unique classes.
node_number : int
    Total number of nodes.
feature_number : int
    Number of node features.
net1 : nn.Module
    Target model to extract.

**Methods**

attack()
--------
Executes the model extraction attack.

Process:
1. Selects a center node and extracts k-hop subgraph
2. Constructs prior distributions for features and node counts per class
3. Generates synthetic graphs using prior distributions
4. Queries target model for labels
5. Trains surrogate model on synthetic data

Implementation details:
- Uses k-hop subgraph sampling (k=2)
- Filters subgraphs to size 10-150 nodes
- Generates n=10 synthetic graphs per class
- Trains surrogate for 200 epochs using Adam optimizer
- Evaluates using fidelity and accuracy metrics

Returns:
- GraphNeuralNetworkMetric
    Performance metrics of extracted model

Key Features:
- Maintains feature distributions per class
- Preserves node feature count distributions
- Combines multiple subgraphs into training set
- Uses DGL for efficient graph operations
- Implements early stopping based on performance

Notes:
- Requires target model to be in eval mode
- Uses dropout and weight decay for regularization
- Monitors both fidelity to target and accuracy on test set
- Prints progress during extraction process
- Returns best achieved performance metrics

.. code-block:: python
    :caption: Example Python Code
    :linenos:

    # Importing necessary classes and functions from the pygip library.
    from pygip.datasets.datasets import *  # Import all available datasets.
    from pygip.protect import *  # Import all core algorithms.

    # Loading the Cora dataset, which is commonly used in graph neural network research.
    dataset = Cora()

    # Initializing a model extraction attack with the Cora dataset.
    adversarial_attack = AdversarialModelExtraction(dataset, 0.25)

    # Executing the attack on the model.
    adversarial_attack.attack()