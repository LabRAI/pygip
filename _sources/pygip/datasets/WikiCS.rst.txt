WikiCS
======

Implementation for the WikiCS dataset.

A Wikipedia-based dataset for graph neural networks with node features and article categories.

**Parameters**

api_type : str, optional
    API selection ('dgl' or 'torch_geometric'), defaults to 'dgl'.
path : str, optional
    Dataset directory path, defaults to './downloads/'.

**Methods**

.. py:method:: load_dgl_data()

    Load WikiCS dataset using DGL's WikiCSDataset loader. Adds self-loops to the graph
    and prepares node features, labels, and masks for training, validation, and testing.

.. py:method:: load_tg_data()

    Load WikiCS dataset using PyTorch Geometric (PyG) loader.
    
    Note: Implementation not shown in the provided code.

**Attributes**

.. py:attribute:: graph
    :type: dgl.DGLGraph

    The graph data structure with self-loops added.

.. py:attribute:: dataset_name
    :type: str

    Name of the dataset ("wikics").

.. py:attribute:: node_number
    :type: int

    Number of nodes in the graph.

.. py:attribute:: feature_number
    :type: int

    Dimension of node features.

.. py:attribute:: label_number
    :type: int

    Number of unique node labels.

.. py:attribute:: features
    :type: torch.Tensor

    Node feature matrix.

.. py:attribute:: labels
    :type: torch.Tensor

    Node label tensor.

.. py:attribute:: train_mask
    :type: torch.Tensor

    Boolean mask indicating training nodes (uses the second column of the original train_mask).

.. py:attribute:: val_mask
    :type: torch.Tensor

    Boolean mask indicating validation nodes.

.. py:attribute:: test_mask
    :type: torch.Tensor

    Boolean mask indicating testing nodes.