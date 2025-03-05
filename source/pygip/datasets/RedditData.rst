RedditData
==========

Implementation for the Reddit network dataset.

A graph dataset representing Reddit posts, where nodes are posts and edges indicate that the same user commented on both posts.

**Parameters**

api_type : str, optional
    API selection ('dgl' or 'torch_geometric'), defaults to 'dgl'.
path : str, optional
    Dataset directory path, defaults to './downloads/'.

**Methods**

.. py:method:: load_dgl_data()

    Load Reddit dataset using DGL's RedditDataset loader. Uses pre-defined
    node features, labels, and splits for training, validation, and testing.

.. py:method:: load_tg_data()

    Load Reddit dataset using PyTorch Geometric (PyG) loader.
    Extracts dataset features, node labels, and masks for training, validation, and testing.

**Attributes**

.. py:attribute:: graph
    :type: dgl.DGLGraph

    The graph data structure (when using DGL).

.. py:attribute:: dataset_name
    :type: str

    Name of the dataset ("reddit").

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

    Boolean mask indicating training nodes.

.. py:attribute:: val_mask
    :type: torch.Tensor

    Boolean mask indicating validation nodes.

.. py:attribute:: test_mask
    :type: torch.Tensor

    Boolean mask indicating testing nodes.
    
.. py:attribute:: dataset
    :type: torch_geometric.datasets.Reddit

    The PyG dataset object (only available in PyG implementation).
    
.. py:attribute:: data
    :type: torch_geometric.data.Data

    The PyG data object (only available in PyG implementation).
    
.. py:attribute:: edge_index
    :type: torch.Tensor

    Edge list representation (only explicitly set in PyG implementation).