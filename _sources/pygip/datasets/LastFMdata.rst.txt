LastFMdata
==========

Implementation for the Last.fm Asia network dataset.

A network representing the Asian music artist listening information from Last.fm.

**Parameters**

api_type : str, optional
    API selection ('dgl' or 'torch_geometric'), defaults to 'dgl'.
path : str, optional
    Dataset directory path, defaults to './downloads/'.

**Methods**

.. py:method:: load_dgl_data()

    Load Last.fm Asia dataset using DGL. Creates a graph from edge indices,
    assigns node features and labels, and randomly splits nodes into training
    and testing sets based on train_ratio.

.. py:method:: load_tg_data()

    Load Last.fm Asia dataset using PyTorch Geometric (PyG) loader.
    Extracts dataset features, node labels, and masks for training and testing.

**Attributes**

.. py:attribute:: graph
    :type: dgl.DGLGraph

    The graph data structure (when using DGL).

.. py:attribute:: dataset_name
    :type: str

    Name of the dataset ("last-fm").

.. py:attribute:: train_ratio
    :type: float

    Ratio of nodes used for training (0.8 in DGL implementation).

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

    Boolean mask indicating training nodes (randomly selected in DGL implementation).

.. py:attribute:: test_mask
    :type: torch.Tensor

    Boolean mask indicating testing nodes.

.. py:attribute:: var_mask
    :type: torch.Tensor

    Boolean mask for validation (only available in PyG implementation).
    
.. py:attribute:: dataset
    :type: torch_geometric.datasets.LastFMAsia

    The PyG dataset object (only available in PyG implementation).
    
.. py:attribute:: data
    :type: torch_geometric.data.Data

    The PyG data object (only available in PyG implementation).
    
.. py:attribute:: edge_index
    :type: torch.Tensor

    Edge list representation (only explicitly set in PyG implementation).