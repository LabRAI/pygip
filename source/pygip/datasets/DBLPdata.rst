DBLPdata
=========

Implementation for the DBLP citation network dataset.

A heterogeneous graph dataset containing nodes representing authors and papers and their relationships.

**Parameters**

api_type : str, optional
    API selection ('dgl' or 'torch_geometric'), defaults to 'dgl'.
path : str, optional
    Dataset directory path, defaults to './'.

**Methods**

.. py:method:: load_dgl_data()

    Load DBLP dataset using DGL's loader. Extracts author-paper relationships, 
    creates an author subgraph with features and labels, and prepares training and testing masks.

.. py:method:: load_tg_data()

    Load DBLP dataset using PyTorch Geometric (PyG) loader. Extracts dataset features, 
    node labels, and masks for training and testing.

**Attributes**

.. py:attribute:: graph
    :type: dgl.DGLGraph

    The graph data structure (when using DGL).

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

.. py:attribute:: test_mask
    :type: torch.Tensor

    Boolean mask indicating testing nodes.

.. py:attribute:: var_mask
    :type: torch.Tensor

    Boolean mask for validation (only available in PyG implementation).

.. py:attribute:: dataset_name
    :type: str

    Name of the dataset ("dblp", only set in PyG implementation).

.. py:attribute:: edge_index
    :type: torch.Tensor

    Edge list representation (only explicitly set in PyG implementation).