MutaData
========

Implementation for the MUTAG dataset.

A chemical compound dataset used for predicting mutagenicity on heterocyclic aromatic compounds.

**Parameters**

api_type : str
    API selection ('dgl' or 'torch_geometric').
path : str
    Dataset directory path.

**Methods**

.. py:method:: load_dgl_data()

    Load MUTAG dataset using DGL's MUTAGDataset loader. Extracts graph structure, 
    node features, and labels for the chemical compounds.

**Attributes**

.. py:attribute:: graph
    :type: dgl.DGLGraph

    The graph data structure representing chemical compounds.

.. py:attribute:: dataset_name
    :type: str

    Name of the dataset ("mutag").

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

    Graph labels from the dataset's predict_category.

.. py:attribute:: train_mask
    :type: torch.Tensor

    Boolean mask indicating training nodes.

.. py:attribute:: val_mask
    :type: torch.Tensor

    Boolean mask indicating validation nodes.

.. py:attribute:: test_mask
    :type: torch.Tensor

    Boolean mask indicating testing nodes.

.. note::
    The implementation references an undefined variable 'category' when setting masks,
    which may need correction in the actual code.