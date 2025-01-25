Cora
====

Implementation for the Cora citation network dataset.

From paper: "Semi-supervised classification with graph convolutional networks"

**Parameters**

api_type : str, optional
    API selection ('dgl' or 'torch_geometric'), defaults to 'dgl'.
path : str, optional
    Dataset directory path, defaults to './'.

**Methods**

load_dgl_data()
    Load Cora using DGL's citation graph loader.
load_tg_data()
    Load Cora using PyG's Planetoid loader.