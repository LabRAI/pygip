Citeseer
========

Implementation for the Citeseer citation network dataset.

From paper: "Semi-supervised classification with graph convolutional networks"

**Parameters**

api_type : str
    API selection ('dgl' or 'torch_geometric').
path : str
    Dataset directory path.

**Methods**

load_dgl_data()
    Load Citeseer using DGL's citation graph loader.
load_tg_data()
    Load Citeseer using PyG's Planetoid loader.