PubMed
======

Implementation for the PubMed citation network dataset.

From paper: "Semi-supervised classification with graph convolutional networks"

**Parameters**

api_type : str
    API selection ('dgl' or 'torch_geometric').
path : str
    Dataset directory path.

**Methods**

load_dgl_data()
    Load PubMed using DGL's citation graph loader.
load_tg_data()
    Load PubMed using PyG's Planetoid loader.