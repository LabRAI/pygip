Photo
=====

Implementation for the Amazon Photo co-purchase network.

**Parameters**

api_type : str, optional
    API selection ('dgl' or 'torch_geometric'), defaults to 'dgl'.
path : str, optional
    Dataset directory path, defaults to './downloads/'.

**Methods**

load_dgl_data()
    Load Photo data using DGL's dataset loader.