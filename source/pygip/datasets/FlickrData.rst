Flickr
======

Implementation for the Facebook Page-Page network dataset.

**Parameters**

api_type : str, optional
    API selection ('dgl' or 'torch_geometric'), defaults to 'dgl'.
path : str, optional
    Dataset directory path, defaults to './downloads/'.

**Methods**

load_dgl_data()
    Load Facebook data using PyG loader and convert to DGL.
load_tg_data()
    Load Facebook data directly using PyG.