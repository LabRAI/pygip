import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import dgl
from dgl import DGLGraph 
from torch_geometric.data import Data as PyGData

from dgl.data import citation_graph as citegrh
from torch_geometric.datasets import Planetoid          ### Cora, CiteSeer, PubMed
from torch_geometric.datasets import DBLP               ### DBLP
from dgl.data import WikiCSDataset                      ### WikiCS
from dgl.data import YelpDataset     
from torch_geometric.datasets import Yelp               ### YelpData
from torch_geometric.datasets import FacebookPagePage   ### Facebook
from dgl.data import FlickrDataset
from torch_geometric.datasets import Flickr             ### FlickrData
from torch_geometric.datasets import PolBlogs           ### Polblogs
from torch_geometric.datasets import LastFMAsia         ### LastFM
from dgl.data import RedditDataset
from torch_geometric.datasets import Reddit             ### RedditData
from dgl.data import AmazonCoBuyComputerDataset         ### Computer
from dgl.data import AmazonCoBuyPhotoDataset            ### Photo
from dgl.data import MUTAGDataset                       ### MUTAGData
from dgl.data import GINDataset                         ### Collab, NCI1, PROTEINS, PTC, IMDB-BINARY
from dgl.data import FakeNewsDataset                    ### Twitter


def dgl_to_tg(dgl_graph):
    """
    Convert a DGLGraph to a PyG Data object.

    Parameters
    ----------
    dgl_graph : dgl.DGLGraph
        The input DGL graph to be converted.

    Returns
    -------
    PyGData
        The resulting PyTorch Geometric Data object containing node features (x),
        edge information (edge_index), labels (y), and train/val/test masks.
    """
    edge_index = torch.stack(dgl_graph.edges())
    x = dgl_graph.ndata.get('feat')
    y = dgl_graph.ndata.get('label')
    
    train_mask = dgl_graph.ndata.get('train_mask')
    val_mask = dgl_graph.ndata.get('val_mask')
    test_mask = dgl_graph.ndata.get('test_mask')

    data = PyGData(
        x=x, 
        edge_index=edge_index, 
        y=y,
        train_mask=train_mask, 
        val_mask=val_mask, 
        test_mask=test_mask
    )
    return data


def tg_to_dgl(py_g_data):
    """
    Convert a PyG Data object to a DGLGraph.

    Parameters
    ----------
    py_g_data : PyGData
        The PyTorch Geometric Data object to be converted.

    Returns
    -------
    dgl.DGLGraph
        The resulting DGL graph with node features, labels, and train/val/test masks attached.
    """
    edge_index = py_g_data.edge_index
    dgl_graph = dgl.graph((edge_index[0], edge_index[1]))

    if py_g_data.x is not None:
        dgl_graph.ndata['feat'] = py_g_data.x
    if py_g_data.y is not None:
        dgl_graph.ndata['label'] = py_g_data.y

    if hasattr(py_g_data, 'train_mask') and py_g_data.train_mask is not None:
        dgl_graph.ndata['train_mask'] = py_g_data.train_mask
    if hasattr(py_g_data, 'val_mask') and py_g_data.val_mask is not None:
        dgl_graph.ndata['val_mask'] = py_g_data.val_mask
    if hasattr(py_g_data, 'test_mask') and py_g_data.test_mask is not None:
        dgl_graph.ndata['test_mask'] = py_g_data.test_mask

    return dgl_graph


class Dataset(object):
    """
    Base class for handling and loading GNN datasets in both DGL and PyG formats.

    Attributes
    ----------
    api_type : str
        Specifies which API to use ('dgl' or 'torch_geometric').
    path : str
        The directory path where the dataset is located or will be downloaded.
    dataset_name : str
        The name of the dataset.
    node_number : int
        Total number of nodes in the graph.
    feature_number : int
        Dimensionality of the node features.
    label_number : int
        Number of distinct labels or classes.
    features : torch.Tensor
        Node features.
    labels : torch.Tensor
        Node labels.
    train_mask : torch.Tensor
        Boolean mask indicating which nodes are used for training.
    test_mask : torch.Tensor
        Boolean mask indicating which nodes are used for testing.
    """

    def __init__(self, api_type='dgl', path='./downloads/'):
        """
        Initialize the Dataset class.

        Parameters
        ----------
        api_type : str, optional
            Type of graph API to use ('dgl' or 'torch_geometric'). Default is 'dgl'.
        path : str, optional
            Path to download or load the dataset. Default is './downloads/'.
        """
        self.api_type = api_type
        self.path = path
        self.dataset_name = ""

        self.node_number = 0
        self.feature_number = 0
        self.label_number = 0

        self.features = None
        self.labels = None
        
        self.train_mask = None
        self.test_mask = None 

    def load_dgl_data(self):
        """
        Load the dataset using DGL format.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclass.
        """
        raise NotImplementedError("load_dgl_data not implemented in subclasses.")
    
    def load_tg_data(self):
        """
        Load the dataset using PyG format.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclass.
        """
        raise NotImplementedError("load_tg_data not implemented in subclasses.")
    
    def generate_train_test_masks(self):
        """
        Generate train, validation, and test masks using the attribute 'train_ratio'
        (and a derived approach for val/test). By default, splits data randomly.

        Note
        ----
        Not all subclasses use this method. Some datasets inherently have train/val/test masks.
        """
        num_nodes = self.node_number
        indices = torch.randperm(num_nodes)
        train_size = int(self.train_ratio * num_nodes)
        val_size = (num_nodes - train_size) // 2

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask


class Cora(Dataset):
    """
    A class for loading the Cora dataset in either DGL or PyG format.
    """
    def __init__(self, api_type='dgl', path='./'):
        """
        Initialize Cora dataset.

        Parameters
        ----------
        api_type : str, optional
            Either 'dgl' or 'torch_geometric'. Default is 'dgl'.
        path : str, optional
            File path for dataset storage or loading. Default is './'.
        """
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        """
        Load the Cora dataset using DGL's built-in citation graph loader.
        Updates attributes such as node_number, feature_number, label_number, etc.
        """
        data = citegrh.load_cora()
        self.graph = data[0]
        self.dataset_name = "cora"

        self.node_number = self.graph.number_of_nodes()
        self.feature_number = len(self.graph.ndata['feat'][0])
        self.label_number = int(max(self.graph.ndata['label']) - min(self.graph.ndata['label'])) + 1

        self.features = torch.FloatTensor(self.graph.ndata['feat'])
        self.labels = torch.LongTensor(self.graph.ndata['label'])

        self.train_mask = torch.BoolTensor(self.graph.ndata['train_mask'])
        self.test_mask = torch.BoolTensor(self.graph.ndata['test_mask'])

    def load_tg_data(self):
        """
        Load the Cora dataset using the PyTorch Geometric Planetoid class.
        Extract node features, labels, and default train/test/var masks.
        """
        dataset = Planetoid(root=self.path, name='Cora')
        data = dataset[0]
        self.dataset_name = "cora"

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes 

        self.features = data.x
        self.labels = data.y

        self.train_mask = data.train_mask
        self.test_mask = data.test_mask
        self.var_mask = data.var_mask
        
        self.node_number = data.num_nodes
        self.edge_index = data.edge_index


class Citeseer(Dataset):
    """
    A class for loading the Citeseer dataset in either DGL or PyG format.
    """
    def __init__(self, api_type, path):
        """
        Initialize Citeseer dataset.

        Parameters
        ----------
        api_type : str
            Either 'dgl' or 'torch_geometric'.
        path : str
            File path for dataset storage or loading.
        """
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        """
        Load the Citeseer dataset using DGL's citation graph loader.
        """
        data = citegrh.load_citeseer()
        self.graph = data[0]
        self.dataset_name = "citeseer"

        self.node_number = self.graph.number_of_nodes()
        self.feature_number = len(self.graph.ndata['feat'][0])
        self.label_number = int(max(self.graph.ndata['label']) - min(self.graph.ndata['label'])) + 1

        self.features = torch.FloatTensor(self.graph.ndata['feat'])
        self.labels = torch.LongTensor(self.graph.ndata['label'])

        self.train_mask = torch.BoolTensor(self.graph.ndata['train_mask'])
        self.test_mask = torch.BoolTensor(self.graph.ndata['test_mask'])

    def load_tg_data(self):
        """
        Load the Citeseer dataset using PyTorch Geometric's Planetoid class.
        """
        dataset = Planetoid(root=self.path, name='Citeseer')
        data = dataset[0]
        self.dataset_name = "citeseer"

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes 

        self.features = data.x
        self.labels = data.y

        self.train_mask = data.train_mask
        self.test_mask = data.test_mask
        self.var_mask = data.var_mask
        
        self.node_number = data.num_nodes
        self.edge_index = data.edge_index


class DBLPdata(Dataset):
    """
    A class for loading the DBLP dataset in either DGL or PyG format.
    """
    def __init__(self, api_type='dgl', path='./'):
        """
        Initialize DBLP dataset.

        Parameters
        ----------
        api_type : str, optional
            Either 'dgl' or 'torch_geometric'. Default is 'dgl'.
        path : str, optional
            Path for dataset storage or loading. Default is './'.
        """
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        """
        Load DBLP data for DGL. Constructs a subgraph based on 'author' nodes 
        and their features/labels, then populates relevant attributes.
        """
        dataset = DBLP(self.path)
        dblp_data = dataset[0]

        edge_index = dblp_data['author', 'to', 'paper']._mapping['edge_index'].numpy()
        num_nodes = max(edge_index[0].max(), edge_index[1].max()) + 1

        self.graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
        
        author_node_indices = np.unique(edge_index[0])
        author_subgraph = dgl.node_subgraph(self.graph, author_node_indices)
        author_subgraph.ndata['feat'] = dblp_data['author'].x
        author_subgraph.ndata['label'] = dblp_data['author'].y
        
        self.train_mask = dblp_data['author'].train_mask
        self.test_mask = dblp_data['author'].test_mask

        self.graph = author_subgraph
        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']

    def load_tg_data(self):
        """
        Load DBLP data for PyTorch Geometric.
        """
        dataset = DBLP(root=self.path)
        data = dataset[0]
        self.dataset_name = "dblp"

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes 

        self.features = data.x
        self.labels = data.y

        self.train_mask = data.train_mask
        self.test_mask = data.test_mask
        self.var_mask = data.var_mask
        
        self.node_number = data.num_nodes
        self.edge_index = data.edge_index


class PubMed(Dataset):
    """
    A class for loading the PubMed dataset in either DGL or PyG format.
    """
    def __init__(self, api_type, path):
        """
        Initialize PubMed dataset.

        Parameters
        ----------
        api_type : str
            Either 'dgl' or 'torch_geometric'.
        path : str
            Path to load/store the dataset.
        """
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        """
        Load PubMed data using DGL's citation graph loader.
        """
        data = citegrh.load_pubmed()
        self.graph = data[0]
        self.dataset_name = "pubmed"

        self.node_number = self.graph.number_of_nodes()
        self.feature_number = len(self.graph.ndata['feat'][0])
        self.label_number = int(max(self.graph.ndata['label']) - min(self.graph.ndata['label'])) + 1

        self.features = torch.FloatTensor(self.graph.ndata['feat'])
        self.labels = torch.LongTensor(self.graph.ndata['label'])

        self.train_mask = torch.BoolTensor(self.graph.ndata['train_mask'])
        self.test_mask = torch.BoolTensor(self.graph.ndata['test_mask'])

    def load_tg_data(self):
        """
        Load PubMed data using PyTorch Geometric's Planetoid class.
        """
        dataset = Planetoid(root=self.path, name='PubMed')
        data = dataset[0]
        self.dataset_name = "pubmed"

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes 

        self.features = data.x
        self.labels = data.y

        self.train_mask = data.train_mask
        self.test_mask = data.test_mask
        self.var_mask = data.var_mask
      
        self.node_number = data.num_nodes
        self.edge_index = data.edge_index


class WikiCS(Dataset):
    """
    A class for loading the WikiCS dataset in either DGL or PyG format.
    """
    def __init__(self, api_type='dgl', path='./downloads/'):
        """
        Initialize WikiCS dataset.

        Parameters
        ----------
        api_type : str, optional
            Either 'dgl' or 'torch_geometric'. Default is 'dgl'.
        path : str, optional
            Path for dataset storage or loading. Default is './downloads/'.
        """
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        """
        Load the WikiCS dataset using DGL. A self-loop is added, 
        and default train/val/test masks are derived from the original dataset.
        """
        dataset = WikiCSDataset(raw_dir=self.path)
        self.graph = dgl.add_self_loop(dataset[0])
        self.dataset_name = "wikics"

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']

        train_mask = self.graph.ndata['train_mask']
        self.train_mask = train_mask[:, 1].bool()
        self.val_mask = self.graph.ndata['val_mask']
        self.test_mask = self.graph.ndata['test_mask']


####################################################################################################
class FacebookData(Dataset):
    """
    A class for loading the Facebook Page-Page dataset in either DGL or PyG format.
    """
    def __init__(self, api_type='dgl', path='./downloads/'):
        """
        Initialize Facebook dataset.

        Parameters
        ----------
        api_type : str, optional
            Either 'dgl' or 'torch_geometric'. Default is 'dgl'.
        path : str, optional
            Path for dataset storage or loading. Default is './downloads/'.
        """
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")  

    def load_dgl_data(self):
        """
        Load Facebook Page-Page data using PyTorch Geometric, convert to DGL,
        and generate a random train/test mask with ratio 0.8 by default.
        """
        dataset = FacebookPagePage(root=self.path)
        data = dataset[0]
        self.train_ratio = 0.8
        self.dataset_name = "facebook"

        edge_index = data.edge_index.numpy()
        self.graph = dgl.graph((edge_index[0], edge_index[1]))

        self.graph.ndata['feat'] = data.x  
        self.graph.ndata['label'] = data.y 

        self.train_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)
        self.test_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)
        num_train = int(self.graph.num_nodes() * self.train_ratio)
        perm = torch.randperm(self.graph.num_nodes())
        self.train_mask[perm[:num_train]] = True
        self.test_mask[perm[num_train:]] = True

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]  
        self.label_number = len(torch.unique(self.graph.ndata['label'])) 
        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']

    def load_tg_data(self):
        """
        Load Facebook Page-Page data directly using PyTorch Geometric.
        """
        dataset = FacebookPagePage(root=self.path)
        data = dataset[0]
        self.dataset_name = "facebook"

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes 

        self.features = data.x
        self.labels = data.y

        self.train_mask = data.train_mask
        self.test_mask = data.test_mask
        self.var_mask = data.var_mask
        
        self.node_number = data.num_nodes
        self.edge_index = data.edge_index


class FlickrData(Dataset):
    """
    A class for loading the Flickr dataset in either DGL or PyG format.
    """
    def __init__(self, api_type='dgl', path='./downloads/'):
        """
        Initialize Flickr dataset.

        Parameters
        ----------
        api_type : str, optional
            Either 'dgl' or 'torch_geometric'. Default is 'dgl'.
        path : str, optional
            Path for dataset storage or loading. Default is './downloads/'.
        """
        super().__init__(api_type, path)
        
        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        """
        Load the Flickr dataset using DGL's built-in FlickrDataset.
        """
        dataset = FlickrDataset(self.path)
        self.graph = dataset[0]
        self.dataset_name = "flickr"

        self.node_number = self.graph.number_of_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']

        self.train_mask = self.graph.ndata['train_mask']
        self.val_mask = self.graph.ndata['val_mask']
        self.test_mask = self.graph.ndata['test_mask']

    def load_tg_data(self):
        """
        Load the Flickr dataset using PyTorch Geometric's Flickr class.
        """
        dataset = Flickr(self.path)
        data = dataset[0]
        self.dataset_name = "flickr"

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes

        self.features = data.x
        self.labels = data.y

        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask
        
        self.node_number = data.num_nodes
        self.edge_index = data.edge_index


class PolblogsData(Dataset):
    """
    A class for loading the Polblogs dataset in either DGL or PyG format.
    """
    def __init__(self, api_type='dgl', path='./downloads/'):
        """
        Initialize Polblogs dataset.

        Parameters
        ----------
        api_type : str, optional
            Either 'dgl' or 'torch_geometric'. Default is 'dgl'.
        path : str, optional
            Path for dataset storage or loading. Default is './downloads/'.
        """
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")  

    def load_dgl_data(self):
        """
        Load Polblogs data using PyTorch Geometric, then convert to DGL.
        Randomly split nodes into train and test using a 0.8 ratio.
        """
        dataset = PolBlogs(root=self.path)
        data = dataset[0]
        self.train_ratio = 0.8
        self.dataset_name = "polblogs"

        edge_index = data.edge_index.numpy()
        self.graph = dgl.graph((edge_index[0], edge_index[1]))

        self.graph.ndata['feat'] = data.x  
        self.graph.ndata['label'] = data.y 

        self.train_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)
        self.test_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)
        num_train = int(self.graph.num_nodes() * self.train_ratio)
        perm = torch.randperm(self.graph.num_nodes())
        self.train_mask[perm[:num_train]] = True
        self.test_mask[perm[num_train:]] = True

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]  
        self.label_number = len(torch.unique(self.graph.ndata['label'])) 
        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']

    def load_tg_data(self):
        """
        Load Polblogs data using PyTorch Geometric.
        """
        dataset = PolBlogs(root=self.path)
        data = dataset[0]
        self.dataset_name = "polblogs"

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes 

        self.features = data.x
        self.labels = data.y

        self.train_mask = data.train_mask
        self.test_mask = data.test_mask
        self.var_mask = data.var_mask
        
        self.node_number = data.num_nodes
        self.edge_index = data.edge_index


class LastFMdata(Dataset):
    """
    A class for loading the LastFM Asia dataset in either DGL or PyG format.
    """
    def __init__(self, api_type='dgl', path='./downloads/'):
        """
        Initialize LastFM Asia dataset.

        Parameters
        ----------
        api_type : str, optional
            Either 'dgl' or 'torch_geometric'. Default is 'dgl'.
        path : str, optional
            Path for dataset storage or loading. Default is './downloads/'.
        """
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")  

    def load_dgl_data(self):
        """
        Load LastFM Asia data using PyTorch Geometric, convert to DGL,
        and randomly split the data with a 0.8 train ratio.
        """
        dataset = LastFMAsia(root=self.path)
        data = dataset[0]
        self.train_ratio = 0.8
        self.dataset_name = "last-fm"

        edge_index = data.edge_index.numpy()
        self.graph = dgl.graph((edge_index[0], edge_index[1]))

        self.graph.ndata['feat'] = data.x  
        self.graph.ndata['label'] = data.y 

        self.train_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)
        self.test_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)
        num_train = int(self.graph.num_nodes() * self.train_ratio)
        perm = torch.randperm(self.graph.num_nodes())
        self.train_mask[perm[:num_train]] = True
        self.test_mask[perm[num_train:]] = True

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]  
        self.label_number = len(torch.unique(self.graph.ndata['label'])) 
        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']

    def load_tg_data(self):
        """
        Load LastFM Asia data directly using PyTorch Geometric.
        """
        dataset = LastFMAsia(root=self.path)
        data = dataset[0]
        self.dataset_name = "last-fm"

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes 

        self.features = data.x
        self.labels = data.y

        self.train_mask = data.train_mask
        self.test_mask = data.test_mask
        self.var_mask = data.var_mask
        
        self.node_number = data.num_nodes
        self.edge_index = data.edge_index


class RedditData(Dataset):
    """
    A class for loading the Reddit dataset in either DGL or PyG format.
    """
    def __init__(self, api_type='dgl', path='./downloads/'):
        """
        Initialize Reddit dataset.

        Parameters
        ----------
        api_type : str, optional
            Either 'dgl' or 'torch_geometric'. Default is 'dgl'.
        path : str, optional
            Path for dataset storage or loading. Default is './downloads/'.
        """
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        """
        Load Reddit data using DGL's built-in RedditDataset. 
        Automatically includes train/val/test masks.
        """
        dataset = RedditDataset(raw_dir=self.path)
        self.graph = dataset[0]
        self.dataset_name = "reddit"

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']

        self.train_mask = self.graph.ndata['train_mask']
        self.val_mask = self.graph.ndata['val_mask']
        self.test_mask = self.graph.ndata['test_mask']

    def load_tg_data(self):
        """
        Load Reddit data using PyTorch Geometric's Reddit class.
        """
        dataset = Reddit(self.path)
        data = dataset[0]
        self.dataset_name = "reddit"

        self.dataset = dataset
        self.data = data
        self.feature_number = dataset.num_node_features
        self.label_number = dataset.num_classes

        self.features = data.x
        self.labels = data.y

        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask
        
        self.node_number = data.num_nodes
        self.edge_index = data.edge_index


class Twitter(Dataset):
    """
    A class for loading the Twitter FakeNews dataset in either DGL or PyG format.
    """
    def __init__(self, api_type, path):
        """
        Initialize Twitter dataset.

        Parameters
        ----------
        api_type : str
            Either 'dgl' or 'torch_geometric'.
        path : str
            Path for dataset storage or loading.
        """
        super().__init__(api_type, path)
        
        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")
        
    def load_dgl_data(self):
        """
        Load Twitter FakeNews data using DGL's FakeNewsDataset for 'gossipcop' 
        with 'bert' features, add self-loops, and generate a random train/test split.
        """
        dataset = FakeNewsDataset('gossipcop', 'bert', raw_dir=self.path)
        graph, _ = dataset[0]
        self.graph = dgl.add_self_loop(graph)
        self.dataset_name = "twitter"
        self.train_ratio = 0.8

        if hasattr(dataset, 'feature'):
            node_ids = self.graph.ndata['_ID'].numpy()
            selected_features = dataset.feature[node_ids]
            self.graph.ndata['feat'] = selected_features.float() 

        if hasattr(dataset, 'labels'):
            selected_labels = dataset.labels[node_ids]
            self.graph.ndata['label'] = selected_labels.long() 

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']

        self.generate_train_test_masks()


####################################################################################################
class MutaData(Dataset):
    """
    A class for loading the MUTAG dataset in either DGL or PyG format.
    """
    def __init__(self, api_type, path):
        """
        Initialize MutaData (MUTAG) dataset.

        Parameters
        ----------
        api_type : str
            Either 'dgl' or 'torch_geometric'.
        path : str
            Path for dataset storage or loading.
        """
        super().__init__(api_type, path)
        
        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        """
        Load the MUTAG dataset using DGL's MUTAGDataset class. 
        Currently an incomplete example (subgraph usage not fully implemented).
        """
        dataset = MUTAGDataset(raw_dir=self.path)
        self.graph = dataset[0]
        self.dataset_name = "mutag"

        self.node_number = self.graph.number_of_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['feat']
        self.labels = self.dataset.predict_category

        self.train_mask = self.graph.nodes[category].data['train_mask']
        self.val_mask = self.graph.nodes[category].data['train_mask']
        self.test_mask = self.graph.nodes[category].data['train_mask']


class PTC(Dataset):
    """
    A class for loading the PTC dataset (from GINDataset) in DGL format by default.
    """
    def __init__(self, api_type='dgl', path='./downloads/'):
        """
        Initialize PTC dataset.

        Parameters
        ----------
        api_type : str, optional
            Either 'dgl' or 'torch_geometric'. Default is 'dgl'.
        path : str, optional
            Path for dataset storage or loading. Default is './downloads/'.
        """
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        """
        Load the PTC dataset using GINDataset (DGL).
        Batches multiple graphs into one using dgl.batch.
        """
        dataset = GINDataset(name='PTC', raw_dir=self.path, self_loop=False)
        graph, _ =  zip(*[dataset[i] for i in range(16)])
        self.graph = dgl.batch(graph)

        self.dataset_name = "ptc"
        self.train_ratio = 0.8

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['attr'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['attr']
        self.labels = self.graph.ndata['label']

        self.generate_train_test_masks()


class NCI1(Dataset):
    """
    A class for loading the NCI1 dataset (from GINDataset) in DGL format by default.
    """
    def __init__(self, api_type='dgl', path='./downloads/'):
        """
        Initialize NCI1 dataset.

        Parameters
        ----------
        api_type : str, optional
            Either 'dgl' or 'torch_geometric'. Default is 'dgl'.
        path : str, optional
            Path for dataset storage or loading. Default is './downloads/'.
        """
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        """
        Load the NCI1 dataset using GINDataset (DGL).
        Batches multiple graphs into one using dgl.batch.
        """
        dataset = GINDataset(name='NCI1', raw_dir=self.path, self_loop=False)
        graph, _ = zip(*[dataset[i] for i in range(16)])
        self.graph = dgl.batch(graph)

        self.dataset_name = "nci1"
        self.train_ratio = 0.8

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['attr'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['attr']
        self.labels = self.graph.ndata['label']

        self.generate_train_test_masks()


####################################################################################################
class PROTEINS(Dataset):
    """
    A class for loading the PROTEINS dataset (from GINDataset) in DGL format by default.
    """
    def __init__(self, api_type='dgl', path='./downloads/'):
        """
        Initialize PROTEINS dataset.

        Parameters
        ----------
        api_type : str, optional
            Either 'dgl' or 'torch_geometric'. Default is 'dgl'.
        path : str, optional
            Path for dataset storage or loading. Default is './downloads/'.
        """
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        """
        Load the PROTEINS dataset using GINDataset (DGL).
        Batches multiple graphs into one using dgl.batch.
        """
        dataset = GINDataset(name='PROTEINS', raw_dir=self.path, self_loop=False)
        graph, _ = zip(*[dataset[i] for i in range(16)])
        self.graph = dgl.batch(graph)

        self.dataset_name = "proteins"
        self.train_ratio = 0.8

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['attr'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['attr']
        self.labels = self.graph.ndata['label']

        self.generate_train_test_masks()


####################################################################################################
class Collab(Dataset):
    """
    A class for loading the COLLAB dataset (from GINDataset) in DGL format by default.
    """
    def __init__(self, api_type='dgl', path='./downloads/'):
        """
        Initialize COLLAB dataset.

        Parameters
        ----------
        api_type : str, optional
            Either 'dgl' or 'torch_geometric'. Default is 'dgl'.
        path : str, optional
            Path for dataset storage or loading. Default is './downloads/'.
        """
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        """
        Load the COLLAB dataset using GINDataset (DGL).
        Batches multiple graphs into one using dgl.batch.
        """
        dataset = GINDataset(name='COLLAB', raw_dir=self.path, self_loop=False)
        graph, _ = zip(*[dataset[i] for i in range(16)])
        self.graph = dgl.batch(graph)

        self.dataset_name = "collab"
        self.train_ratio = 0.8

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['attr'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['attr']
        self.labels = self.graph.ndata['label']

        self.generate_train_test_masks()


class IMDB(Dataset):
    """
    A class for loading the IMDB-BINARY dataset (from GINDataset) in DGL format by default.
    """
    def __init__(self, api_type, path):
        """
        Initialize IMDB dataset.

        Parameters
        ----------
        api_type : str
            Either 'dgl' or 'torch_geometric'.
        path : str
            Path for dataset storage or loading.
        """
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        """
        Load the IMDB-BINARY dataset using GINDataset (DGL).
        Batches multiple graphs into one using dgl.batch.
        """
        dataset = GINDataset(name='IMDB-BINARY', raw_dir=self.path, self_loop=False)
        graph, _ = zip(*[dataset[i] for i in range(16)])
        self.graph = dgl.batch(graph)

        self.dataset_name = "imdb"
        self.train_ratio = 0.8

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['attr'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['attr']
        self.labels = self.graph.ndata['label']

        self.generate_train_test_masks()


####################################################################################################
class Computer(Dataset):
    """
    A class for loading the AmazonCoBuyComputer dataset in either DGL or PyG format.
    """
    def __init__(self, api_type='dgl', path='./downloads/'):
        """
        Initialize Computer dataset.

        Parameters
        ----------
        api_type : str, optional
            Either 'dgl' or 'torch_geometric'. Default is 'dgl'.
        path : str, optional
            Path for dataset storage or loading. Default is './downloads/'.
        """
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        """
        Load the AmazonCoBuyComputer dataset using DGL's AmazonCoBuyComputerDataset.
        Adds self-loops and randomly splits nodes into train/test using 'train_ratio'.
        """
        data = AmazonCoBuyComputerDataset(raw_dir=self.path)
        self.graph = dgl.add_self_loop(data[0])
        self.dataset_name = "computer"
        self.train_ratio = 0.8

        self.node_number = self.graph.number_of_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = torch.FloatTensor(self.graph.ndata['feat'])
        self.labels = torch.LongTensor(self.graph.ndata['label'])

        self.generate_train_test_masks()


class Photo(Dataset):
    """
    A class for loading the AmazonCoBuyPhoto dataset in either DGL or PyG format.
    """
    def __init__(self, api_type='dgl', path='./downloads/'):
        """
        Initialize Photo dataset.

        Parameters
        ----------
        api_type : str, optional
            Either 'dgl' or 'torch_geometric'. Default is 'dgl'.
        path : str, optional
            Path for dataset storage or loading. Default is './downloads/'.
        """
        super().__init__(api_type, path)
        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        """
        Load the AmazonCoBuyPhoto dataset using DGL's AmazonCoBuyPhotoDataset.
        Adds self-loops and randomly splits the data into train/test sets.
        """
        data = AmazonCoBuyPhotoDataset(raw_dir=self.path)
        self.graph = dgl.add_self_loop(data[0])
        self.dataset_name = "photo"
        self.train_ratio = 0.8

        self.node_number = self.graph.number_of_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = torch.FloatTensor(self.graph.ndata['feat'])
        self.labels = torch.LongTensor(self.graph.ndata['label'])

        self.generate_train_test_masks()
    

class YelpData(Dataset):
    """
    A class for loading the Yelp dataset in either DGL or PyG format.
    """
    def __init__(self, api_type='dgl', path='./downloads/'):
        """
        Initialize Yelp dataset.

        Parameters
        ----------
        api_type : str, optional
            Either 'dgl' or 'torch_geometric'. Default is 'dgl'.
        path : str, optional
            Path for dataset storage or loading. Default is './downloads/'.
        """
        super().__init__(api_type, path)

        if self.api_type == 'dgl':
            self.load_dgl_data()
        elif self.api_type == 'torch_geometric':
            self.load_tg_data()
        else:
            raise ValueError("Unsupported api_type.")

    def load_dgl_data(self):
        """
        Load the Yelp dataset using DGL's YelpDataset.
        Automatically provides train/val/test masks.
        """
        dataset = dgl.data.YelpDataset(raw_dir=self.path)
        self.graph = dataset[0]
        self.dataset_name = "yelp"

        self.node_number = self.graph.num_nodes()
        self.feature_number = self.graph.ndata['feat'].shape[1]
        self.label_number = len(torch.unique(self.graph.ndata['label']))

        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']

        self.train_mask = self.graph.ndata['train_mask']
        self.val_mask = self.graph.ndata['val_mask']
        self.test_mask = self.graph.ndata['test_mask']

    def load_tg_data(self):
        """
        Load the Yelp dataset using PyTorch Geometric's Yelp class.
        Also has built-in train/val/test masks.
        """
        dataset = Yelp(root=self.path)
        self.data = dataset[0]
        self.dataset_name = "yelp"

        self.node_number = self.data.num_nodes
        self.feature_number = self.data.x.shape[1]
        self.label_number = len(torch.unique(self.data.y))

        self.features = self.data.x
        self.labels = self.data.y

        self.train_mask = self.data.train_mask
        self.val_mask = self.data.val_mask
        self.test_mask = self.data.test_mask
