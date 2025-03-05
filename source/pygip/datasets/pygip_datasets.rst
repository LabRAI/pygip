Datasets
==============

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:
   :titlesonly:

   Cora
   Citeseer
   Computer
   FacebookData
   FlickrData
   Photo
   PubMed
   RedditData
   Twitter
   WikiCS
   YelpData
   PROTEINS
   PTC
   PolblogsData
   NCI1
   IMDB
   MutaData
   DBLPdata
   COLLAB
   LastFMdata

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Class
     - Usage
   * - Dataset
     - Base class for all dataset implementations. Initialize with desired API type 
       ('dgl' or 'torch_geometric') and data path. Provides standard interface for 
       accessing node features, labels, and train/test masks.
   * - Cora
     - Load the Cora citation network. Access graph structure via graph attribute, 
       node features via features, and labels via labels. Training and testing 
       masks are automatically provided.
   * - Citeseer
     - Load the Citeseer publication network. Similar to Cora, provides access to graph 
       structure, features, and labels. Includes pre-defined train/test splits.
   * - PubMed
     - Load the PubMed biomedical publication network. Access node features representing 
       TF-IDF weighted word vectors and labels indicating paper types.
   * - FacebookData
     - Load the Facebook page-page network. Access the social network structure and page 
       features. Train/test splits are randomly generated with configurable ratio.
   * - FlickrData
     - Load the Flickr image network. Access image features and relationship graph. 
       Includes pre-defined train/validation/test splits.
   * - Computer
     - Load the Amazon computer co-purchase network. Access product features and 
       co-purchase relationships. Random train/test split is generated automatically.
   * - Photo
     - Load the Amazon photo co-purchase network. Similar to Computer dataset, provides 
       product features and relationships with automatic split generation.