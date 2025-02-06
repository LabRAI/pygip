GNNStealing
======================

A sophisticated model extraction attack framework for Graph Neural Networks that creates surrogate models by querying target models and reconstructing their behavior.

Description
-----------
GNNStealing implements an advanced model extraction attack against Graph Neural Networks. It supports multiple target and surrogate model architectures (GAT, GIN, GraphSAGE) and different feature recovery methods.

Parameters
----------
dataset : object
    Dataset object containing the graph structure and features.

attack_node_fraction : float
    Fraction of nodes to use in the attack process.

target_model_type : str
    Type of target model to attack. Options: ['gat', 'gin', 'sage']

surrogate_model_type : str
    Type of surrogate model to train. Options: ['gat', 'gin', 'sage']

recovery_method : str
    Method to recover features. Options: ['prediction', 'embedding', 'projection']

structure : str, optional
    Graph structure to use. Options: ['original', 'idgl']. Default: 'original'

delete_edges : str, optional
    Whether to delete edges from training graph. Options: ['yes', 'no']. Default: 'no'

transform : str, optional
    Transform method for embeddings. Default: 'TSNE'

device : str, optional
    Device to run computations on. Default: 'cuda' if available, else 'cpu'

Methods
-------
train_target_model_stl(self):
    """
    Trains the target model to be attacked.
    
    Returns:
        torch.nn.Module: Trained target model
    """

preprocess_dataset(self):
    """
    Preprocesses the dataset by:
    - Creating train/val/test masks if not present
    - Loading node features and labels
    - Validating data presence
    """

generate_query_graph(self):
    """
    Generates the query graph for the attack.
    
    Returns:
        dgl.DGLGraph: Query graph with self-loops added
    """

query_target_model(self, G_QUERY):
    """
    Queries the target model with the generated query graph.
    
    Parameters:
        G_QUERY (dgl.DGLGraph): Query graph
        
    Returns:
        tuple: (accuracy, predictions, embeddings)
    """

train_surrogate_model(self, data):
    """
    Trains the surrogate model using queried data.
    
    Parameters:
        data (tuple): Training data including features, labels and graph structure
        
    Returns:
        tuple: (model, classifier, detached_classifier)
    """

evaluate_surrogate_model(self, model_s, classifier, test_g):
    """
    Evaluates the trained surrogate model.
    
    Parameters:
        model_s: Surrogate model
        classifier: Model classifier
        test_g (dgl.DGLGraph): Test graph
        
    Returns:
        tuple: (accuracy, predictions, embeddings)
    """

attack(self):
    """
    Executes the complete attack procedure:
    1. Generates query graph
    2. Queries target model
    3. Trains surrogate model
    4. Evaluates attack success
    
    Prints attack results including:
    - Surrogate model accuracy
    - Target model accuracy
    - Attack accuracy
    - Model fidelity
    """

Notes
-----
- The attack supports three different model architectures: GAT, GIN, and GraphSAGE
- Features can be recovered using predictions, embeddings, or projections
- Graph structure can be original or reconstructed using IDGL
- Evaluation metrics include model accuracy and fidelity
- The implementation uses DGL (Deep Graph Library) for graph operations

Requirements
-----------
- torch
- dgl
- numpy
- sklearn
- scipy

.. code-block:: python
   :caption: Example Python Code
   :linenos:

   # Importing necessary classes and functions from the pygip library.
   from pygip.datasets.datasets import *  # Import all available datasets.
   from pygip.protect import *  # Import all core algorithms.

   # Loading the Cora dataset, which is commonly used in graph neural network research.
   dataset = Cora()

   # Initializing the GNNStealing attack with the Cora dataset.
   gnnstealing_attack = GNNStealing(
       dataset=dataset,
       attack_node_fraction=0.25,    # Fraction of nodes to use in the attack
       target_model_type='gin',      # Target model type (e.g., 'gin', 'gat', 'sage')
       surrogate_model_type='gin',   # Surrogate model type (e.g., 'gin', 'gat', 'sage')
       recovery_method='embedding',  # Recovery method ('embedding', 'projection', 'prediction')
       structure='original',         # Graph structure ('original' or other transformations)
       delete_edges='no',           # Whether to delete edges ('yes' or 'no')
       transform='TSNE',            # Dimensionality reduction method ('TSNE' or others)
       device='cpu'                 # Device to run the attack on ('cpu' or 'cuda')
   )

   # Executing the GNNStealing attack on the model.
   gnnstealing_attack.attack()

1. **GNNStealing Attack on Cora**
   
   - **NumNodes**: 2708  
   - **NumEdges**: 10556  
   - **NumFeats**: 1433  
   - **NumClasses**: 7  

   .. code-block:: console

      100%|██████████████████████| 200/200 [00:14<00:00, 14.03it/s]
      100%|██████████████████████| 100/100 [00:01<00:00, 51.57it/s]

   - **Fidelity**: 0.4989  
   - **Accuracy**: 0.8337  

2. **GNNStealing Attack on PROTEINS**
   
   - **NumNodes**: 939  
   - **NumEdges**: 651  
   - **NumFeats**: 3  
   - **NumClasses**: 2  

   .. code-block:: console

      100%|██████████████████████| 200/200 [00:02<00:00, 81.92it/s]
      100%|██████████████████████| 100/100 [00:00<00:00, 166.02it/s]

   - **Fidelity**: 0.9384  
   - **Accuracy**: 0.9288  
