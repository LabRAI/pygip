Model_Stealing_Attacks_Against_Inductive_Graph_Neural_Networks
==============================================================

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:
   :titlesonly:

   gin
   gat
   sage
   ginsurrogate
   gatsurrogate
   sagesurrogate

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Class
     - Usage
   * - GAT
     - Base Graph Attention Network implementation. Uses multi-head attention
        mechanisms to learn node representations by attending over node
        neighborhoods. Suitable for node classification tasks.
   * - GIN
     - Graph Isomorphism Network implementation. Leverages sum aggregator
        with learnable linear transformations to capture graph structure.
        Designed for graph-level and node-level tasks.
   * - SAGE
     - GraphSAGE implementation using mean aggregator. Learns node embeddings
        by sampling and aggregating features from local neighborhoods.
        Effective for large-scale graph learning.
   * - GAT_SURROGATE
     - GAT-based surrogate model for embedding learning. Extends GAT with
        an embedding output layer and dual training objectives (embedding
        matching and classification). Used for model extraction.
   * - GIN_SURROGATE
     - GIN-based surrogate model focused on embedding learning. Adapts GIN
        architecture with embedding output and combines MSE loss for embedding
        matching with classification loss.
   * - SAGE_SURROGATE
     - GraphSAGE-based surrogate model for embedding extraction. Modifies
        GraphSAGE with embedding output dimension and trains with both
        embedding reconstruction and classification objectives.

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
