Model_Stealing_Attacks_Against_Inductive_Graph_Neural_Networks
===============================================================

.. toctree::
   :maxdepth: 2
   :caption: ADDITIONAL INFORMATION
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