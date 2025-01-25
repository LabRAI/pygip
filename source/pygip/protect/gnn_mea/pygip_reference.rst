pygip.protect.gnn_mea
=====================

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:
   :titlesonly:

   GraphNeuralNetworkMetric
   Gcn_Net
   Net_shadow
   Net_attack
   ModelExtractionAttack
   ModelExtractionAttack0
   ModelExtractionAttack1
   ModelExtractionAttack2
   ModelExtractionAttack3
   ModelExtractionAttack4
   ModelExtractionAttack5

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Class
     - Usage
   * - GraphNeuralNetworkMetric
     - Evaluate GNN model performance. Initialize with model, graph, and features to 
       compute both accuracy and fidelity metrics. Call evaluate() to update metrics.
   * - Gcn_Net
     - Basic GCN implementation with two layers. Initialize with feature dimensions 
       and number of classes. Used as the base model for both legitimate training 
       and attacks.
   * - Net_shadow
     - Shadow model for extraction attacks. Similar structure to Gcn_Net but 
       specifically designed for model extraction attacks.
   * - ModelExtractionAttack
     - Base class for all extraction attacks. Initialize with target dataset and 
       attack node fraction. Provides core attack utilities.
   * - ModelExtractionAttack0
     - Basic extraction attack using neighbor features. Synthesizes node features 
       based on first and second-order neighbors with configurable weighting.
   * - ModelExtractionAttack1
     - Query-based extraction using shadow graph. Reads selected nodes from file 
       and builds attack model based on queried labels.
   * - ModelExtractionAttack2
     - Structure-based extraction using identity features. Randomly samples nodes 
       and trains attack model with synthetic features.
   * - ModelExtractionAttack3
     - Shadow model attack using subgraph indices. Merges multiple partial graphs 
       for attack model training.
   * - ModelExtractionAttack4
     - Enhanced shadow attack with feature-based edge linking. Uses feature 
       similarity to establish connections in attack graph.
   * - ModelExtractionAttack5
     - Advanced shadow attack with threshold-based linking. Similar to Attack4 but 
       uses distance thresholds for edge creation.
