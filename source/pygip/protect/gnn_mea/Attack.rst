Attack
=====================================================

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:
   :titlesonly:

   AdversarialModelExtraction
   Model_Stealing_Attacks_Against_Inductive_Graph_Neural_Networks
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
     - Source
   * - ModelExtractionAttack0
     - Model Extraction Attacks against Graph Neural Network
   * - ModelExtractionAttack1
     - Model Extraction Attacks against Graph Neural Network
   * - ModelExtractionAttack2
     - Model Extraction Attacks against Graph Neural Network
   * - ModelExtractionAttack3
     - Model Extraction Attacks against Graph Neural Network
   * - ModelExtractionAttack4
     - Model Extraction Attacks against Graph Neural Network
   * - ModelExtractionAttack5
     - Model Extraction Attacks against Graph Neural Network
   * - GNNStealing                    
     - Model Stealing Attacks Against Inductive Graph Neural Networks
   * - AdversarialModelExtraction    
     - Adversarial Model Extraction on Graph Neural Networks

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Class
     - Usage
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
   * - GNNStealing                    
     - Advanced model extraction attack that employs surrogate models (GAT/GIN/GraphSAGE) 
       to clone target GNN behavior. Supports multiple feature recovery methods including 
       predictions, embeddings and projections. Uses graph structure preservation and 
       validation through fidelity metrics.
   * - AdversarialModelExtraction    
     - Subgraph-based model extraction that samples k-hop neighborhoods and synthesizes 
       features using class-conditional prior distributions. Constructs attack graphs by 
       combining sampled subgraphs and trains a GCN-based surrogate model to replicate 
       target model behavior.