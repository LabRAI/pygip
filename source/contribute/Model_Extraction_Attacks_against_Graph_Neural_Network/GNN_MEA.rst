GNN_MEA
=====================================================

.. toctree::
   :maxdepth: 1
   :caption: ADDITIONAL INFORMATION
   :hidden:
   :titlesonly:

   ModelExtractionAttack
   GraphNeuralNetworkMetric
   Gcn_Net
   Net_shadow
   Net_attack

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Class
     - Usage
   * - ModelExtractionAttack
     - Base class for all extraction attacks. Initialize with target dataset and 
       attack node fraction. Provides core attack utilities.
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