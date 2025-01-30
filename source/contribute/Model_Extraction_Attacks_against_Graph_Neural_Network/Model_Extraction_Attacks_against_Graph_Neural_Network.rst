Model_Extraction_Attacks_against_Graph_Neural_Network
=====================================================

.. toctree::
   :maxdepth: 2
   :caption: ADDITIONAL INFORMATION
   :hidden:
   :titlesonly:

   GraphNeuralNetworkMetric
   Gcn_Net
   Net_shadow
   Net_attack

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