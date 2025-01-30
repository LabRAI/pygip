Model_Extraction_Attacks_against_Graph_Neural_Network
=====================================================

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:
   :titlesonly:

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

.. code-block:: python
   :caption: Example Python Code
   :linenos: 
   
   # Importing necessary classes and functions from the pygip library.
   from pygip.datasets.datasets import *  # Import all available datasets.
   from pygip.protect import *  # Import all core algorithms.

   # Loading the Cora dataset, which is commonly used in graph neural network research.
   dataset = Cora()

   # Initializing a model extraction attack with the Cora dataset.
   # The second parameter (0.25) might represent the fraction of the data.
   modelExtractionAttack = ModelExtractionAttack0(dataset, 0.25)

   # Executing the attack on the model.
   modelExtractionAttack.attack()
