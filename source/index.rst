.. raw:: html

   <div style="margin-top: 50px; text-align: center;">
     <img src="_static/icon.png" alt="PyGIP Icon">
   </div>

----

**PyGIP** is a comprehensive Python library focused on model extraction attacks and defenses in Graph Neural Networks (GNNs). Built on PyTorch, PyTorch Geometric, and DGL, the library offers a robust framework for understanding, implementing, and defending against attacks targeting GNN models.

**PyGIP is featured for:**

- **Extensive Attack Implementations**: Multiple strategies for GNN model extraction attacks, including fidelity and accuracy evaluation.
- **Defensive Techniques**: Tools for creating robust defense mechanisms, such as watermarking graphs and inserting synthetic nodes.
- **Unified API**: Intuitive APIs for both attacks and defenses.
- **Integration with PyTorch/DGL**: Seamlessly integrates with PyTorch Geometric and DGL for scalable graph processing.
- **Customizable**: Supports user-defined attack and defense configurations.

**Quick Start Example:**

Outlier Detection Example with 5 Lines of Code:

.. code-block:: python

   from gnn_mae import ModelExtractionAttack2

   dataset = ...  # Load your graph dataset as a DGL object
   attack = ModelExtractionAttack2(dataset, attack_node_fraction=0.25)
   attack.attack()

   # Evaluate fidelity and accuracy
   print(f"Fidelity: {attack.fidelity}, Accuracy: {attack.accuracy}")

Attack Modules
---------------

.. list-table::
   :header-rows: 1

   * - Abbreviation
     - Description
     - Class Name
   * - ModelExtractionAttack
     - Implements several model extraction attack strategies including feature synthesis and shadow models.
     - `ModelExtractionAttack0` - `ModelExtractionAttack5`
   * - GNN_Metric
     - Metrics for evaluating attack effectiveness.
     - `GraphNeuralNetworkMetric`

Defense Modules
---------------

.. list-table::
   :header-rows: 1

   * - Abbreviation
     - Description
     - Class Name
   * - Watermark
     - A graph watermarking defense mechanism, leveraging synthetic nodes and features.
     - `WatermarkGraph`, `Watermark_sage`
   * - Graph_Merge
     - Tools to merge the original graph with synthetic watermarked nodes.
     - `Defense`

Additional Details
-------------------

For further information, refer to individual classes like `GraphNeuralNetworkMetric`, `WatermarkGraph`, or the specific model extraction attacks in the source code.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   quick_start

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   pygip_reference
   pygip_protect_defense
   pygip_datasets

.. toctree::
   :maxdepth: 2
   :caption: Additional Information
   :hidden:

   Cite
   Core Team
   Reference


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

