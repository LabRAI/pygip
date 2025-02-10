.. raw:: html

   <div style="margin-top: 50px; text-align: center;">
     <img src="_static/icon.png" alt="PyGIP Icon" style="width: 600px; height: auto;">
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
   :widths: 15 40 20 25

   * - Abbreviation
     - Description
     - Class Name
     - Reference

   * - ModelExtractionAttack
     - Implements several model extraction attack strategies including feature synthesis and shadow models.
     - ``ModelExtractionAttack0`` - ``ModelExtractionAttack5``
     - Wu, B., Yang, X., Pan, S., & Yuan, X. (2022). Model extraction attacks on graph neural networks: Taxonomy and realisation. In Proceedings of the 2022 ACM on Asia conference on computer and communications security, 337-350.

   * - GNNStealing
     - Implements model extraction attack creating surrogate models by querying target models and reconstructing their behavior.
     - ``GNNStealing``
     - Shen, Y., He, X., Han, Y., & Zhang, Y. (2022, May). Model stealing attacks against inductive graph neural networks. In 2022 IEEE Symposium on Security and Privacy (SP) (pp. 1175-1192). IEEE.

   * - GNN_Metric
     - Metrics for evaluating attack effectiveness.
     - ``GraphNeuralNetworkMetric``
     -



Defense Modules
---------------

.. list-table::
   :header-rows: 1
   :widths: 15 35 20 30

   * - Abbreviation
     - Description
     - Class Name
     - Reference

   * - Watermark
     - A graph watermarking defense mechanism, leveraging synthetic nodes and features.
     - ``WatermarkGraph``, ``Watermark_sage``
     - Zhao, X., Wu, H., & Zhang, X. (2021). Watermarking graph neural networks by random graphs. In 2021 9th International Symposium on Digital Forensics and Security (ISDFS), 1-6. IEEE.

   * - Graph_Merge
     - Tools to merge the original graph with synthetic watermarked nodes.
     - ``Defense``
     -


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
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   pygip/protect/gnn_mea/Attack
   pygip/protect/defense/pygip_protect_defense
   pygip/datasets/pygip_datasets

.. toctree::
   :maxdepth: 2
   :caption: Additional Information
   :hidden:

   Contribute
   Cite
   Core Team
   Reference


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

