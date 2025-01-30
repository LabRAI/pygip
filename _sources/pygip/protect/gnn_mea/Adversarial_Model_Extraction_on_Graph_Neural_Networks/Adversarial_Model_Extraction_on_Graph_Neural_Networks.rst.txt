Adversarial_Model_Extraction_on_Graph_Neural_Networks
=====================================================

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:
   :titlesonly:

   AdversarialModelExtraction

.. code-block:: python
    :caption: Example Python Code
    :linenos:

    # Importing necessary classes and functions from the pygip library.
    from pygip.datasets.datasets import *  # Import all available datasets.
    from pygip.protect import *  # Import all core algorithms.

    # Loading the Cora dataset, which is commonly used in graph neural network research.
    dataset = Cora()

    # Initializing a model extraction attack with the Cora dataset.
    adversarial_attack = AdversarialModelExtraction(dataset, 0.25)

    # Executing the attack on the model.
    adversarial_attack.attack()