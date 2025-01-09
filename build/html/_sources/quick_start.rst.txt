Quick Start Guide
===============
This guide will help you get started with PyGIP quickly.

Initial Setup
-----------

First, set up your environment:

.. code-block:: bash

    # Create and activate conda environment
    conda env create -f environment.yml -n pygip
    conda activate pygip

    # Install DGL manually (version 2.2.1 required)
    pip install dgl -f https://data.dgl.ai/wheels/repo.html

    # Under the GNNIP directory, set Python path
    export PYTHONPATH=`pwd`

Attack Examples
-------------

Model Extraction Attack
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Import required modules
    from pygip.datasets.datasets import *  # Import all available datasets
    from pygip.protect import *  # Import all core algorithms

    # Load the Cora dataset
    dataset = Cora()

    # Initialize model extraction attack with 25% of the data
    modelExtractionAttack = ModelExtractionAttack0(dataset, 0.25)

    # Execute the attack
    modelExtractionAttack.attack()

To run the attack example:

.. code-block:: bash

    python3 examples/examples.py

Defense Examples
--------------

Watermarking Defense
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Import required modules
    from pygip.protect.defense import Watermark_sage
    from pygip.protect import *

    # Initialize watermarking defense with Cora dataset
    model = Watermark_sage(Cora(), 0.25)

    # Apply watermark defense against model extraction attack
    # Parameters: dataset, attack_model_type (1=ModelExtractionAttack0), dataset_type (1=Cora)
    model.watermark_attack(Cora(), 1, 1)

To run the defense example:

.. code-block:: bash

    python3 examples/Watermarking_Graph_Neural_Networks_By_Random_Graphs.py

Next Steps
---------

For more detailed documentation, please refer to:

- :doc:`pygip_reference` - Complete API reference
- :doc:`pygip_protect_defense` - Detailed defense mechanisms
- :doc:`pygip_datasets` - Available datasets