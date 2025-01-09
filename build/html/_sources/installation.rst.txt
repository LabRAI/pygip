Installation
============

PyGIP requires Python 3.10+ and can be installed using pip. We recommend using a conda environment for installation.

Creating a Conda Environment
--------------------------

First, create and activate a new conda environment:

.. code-block:: bash

    conda create -n pygip python=3.10
    conda activate pygip
    conda env update -f environment.yml

Installing PyGIP
--------------

The installation command depends on your CUDA version.

For CUDA 11.x
~~~~~~~~~~~~

.. code-block:: bash

    pip install pygip -f https://data.dgl.ai/wheels/torch-2.3/cu118/repo.html --extra-index-url https://download.pytorch.org/whl/cu118

For CUDA 12.x
~~~~~~~~~~~~

.. code-block:: bash

    pip install pygip -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html --extra-index-url https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

Verifying Installation
--------------------

To verify that PyGIP is installed correctly, you can run:

.. code-block:: python

    import pygip
    print(pygip.__version__)

Requirements
-----------

- Python >= 3.10
- PyTorch >= 2.3
- CUDA 11.x or 12.x