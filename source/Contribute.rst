How to Contribute
==================

PyGIP is designed to be an extensible framework for GNN Intellectual Property protection. Here's how you can contribute new methods:

Adding New Methods
----------------

1. Choose the Correct Category
~~~~~~~~~~~~~~~~~~~~~~~~~
* Place attack methods under ``models/attacks/``
* Place defense methods under ``models/defenses/``
* All methods should inherit from base classes in ``models/base/``

2. Implementation Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~
* Create a new file for your method (e.g., ``models/attacks/your_method.py``)
* Inherit from ``BaseAttack`` or ``BaseDefense`` as appropriate
* Implement required interface methods:
    * For attacks: ``train()``, ``extract()``, ``evaluate()``
    * For defenses: ``protect()``, ``verify()``

3. Code Integration
~~~~~~~~~~~~~~~
* Add your model class to ``__init__.py`` in the respective directory
* Add any new metrics to ``utils/metrics.py``
* Add any new neural network architectures to ``utils/models.py``
* If needed, extend dataset functionality in ``datasets/datasets.py``

Example Structure
--------------

.. code-block:: python

    # models/attacks/your_method.py
    from ..base.attack import BaseAttack

    class YourAttack(BaseAttack):
        def __init__(self, args):
            super().__init__()
            self.args = args
        
        def train(self, victim_model, training_data):
            # Implementation
            pass
        
        def extract(self, query_data):
            # Implementation
            pass

Example Implementations
--------------------

For detailed examples of implementing different methods, please refer to:

.. toctree::
   :maxdepth: 1
   
   contribute/Attack
   contribute/Defense