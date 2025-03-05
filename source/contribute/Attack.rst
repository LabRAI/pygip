Attack
=====================================================

.. toctree::
   :maxdepth: 0
   :caption: ADDITIONAL INFORMATION
   :hidden:
   :titlesonly:

Adding New Methods
--------------------

1. Choose the Correct Category
~~~~~~~~~~~~~~~~~~~~~~~~~
* Place attack methods under ``models/attacks/``
* All methods should inherit from base classes in ``models/base/``

2. Implementation Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~
* Create a new file for your method (e.g., ``models/attacks/your_method.py``)
* Inherit from ``BaseAttack`` as appropriate
* Implement required interface methods:
    * For attacks: ``train()``, ``extract()``, ``evaluate()``

3. Code Integration
~~~~~~~~~~~~~~~~~~~~~
* Add your model class to ``__init__.py`` in the respective directory
* Add any new metrics to ``utils/metrics.py``
* Add any new neural network architectures to ``utils/models.py``
* If needed, extend dataset functionality in ``datasets/datasets.py``
* It would be better if you could package your code into one class and add to attack.py

Example Structure
-------------------

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