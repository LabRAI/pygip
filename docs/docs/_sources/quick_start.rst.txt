Quick Start
=================
This guide will help you get started with PyHazards quickly using the hazard-first API.

Basic Usage
-----------

Toy Example (tabular classification)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    from pyhazards.datasets import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec
    from pyhazards.models import build_model
    from pyhazards.engine import Trainer
    from pyhazards.metrics import ClassificationMetrics

    class ToyHazard(Dataset):
        def _load(self):
            x = torch.randn(500, 16)
            y = torch.randint(0, 2, (500,))
            splits = {
                "train": DataSplit(x[:350], y[:350]),
                "val": DataSplit(x[350:425], y[350:425]),
                "test": DataSplit(x[425:], y[425:]),
            }
            return DataBundle(
                splits=splits,
                feature_spec=FeatureSpec(input_dim=16, description="toy features"),
                label_spec=LabelSpec(num_targets=2, task_type="classification"),
            )

    data = ToyHazard().load()
    model = build_model(name="mlp", task="classification", in_dim=16, out_dim=2)
    trainer = Trainer(model=model, metrics=[ClassificationMetrics()], mixed_precision=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    trainer.fit(data, optimizer=optimizer, loss_fn=loss_fn, max_epochs=5)
    results = trainer.evaluate(data, split="test")
    print(results)

GPU Support
-----------

PyHazards automatically detects CUDA availability. To explicitly set the device:

**Using Environment Variable:**

.. code-block:: bash

    export PYHAZARDS_DEVICE=cuda:0

**Using Python API:**

.. code-block:: python

    from pyhazards.utils import set_device

    # Set to use CUDA device 0
    set_device("cuda:0")

    # Or use CPU
    set_device("cpu")

Next Steps
----------

For more detailed documentation, please refer to:

- :doc:`pyhazards_datasets` - Dataset interface and registration
- :doc:`pyhazards_utils` - Utility functions and helpers
- :doc:`implementation` - Guide for implementing custom datasets and models
