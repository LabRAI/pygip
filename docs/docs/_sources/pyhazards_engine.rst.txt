Engine
===================

Summary
-------

The engine wraps training/evaluation/prediction with sensible defaults and optional distributed support.

Core modules
------------

- ``pyhazards.engine.trainer`` — ``Trainer`` class with ``fit``, ``evaluate``, ``predict``, checkpoint save.
- ``pyhazards.engine.distributed`` — strategy selection and config helpers.
- ``pyhazards.engine.inference`` — sliding-window inference placeholder for large rasters/grids.

Typical usage
-------------

.. code-block:: python

    import torch
    from pyhazards.engine import Trainer
    from pyhazards.metrics import ClassificationMetrics
    from pyhazards.models import build_model

    model = build_model(name="mlp", task="classification", in_dim=16, out_dim=2)
    trainer = Trainer(model=model, metrics=[ClassificationMetrics()], mixed_precision=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    trainer.fit(data_bundle, optimizer=optimizer, loss_fn=loss_fn, max_epochs=10)
    results = trainer.evaluate(data_bundle, split="test")
    preds = trainer.predict(data_bundle, split="test")

Distributed and devices
-----------------------

- ``Trainer(strategy="auto")`` uses DDP when multiple GPUs are available; otherwise runs single-device.
- ``mixed_precision=True`` enables AMP when on CUDA.
- Device selection is handled via ``pyhazards.utils.hardware.auto_device`` by default.
