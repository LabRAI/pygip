pygip.protect.defense
=====================

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:
   :titlesonly:

   graph_to_dataset
   WatermarkGraph
   GraphSAGE
   Defense
   Watermark_sage

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Class
     - Usage
   * - graph_to_dataset
     - Convert DGL graphs to standardized format. Initialize with graph and attack 
       parameters to create dataset wrapper with consistent interface.
   * - WatermarkGraph
     - Generate watermark graphs for model protection. Initialize with desired node 
       count and feature dimensions. Uses Erdős-Rényi model for edge generation.
   * - GraphSAGE
     - GraphSAGE model implementation for watermarking. Initialize with feature 
       dimensions and layer sizes. Used as base model in watermarking process.
   * - Defense
     - Base defense class against extraction attacks. Initialize with dataset and 
       attack parameters. Provides utilities for watermark injection and attack 
       handling.
   * - Watermark_sage
     - GraphSAGE-based watermarking implementation. Initialize with dataset and 
       watermark parameters. Performs two-stage training process: pre-training on 
       original graph and fine-tuning on watermark.