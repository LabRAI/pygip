WatermarkDefense
=================

Base class for defense operations against model extraction attacks in graph neural networks.

**Parameters**

dataset : graph_to_dataset
    Dataset wrapper containing the graph and its attributes.
attack_node_fraction : float
    Fraction of nodes to be used for attack purposes.

**Methods**

train(loader: DataLoader) -> float
    Trains the model for one epoch.
    
    Parameters:
        loader (DataLoader): Batch training data loader.
    
    Returns:
        float: Average loss for the epoch.

test(loader: DataLoader) -> float
    Evaluates the model.
    
    Parameters:
        loader (DataLoader): Test data loader.
    
    Returns:
        float: Test accuracy.

merge_cora_and_datawm(cora_graph: dgl.DGLGraph, datawm: dgl.DGLGraph) -> dgl.DGLGraph
    Merges original graph with watermark graph.
    
    Parameters:
        cora_graph (dgl.DGLGraph): Original graph.
        datawm (dgl.DGLGraph): Watermark graph.
    
    Returns:
        dgl.DGLGraph: Merged graph.

watermark_attack(dataset: graph_to_dataset, attack_name: int, dataset_name: int)
    Executes specified watermark attack.
    
    Parameters:
        dataset (graph_to_dataset): Target dataset.
        attack_name (int): ID of attack method.
        dataset_name (int): Dataset identifier.

.. code-block:: python
  :caption: Example Python Code
  :linenos:

  from pygip.protect import *
  from pygip.protect.Defense import Watermark_sage

  attack_name = int(input("""Please choose the number:\n1.ModelExtractionAttack0\n2.ModelExtractionAttack1\n
                          3.ModelExtractionAttack2\n4.ModelExtractionAttack3\n
                          5.ModelExtractionAttack4\n6.ModelExtractionAttack5\n"""))
  dataset_name = int(input("Please choose the number:\n1.Cora\n2.Citeseer\n3.PubMed\n"))
  if (dataset_name == 1):
      defense = Watermark_sage(Cora(),0.25)
      defense.watermark_attack(Cora(), attack_name, dataset_name)
  elif (dataset_name == 2):
      defense = Watermark_sage(Citeseer("dgl","./"),0.25)
      defense.watermark_attack(Citeseer("dgl","./"), attack_name, dataset_name)
  elif (dataset_name == 3):
      defense = Watermark_sage(PubMed("dgl","./"),0.25)
      defense.watermark_attack(PubMed("dgl","./"), attack_name, dataset_name)