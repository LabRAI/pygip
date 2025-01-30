Defense
=======

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