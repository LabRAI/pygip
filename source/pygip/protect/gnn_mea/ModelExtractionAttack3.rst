GNNMEA3
======================

Shadow model attack using subgraph indices to build partial graphs.

**Parameters**

dataset : object
    Dataset containing graph structures.
attack_node_fraction : float
    Fraction of nodes for attack.
model_path : str, optional
    Path to pre-trained model.

**Methods**

attack()
    Executes attack procedure:
    1. Loads subgraph indices
    2. Merges adjacency matrices
    3. Trains attack model on combined graph

1. **Attack 3 on Cora**
   
   - **NumNodes**: 2708  
   - **NumEdges**: 10556  
   - **NumFeats**: 1433  
   - **NumClasses**: 7  
   - **NumTrainingSamples**: 140  
   - **NumValidationSamples**: 500  
   - **NumTestSamples**: 1000  

   .. code-block:: console

      Enter dataset name (Cora, Citeseer, PubMed): Cora
      Enter attack type (0-5): 3
      Done loading data from cached files.
      =========Target Model Generating==========================
      100%|█████████████████| 200/200 [00:01<00:00, 168.47it/s]
      generated_train_mask 1977
        0%| | 0/300 [00:00<?, ?it/s]
      100%|█████████████████| 300/300 [00:02<00:00, 106.61it/s]

   - **Fidelity**: 0.7894  
   - **Accuracy**: 0.8155  
