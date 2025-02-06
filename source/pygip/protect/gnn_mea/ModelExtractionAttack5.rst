ModelExtractionAttack5
======================

Advanced shadow model attack with threshold-based edge linking.

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
    1. Builds block adjacency matrix
    2. Links edges using distance threshold
    3. Trains attack model

1. **Attack 5 on Cora**
   
   - **NumNodes**: 2708  
   - **NumEdges**: 10556  
   - **NumFeats**: 1433  
   - **NumClasses**: 7  
   - **NumTrainingSamples**: 140  
   - **NumValidationSamples**: 500  
   - **NumTestSamples**: 1000  

   .. code-block:: console

      Enter dataset name (Cora, Citeseer, PubMed): Cora
      Enter attack type (0-5): 5
      Done loading data from cached files.
      100%|█████████████████| 300/300 [00:02<00:00, 112.59it/s]

   - **Fidelity**: 0.0108  
   - **Accuracy**: 0.1543  

2. **Attack 5 on Citeseer**
   
   - **NumNodes**: 3327  
   - **NumEdges**: 9228  
   - **NumFeats**: 3703  
   - **NumClasses**: 6  
   - **NumTrainingSamples**: 120  
   - **NumValidationSamples**: 500  
   - **NumTestSamples**: 1000  

   .. code-block:: console

      Enter dataset name (Cora, Citeseer, PubMed): Citeseer
      Enter attack type (0-5): 5
      Done loading data from cached files.
      100%|█████████████████| 300/300 [00:05<00:00, 54.44it/s]

   - **Fidelity**: 0.2326  
   - **Accuracy**: 0.2069  
