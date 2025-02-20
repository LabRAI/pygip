GNNMEA4
======================

Enhanced shadow model attack with feature-based edge linking.

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
    1. Reads node indices
    2. Links edges based on feature similarity
    3. Trains model on augmented graph

1. **Attack 4 on Cora**
   
   - **NumNodes**: 2708  
   - **NumEdges**: 10556  
   - **NumFeats**: 1433  
   - **NumClasses**: 7  
   - **NumTrainingSamples**: 140  
   - **NumValidationSamples**: 500  
   - **NumTestSamples**: 1000  

   .. code-block:: console

      Enter dataset name (Cora, Citeseer, PubMed): Cora
      Enter attack type (0-5): 4
      Done loading data from cached files.
      100%|█████████████████| 300/300 [00:02<00:00, 111.63it/s]

   - **Fidelity**: 0.1381  
   - **Accuracy**: 0.0758  

2. **Attack 4 on Citeseer**
   
   - **NumNodes**: 3327  
   - **NumEdges**: 9228  
   - **NumFeats**: 3703  
   - **NumClasses**: 6  
   - **NumTrainingSamples**: 120  
   - **NumValidationSamples**: 500  
   - **NumTestSamples**: 1000  

   .. code-block:: console

      Enter dataset name (Cora, Citeseer, PubMed): Citeseer
      Enter attack type (0-5): 4
      Done loading data from cached files.
      100%|█████████████████| 300/300 [00:05<00:00, 54.71it/s]

   - **Fidelity**: 0.2326  
   - **Accuracy**: 0.2069  
