GNNMEA2
======================

Structure-based extraction using randomly sampled nodes and identity features.

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
    1. Randomly selects training nodes
    2. Creates identity features
    3. Trains Net_attack model

1. **Attack 2 on Cora**
   
   - **NumNodes**: 2708  
   - **NumEdges**: 10556  
   - **NumFeats**: 1433  
   - **NumClasses**: 7  
   - **NumTrainingSamples**: 140  
   - **NumValidationSamples**: 500  
   - **NumTestSamples**: 1000  

   .. code-block:: console

      Enter dataset name (Cora, Citeseer, PubMed): Cora
      Enter attack type (0-5): 2
      Done loading data from cached files.
      =========Target Model Generating==========================
      100%|█████████████████| 200/200 [00:01<00:00, 149.31it/s]
      100%|█████████████████| 200/200 [00:04<00:00, 46.31it/s]

   - **Fidelity**: 0.791  
   - **Accuracy**: 0.754  

2. **Attack 2 on Citeseer**
   
   - **NumNodes**: 3327  
   - **NumEdges**: 9228  
   - **NumFeats**: 3703  
   - **NumClasses**: 6  
   - **NumTrainingSamples**: 120  
   - **NumValidationSamples**: 500  
   - **NumTestSamples**: 1000  

   .. code-block:: console

      Enter dataset name (Cora, Citeseer, PubMed): Citeseer
      Enter attack type (0-5): 2
      Done loading data from cached files.
      =========Target Model Generating==========================
      100%|█████████████████| 200/200 [00:02<00:00, 69.09it/s]
      100%|█████████████████| 200/200 [00:05<00:00, 34.15it/s]

   - **Fidelity**: 0.618  
   - **Accuracy**: 0.521  

3. **Attack 2 on PubMed**
   
   - **NumNodes**: 19717  
   - **NumEdges**: 88651  
   - **NumFeats**: 500  
   - **NumClasses**: 3  
   - **NumTrainingSamples**: 60  
   - **NumValidationSamples**: 500  
   - **NumTestSamples**: 1000  

   .. code-block:: console

      Enter dataset name (Cora, Citeseer, PubMed): PubMed
      Enter attack type (0-5): 2
      Done loading data from cached files.
      =========Target Model Generating==========================
      100%|█████████████████| 200/200 [00:03<00:00, 51.98it/s]
      100%|█████████████████| 200/200 [03:21<00:00,  1.01s/it]

   - **Fidelity**: 0.910  
   - **Accuracy**: 0.782  