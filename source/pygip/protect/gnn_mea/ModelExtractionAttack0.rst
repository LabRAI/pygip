ModelExtractionAttack0
======================

Basic extraction attack that queries a subset of nodes and synthesizes features based on multi-hop neighbors.

**Parameters**

dataset : object
    Dataset containing graph structures.
attack_node_fraction : float
    Fraction of nodes to use in attack.
model_path : str, optional
    Path to load pre-trained model.
alpha : float, optional
    Weight between first-order and second-order neighbors (default: 0.8).

**Methods**

get_nonzero_indices(matrix_row: np.ndarray) -> np.ndarray
    Gets indices of nonzero entries in adjacency matrix row.

    Returns:
        np.ndarray: Indices of nonzero entries.

attack()
    Executes attack procedure:
    1. Samples subset of nodes for querying
    2. Synthesizes neighbor features 
    3. Builds sub-graph and trains extraction model

1. **Attack 0 on Cora**
   
   - **NumNodes**: 2708  
   - **NumEdges**: 10556  
   - **NumFeats**: 1433  
   - **NumClasses**: 7  
   - **NumTrainingSamples**: 140  
   - **NumValidationSamples**: 500  
   - **NumTestSamples**: 1000  

   .. code-block:: console

      Enter dataset name (Cora, Citeseer, PubMed): Cora
      Enter attack type (0-5): 0
      Done loading data from cached files.
      =========Target Model Generating==========================
      100%|█████████████████| 200/200 [00:01<00:00, 158.74it/s]
      =========Model Extracting==========================
      100%|█████████████████| 200/200 [00:02<00:00, 75.90it/s]
      ========================Final results:=========================================
      Fidelity: 0.8567, Accuracy: 0.7853

2. **Attack 0 on Citeseer**
   
   - **NumNodes**: 3327  
   - **NumEdges**: 9228  
   - **NumFeats**: 3703  
   - **NumClasses**: 6  
   - **NumTrainingSamples**: 120  
   - **NumValidationSamples**: 500  
   - **NumTestSamples**: 1000  

   .. code-block:: console

      Enter dataset name (Cora, Citeseer, PubMed): Citeseer
      Enter attack type (0-5): 0
      Done loading data from cached files.
      =========Target Model Generating==========================
      100%|█████████████████| 200/200 [00:02<00:00, 67.16it/s]
      =========Model Extracting==========================
      100%|█████████████████| 200/200 [00:06<00:00, 33.13it/s]
      ========================Final results:=========================================
      Fidelity: 0.7784, Accuracy: 0.6779

3. **Attack 0 on PubMed**
   
   - **NumNodes**: 19717  
   - **NumEdges**: 88651  
   - **NumFeats**: 500  
   - **NumClasses**: 3  
   - **NumTrainingSamples**: 60  
   - **NumValidationSamples**: 500  
   - **NumTestSamples**: 1000  

   .. code-block:: console

      Enter dataset name (Cora, Citeseer, PubMed): PubMed
      Enter attack type (0-5): 0
      Done loading data from cached files.
      =========Target Model Generating==========================
      100%|█████████████████| 200/200 [00:04<00:00, 49.73it/s]
      =========Model Extracting==========================
      100%|█████████████████| 200/200 [00:11<00:00, 17.35it/s]
      ========================Final results:=========================================
      Fidelity: 0.9077, Accuracy: 0.7790

4. **Attack 0 on DBLP**
   
   .. code-block:: console

      Enter dataset name (Cora, Citeseer, PubMed or more): DBLP
      Currently, only attack 0 is supported for this dataset.
      Enter attack type (0-5): 0
      =========Target Model Generating==========================
      100%|█████████████████| 200/200 [00:00<00:00, 281.01it/s]
      =========Model Extracting==========================
      100%|█████████████████| 200/200 [00:01<00:00, 135.48it/s]
      ========================Final results:=========================================
      Fidelity: 0.2764, Accuracy: 0.2948

5. **Attack 0 on Flickr (Stuck somewhere)**  
   
   .. code-block:: console

      Enter dataset name (Cora, Citeseer, PubMed or more): Flickr
      Currently, only attack 0 is supported for this dataset.
      Enter attack type (0-5): 0
      Downloading ./downloads/flickr.zip...
      Extracting file to ./downloads/flickr_b05c56ca
      =========Target Model Generating==========================
      100%|█████████████████| 200/200 [00:25<00:00, 7.91it/s]


