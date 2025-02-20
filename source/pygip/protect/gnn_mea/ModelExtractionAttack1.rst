GNNMEA1
======================

Query-based extraction that reads selected nodes from file and builds shadow graph.

**Parameters**

dataset : object
    Dataset containing graph structures.
attack_node_fraction : float
    Fraction of nodes for attack.
selected_node_file : str
    File containing selected node IDs.
query_label_file : str
    File containing query labels.
shadow_graph_file : str, optional
    File describing shadow graph structure.

**Methods**

attack()
    Executes attack procedure:
    1. Reads nodes & labels from files
    2. Constructs shadow graph
    3. Trains extraction model

1. **Attack 1 on Cora**
   
   - **NumNodes**: 2708  
   - **NumEdges**: 10556  
   - **NumFeats**: 1433  
   - **NumClasses**: 7  
   - **NumTrainingSamples**: 140  
   - **NumValidationSamples**: 500  
   - **NumTestSamples**: 1000  

   .. code-block:: console

      Enter dataset name (Cora, Citeseer, PubMed): Cora
      Enter attack type (0-5): 1
      Done loading data from cached files.
      =========Target Model Generating==========================
      100%|█████████████████| 200/200 [00:01<00:00, 146.82it/s]
      Net_shadow(
        (layer1): GraphConv(in=1433, out=16, normalization=both, activation=None)
        (layer2): GraphConv(in=16, out=7, normalization=both, activation=None)
      )
      ===================Model Extracting================================
      100%|█████████████████| 200/200 [00:01<00:00, 100.62it/s]

   - **Fidelity**: 0.2176  
   - **Accuracy**: 0.3342  

2. **Attack 1 on Citeseer**
   
   - **NumNodes**: 3327  
   - **NumEdges**: 9228  
   - **NumFeats**: 3703  
   - **NumClasses**: 6  
   - **NumTrainingSamples**: 120  
   - **NumValidationSamples**: 500  
   - **NumTestSamples**: 1000  

   .. code-block:: console

      Enter dataset name (Cora, Citeseer, PubMed): Citeseer
      Enter attack type (0-5): 1
      Done loading data from cached files.
      =========Target Model Generating==========================
      100%|█████████████████| 200/200 [00:03<00:00, 66.00it/s]
      Net_shadow(
        (layer1): GraphConv(in=3703, out=16, normalization=both, activation=None)
        (layer2): GraphConv(in=16, out=6, normalization=both, activation=None)
      )
      ===================Model Extracting================================
      100%|█████████████████| 200/200 [00:04<00:00, 43.72it/s]

   - **Fidelity**: 0.6368  
   - **Accuracy**: 0.6597  