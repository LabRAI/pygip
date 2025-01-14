from pygip.datasets import *
from pygip.protect import gnn_mea
import networkx as nx
import numpy as np
import torch as th
import math
import random
from scipy.stats import wasserstein_distance
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import time
from dgl.nn.pytorch import GraphConv
from tqdm import tqdm
import inspect
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphNeuralNetworkMetric:
    """
    Graph Neural Network Metric Class.

    This class evaluates two metrics, fidelity and accuracy, for a given
    GNN model on a specified graph and features. The class also provides
    a utility function to help evaluate model predictions.

    Parameters
    ----------
    fidelity : float, optional
        Fidelity score. Default is 0.
    accuracy : float, optional
        Accuracy score. Default is 0.
    model : nn.Module, optional
        A PyTorch model. Default is None.
    graph : DGLGraph, optional
        A DGLGraph object of the data. Default is None.
    features : torch.Tensor, optional
        Node features. Default is None.
    mask : torch.Tensor, optional
        A boolean mask to select specific nodes. Default is None.
    labels : torch.Tensor, optional
        Ground truth labels for the nodes. Default is None.
    query_labels : torch.Tensor, optional
        Labels obtained from a queried model. Used to calculate fidelity.
        Default is None.

    Attributes
    ----------
    fidelity : float
        Fidelity metric after evaluation.
    accuracy : float
        Accuracy metric after evaluation.
    model : nn.Module or None
        GNN model to be evaluated.
    graph : DGLGraph or None
        Input graph structure.
    features : torch.Tensor or None
        Node features used for inference.
    mask : torch.Tensor or None
        Mask for selecting which nodes to evaluate.
    labels : torch.Tensor or None
        Ground-truth labels.
    query_labels : torch.Tensor or None
        Labels from the attacked (queried) model, used for fidelity.

    Methods
    -------
    evaluate_helper(model, graph, features, labels, mask)
        Computes prediction correctness given a model and data.
    evaluate()
        Updates the fidelity and accuracy metrics based on evaluation.
    __str__()
        Returns a string representation of the metrics.
    """

    def __init__(self, fidelity=0, accuracy=0, model=None,
                 graph=None, features=None, mask=None,
                 labels=None, query_labels=None):
        self.model = model.to(device) if model is not None else None
        self.graph = graph.to(device) if graph is not None else None
        self.features = features.to(device) if features is not None else None
        self.mask = mask.to(device) if mask is not None else None
        self.labels = labels.to(device) if labels is not None else None
        self.query_labels = query_labels.to(device) if query_labels is not None else None
        self.accuracy = accuracy
        self.fidelity = fidelity

    def evaluate_helper(self, model, graph, features, labels, mask):
        """
        Helper function to evaluate the model's performance on the specified data.

        Parameters
        ----------
        model : nn.Module
            The model to be evaluated.
        graph : DGLGraph
            The graph to run inference on.
        features : torch.Tensor
            The features corresponding to the graph's nodes.
        labels : torch.Tensor
            Ground truth labels for the nodes.
        mask : torch.Tensor
            A boolean mask selecting which nodes to evaluate on.

        Returns
        -------
        float or None
            The evaluation accuracy over the masked nodes, or None if any input
            is missing.
        """
        if model is None or graph is None or features is None or labels is None or mask is None:
            return None
        model.eval()
        with th.no_grad():
            logits = model(graph, features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = th.max(logits, dim=1)
            correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

    def evaluate(self):
        """
        Main function to update fidelity and accuracy scores.

        This function sets `self.accuracy` and `self.fidelity` by
        evaluating `self.model` on `(self.graph, self.features, self.labels, self.mask)`
        and `(self.graph, self.features, self.query_labels, self.mask)`.
        """
        self.accuracy = self.evaluate_helper(
            self.model, self.graph, self.features, self.labels, self.mask)
        self.fidelity = self.evaluate_helper(
            self.model, self.graph, self.features, self.query_labels, self.mask)

    def __str__(self):
        """
        Returns a string representation of the Metric instance,
        showing fidelity and accuracy.

        Returns
        -------
        str
            A string describing the fidelity and accuracy metrics.
        """
        return f"Fidelity: {self.fidelity}, Accuracy: {self.accuracy}"


class Gcn_Net(nn.Module):
    """
    A simple GCN Network.

    This network uses two GraphConv layers. The first layer is of size
    [feature_number, 16], and the second layer outputs label_number
    classes.

    Parameters
    ----------
    feature_number : int
        The size of the input feature dimension.
    label_number : int
        The number of classes for prediction.

    Attributes
    ----------
    layers : nn.ModuleList
        A list containing two GraphConv layers.
    dropout : nn.Dropout
        Dropout layer with p=0.5.

    Methods
    -------
    forward(g, features)
        Forward pass through the network.
    """

    def __init__(self, feature_number, label_number):
        super(Gcn_Net, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(feature_number, 16, activation=F.relu))
        # output layer
        self.layers.append(GraphConv(16, label_number))
        self.dropout = nn.Dropout(p=0.5)
        self.to(device)

    def forward(self, g, features):
        """
        Forward computation of the GCN.

        Parameters
        ----------
        g : DGLGraph
            The input graph.
        features : torch.Tensor
            Node features for each node in the graph.

        Returns
        -------
        torch.Tensor
            Logits (un-normalized predictions) for each node.
        """
        g = g.to(device)
        features = features.to(device)
        x = F.relu(self.layers[0](g, features))
        x = self.layers[1](g, x)
        return x


class Net_shadow(th.nn.Module):
    """
    A shadow model GCN, used in model extraction or membership inference.

    Similar to Gcn_Net but structured with two GraphConv layers.

    Parameters
    ----------
    feature_number : int
        The size of the input feature dimension.
    label_number : int
        The number of classes for prediction.

    Attributes
    ----------
    layer1 : GraphConv
        First GraphConv layer.
    layer2 : GraphConv
        Second GraphConv layer.

    Methods
    -------
    forward(g, features)
        Forward pass of the shadow model.
    """

    def __init__(self, feature_number, label_number):
        super(Net_shadow, self).__init__()
        self.layer1 = GraphConv(feature_number, 16)
        self.layer2 = GraphConv(16, label_number)

    def forward(self, g, features):
        """
        Forward computation of the shadow GCN.

        Parameters
        ----------
        g : DGLGraph
            The input graph.
        features : torch.Tensor
            Node features for each node.

        Returns
        -------
        torch.Tensor
            Logits for each node of the graph.
        """
        x = th.nn.functional.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x


class Net_attack(nn.Module):
    """
    An attack model GCN used for model extraction.

    This network uses two GraphConv layers, similar to `Gcn_Net`,
    potentially with different hyperparameters or structure to perform
    the extraction process.

    Parameters
    ----------
    feature_number : int
        The size of the input feature dimension.
    label_number : int
        The number of classes for prediction.

    Attributes
    ----------
    layers : nn.ModuleList
        A list containing two GraphConv layers.
    dropout : nn.Dropout
        Dropout layer with p=0.5.

    Methods
    -------
    forward(g, features)
        Forward pass through the attack model.
    """

    def __init__(self, feature_number, label_number):
        super(Net_attack, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(feature_number, 16, activation=F.relu))
        # output layer
        self.layers.append(GraphConv(16, label_number))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, g, features):
        """
        Forward computation of the attack GCN.

        Parameters
        ----------
        g : DGLGraph
            The input graph.
        features : torch.Tensor
            Node features for each node.

        Returns
        -------
        torch.Tensor
            Logits (un-normalized predictions) for each node.
        """
        x = F.relu(self.layers[0](g, features))
        x = self.layers[1](g, x)
        return x


def evaluate(model, g, features, labels, mask):
    """
    Evaluate function to compute accuracy of a model.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    g : DGLGraph
        The graph structure.
    features : torch.Tensor
        The node features.
    labels : torch.Tensor
        Ground truth labels of the nodes.
    mask : torch.Tensor
        Boolean mask selecting which nodes to evaluate.

    Returns
    -------
    float
        The accuracy of the model on the specified mask.
    """
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


class ModelExtractionAttack:
    """
    The base class for Model Extraction Attacks.

    This class either trains or loads a target GCN model (`net1`) and
    sets up basic data structures. Subclasses implement specific
    extraction strategies.

    Parameters
    ----------
    dataset : object
        A dataset object containing graph, features, labels, masks, etc.
    attack_node_fraction : float
        The fraction of nodes to use in the attack.
    model_path : str, optional
        If provided, will load a pre-trained target model from the given path.

    Attributes
    ----------
    dataset : object
        The dataset object used in the attack.
    graph : DGLGraph
        The DGL graph.
    node_number : int
        Total number of nodes in the graph.
    feature_number : int
        Number of features per node.
    label_number : int
        Number of label classes.
    attack_node_number : int
        Number of nodes used for the attack.
    attack_node_fraction : float
        Fraction of nodes used for the attack.
    features : torch.Tensor
        Node features.
    labels : torch.Tensor
        Node labels.
    train_mask : torch.Tensor
        Training mask for the nodes.
    test_mask : torch.Tensor
        Test mask for the nodes.
    net1 : nn.Module
        The target GCN model to be extracted.

    Methods
    -------
    train_target_model()
        Trains the target GCN model if no `model_path` is provided.
    """

    def __init__(self, dataset, attack_node_fraction, model_path=None):
        self.dataset = dataset
        # graph
        self.graph = dataset.graph.to(device)

        # node_number, feature_number, label_number, attack_node_number
        self.node_number = dataset.node_number
        self.feature_number = dataset.feature_number
        self.label_number = dataset.label_number
        self.attack_node_number = int(dataset.node_number * attack_node_fraction)
        self.attack_node_fraction = attack_node_fraction

        # features, labels
        self.features = dataset.features.to(device)
        self.labels = dataset.labels.to(device)

        # train_mask, test_mask
        self.train_mask = dataset.train_mask.to(device)
        self.test_mask = dataset.test_mask.to(device)

        if model_path is None:
            self.train_target_model()
        else:
            self.net1 = Gcn_Net(self.feature_number, self.label_number)
            optimizer_b = th.optim.Adam(self.net1.parameters(), lr=1e-2, weight_decay=5e-4)
            self.net1.load_state_dict(th.load(model_path))
            self.net1.to(device)

    def train_target_model(self):
        """
        Train the target GCN model (net1) from scratch.

        Uses a simple GCN (two-layer GraphConv). Runs for 200 epochs.
        """
        focus_graph = self.graph
        degs = focus_graph.in_degrees().float()
        norm = th.pow(degs, -0.5)
        norm[th.isinf(norm)] = 0
        norm = norm.to(device)
        focus_graph.ndata['norm'] = norm.unsqueeze(1)

        self.net1 = Gcn_Net(self.feature_number, self.label_number).to(device)
        optimizer = th.optim.Adam(self.net1.parameters(), lr=1e-2, weight_decay=5e-4)
        dur = []

        print("=========Target Model Generating==========================")
        for epoch in tqdm(range(200)):
            if epoch >= 3:
                t0 = time.time()

            self.net1.train()
            logits = self.net1(focus_graph, self.features)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp[self.train_mask], self.labels[self.train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)


class ModelExtractionAttack0(ModelExtractionAttack):
    """
    ModelExtractionAttack0.

    A specific extraction strategy that queries a subset of nodes and
    synthesizes features for other nodes based on multi-hop neighbors.

    Inherits from ModelExtractionAttack.

    Parameters
    ----------
    dataset : object
        The dataset containing the graph, features, etc.
    attack_node_fraction : float
        The fraction of nodes used for the attack.
    model_path : str, optional
        Path to load the target model. If None, trains from scratch.
    alpha : float, optional
        Weight controlling how much emphasis is placed on first-order vs
        second-order neighbors when synthesizing features.
    """

    def __init__(self, dataset, attack_node_fraction, model_path=None, alpha=0.8):
        super().__init__(dataset, attack_node_fraction, model_path)
        self.alpha = alpha

    def get_nonzero_indices(self, matrix_row):
        """
        Helper function to get indices of nonzero entries in a row.

        Parameters
        ----------
        matrix_row : np.ndarray
            A single row of the adjacency matrix.

        Returns
        -------
        np.ndarray
            Indices of nonzero entries in the adjacency matrix row.
        """
        return np.where(matrix_row != 0)[0]

    def attack(self):
        """
        Main attack procedure.

        1. Samples a subset of nodes (`sub_graph_node_index`) for querying.
        2. Synthesizes features for neighboring nodes and their neighbors.
        3. Builds a sub-graph, trains a new GCN on it, and evaluates
           fidelity & accuracy w.r.t. the target model.
        """
        try:
            torch.cuda.empty_cache()
            g = self.graph.clone().to(device)
            g_matrix = g.adjacency_matrix().to_dense().cpu().numpy()
            del g

            sub_graph_node_index = np.random.choice(
                self.node_number, self.attack_node_number, replace=False).tolist()

            batch_size = 32
            features_query = self.features.clone()

            syn_nodes = []
            for node_index in sub_graph_node_index:
                one_step_node_index = self.get_nonzero_indices(g_matrix[node_index]).tolist()
                syn_nodes.extend(one_step_node_index)

                for first_order_node_index in one_step_node_index:
                    two_step_node_index = self.get_nonzero_indices(g_matrix[first_order_node_index]).tolist()
                    syn_nodes.extend(two_step_node_index)

            sub_graph_syn_node_index = list(set(syn_nodes) - set(sub_graph_node_index))
            total_sub_nodes = list(set(sub_graph_syn_node_index + sub_graph_node_index))

            # Process synthetic nodes in batches
            for i in range(0, len(sub_graph_syn_node_index), batch_size):
                batch_indices = sub_graph_syn_node_index[i:i + batch_size]

                for node_index in batch_indices:
                    features_query[node_index] = 0
                    one_step_node_index = self.get_nonzero_indices(g_matrix[node_index]).tolist()
                    one_step_node_index = list(set(one_step_node_index).intersection(set(sub_graph_node_index)))

                    num_one_step = len(one_step_node_index)
                    if num_one_step > 0:
                        for first_order_node_index in one_step_node_index:
                            this_node_degree = len(self.get_nonzero_indices(g_matrix[first_order_node_index]))
                            features_query[node_index] += (
                                self.features[first_order_node_index] * self.alpha /
                                torch.sqrt(torch.tensor(num_one_step * this_node_degree, device=device))
                            )

                    two_step_nodes = []
                    for first_order_node_index in one_step_node_index:
                        two_step_nodes.extend(self.get_nonzero_indices(g_matrix[first_order_node_index]).tolist())

                    total_two_step_node_index = list(set(two_step_nodes) - set(one_step_node_index))
                    total_two_step_node_index = list(set(total_two_step_node_index).intersection(set(sub_graph_node_index)))

                    num_two_step = len(total_two_step_node_index)
                    if num_two_step > 0:
                        for second_order_node_index in total_two_step_node_index:
                            this_node_first_step_nodes = self.get_nonzero_indices(g_matrix[second_order_node_index]).tolist()
                            this_node_second_step_nodes = set()

                            for nodes_in_this_node in this_node_first_step_nodes:
                                this_node_second_step_nodes.update(
                                    self.get_nonzero_indices(g_matrix[nodes_in_this_node]).tolist())

                            this_node_second_step_nodes = this_node_second_step_nodes - set(this_node_first_step_nodes)
                            this_node_second_degree = len(this_node_second_step_nodes)

                            if this_node_second_degree > 0:
                                features_query[node_index] += (
                                    self.features[second_order_node_index] * (1 - self.alpha) /
                                    torch.sqrt(torch.tensor(num_two_step * this_node_second_degree, device=device))
                                )

                torch.cuda.empty_cache()

            # Update masks
            for i in range(self.node_number):
                if i in sub_graph_node_index:
                    self.test_mask[i] = 0
                    self.train_mask[i] = 1
                elif i in sub_graph_syn_node_index:
                    self.test_mask[i] = 1
                    self.train_mask[i] = 0
                else:
                    self.test_mask[i] = 1
                    self.train_mask[i] = 0

            # Create subgraph adjacency matrix
            sub_g = np.zeros((len(total_sub_nodes), len(total_sub_nodes)))
            for sub_index in range(len(total_sub_nodes)):
                sub_g[sub_index] = g_matrix[total_sub_nodes[sub_index], total_sub_nodes]

            del g_matrix

            sub_train_mask = self.train_mask[total_sub_nodes]
            sub_features = features_query[total_sub_nodes]
            sub_labels = self.labels[total_sub_nodes]

            # Get query labels
            self.net1.eval()
            with torch.no_grad():
                g = self.graph.to(device)
                logits_query = self.net1(g, features_query)
                _, labels_query = torch.max(logits_query, dim=1)
                sub_labels_query = labels_query[total_sub_nodes]
                del logits_query

            # Create DGL graph
            sub_g = nx.from_numpy_array(sub_g)
            sub_g.remove_edges_from(nx.selfloop_edges(sub_g))
            sub_g.add_edges_from(zip(sub_g.nodes(), sub_g.nodes()))
            sub_g = DGLGraph(sub_g)
            sub_g = sub_g.to(device)

            degs = sub_g.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.to(device)
            sub_g.ndata['norm'] = norm.unsqueeze(1)

            # Train extraction model
            net = Gcn_Net(self.feature_number, self.label_number).to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=5e-4)
            best_performance_metrics = GraphNeuralNetworkMetric()

            print("=========Model Extracting==========================")
            for epoch in tqdm(range(200)):
                net.train()
                logits = net(sub_g, sub_features)
                logp = F.log_softmax(logits, dim=1)
                loss = F.nll_loss(logp[sub_train_mask], sub_labels_query[sub_train_mask])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    focus_gnn_metrics = GraphNeuralNetworkMetric(
                        0, 0, net, g, self.features, self.test_mask, self.labels, labels_query
                    )
                    focus_gnn_metrics.evaluate()

                    best_performance_metrics.fidelity = max(
                        best_performance_metrics.fidelity, focus_gnn_metrics.fidelity)
                    best_performance_metrics.accuracy = max(
                        best_performance_metrics.accuracy, focus_gnn_metrics.accuracy)

                if epoch % 10 == 0:
                    torch.cuda.empty_cache()

            print("========================Final results:=========================================")
            print(best_performance_metrics)

        except RuntimeError as e:
            print(f"Runtime error: {e}")
            torch.cuda.empty_cache()
            raise


class ModelExtractionAttack1(ModelExtractionAttack):
    """
    ModelExtractionAttack1.

    A variant of extraction attack that reads selected nodes from a file
    and constructs a shadow graph from another file.

    Parameters
    ----------
    dataset : object
        The dataset containing the graph, features, labels, etc.
    attack_node_fraction : float
        Fraction of nodes used for the attack.
    selected_node_file : str
        Path to a file containing the selected node IDs used for extraction.
    query_label_file : str
        Path to a file containing the query labels (node ID + label).
    shadow_graph_file : str, optional
        Path to a file describing the adjacency matrix of the shadow graph.

    Inherits
    --------
    ModelExtractionAttack
    """

    def __init__(self, dataset, attack_node_fraction, selected_node_file,
                 query_label_file, shadow_graph_file=None):
        super().__init__(dataset, attack_node_fraction)
        self.attack_node_number = 700
        self.selected_node_file = selected_node_file
        self.query_label_file = query_label_file
        self.shadow_graph_file = shadow_graph_file

    def attack(self):
        """
        Main attack procedure.

        1. Reads selected nodes from file for training (attack) nodes.
        2. Reads query labels from another file.
        3. Builds a shadow graph from the given adjacency matrix file.
        4. Trains a shadow model on the selected nodes, then evaluates
           fidelity & accuracy against the original target graph.
        """
        try:
            torch.cuda.empty_cache()

            with open(self.selected_node_file, "r") as selected_node_file:
                attack_nodes = [int(line.strip()) for line in selected_node_file]

            # Identify the test nodes
            testing_nodes = [i for i in range(self.node_number) if i not in attack_nodes]

            attack_features = self.features[attack_nodes]

            # Update masks
            for i in range(self.node_number):
                if i in attack_nodes:
                    self.test_mask[i] = 0
                    self.train_mask[i] = 1
                else:
                    self.test_mask[i] = 1
                    self.train_mask[i] = 0

            sub_test_mask = self.test_mask

            with open(self.query_label_file, "r") as query_label_file:
                lines = query_label_file.readlines()
                all_query_labels = []
                attack_query = []
                for line in lines:
                    node_id, label = map(int, line.split())
                    all_query_labels.append(label)
                    if node_id in attack_nodes:
                        attack_query.append(label)

            attack_query = torch.LongTensor(attack_query).to(device)
            all_query_labels = torch.LongTensor(all_query_labels).to(device)

            with open(self.shadow_graph_file, "r") as shadow_graph_file:
                lines = shadow_graph_file.readlines()
                adj_matrix = np.zeros((self.attack_node_number, self.attack_node_number))
                for line in lines:
                    src, dst = map(int, line.split())
                    adj_matrix[src][dst] = 1
                    adj_matrix[dst][src] = 1

            g_shadow = np.asmatrix(adj_matrix)
            sub_g = nx.from_numpy_array(g_shadow)

            sub_g.remove_edges_from(nx.selfloop_edges(sub_g))
            sub_g.add_edges_from(zip(sub_g.nodes(), sub_g.nodes()))
            sub_g = DGLGraph(sub_g)
            sub_g = sub_g.to(device)

            degs = sub_g.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.to(device)
            sub_g.ndata['norm'] = norm.unsqueeze(1)

            # Create target graph
            adj_matrix = self.graph.adjacency_matrix().to_dense().cpu().numpy()
            sub_g_b = nx.from_numpy_array(adj_matrix)

            sub_g_b.remove_edges_from(nx.selfloop_edges(sub_g_b))
            sub_g_b.add_edges_from(zip(sub_g_b.nodes(), sub_g_b.nodes()))
            sub_g_b = DGLGraph(sub_g_b)
            sub_g_b = sub_g_b.to(device)

            degs = sub_g_b.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.to(device)
            sub_g_b.ndata['norm'] = norm.unsqueeze(1)

            net = Net_shadow(self.feature_number, self.label_number).to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=5e-4)
            dur = []
            best_performance_metrics = GraphNeuralNetworkMetric()

            print("===================Model Extracting================================")
            for epoch in tqdm(range(200)):
                if epoch >= 3:
                    t0 = time.time()

                net.train()
                logits = net(sub_g, attack_features)
                logp = F.log_softmax(logits, dim=1)
                loss = F.nll_loss(logp, attack_query)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch >= 3:
                    dur.append(time.time() - t0)

                with torch.no_grad():
                    focus_gnn_metrics = GraphNeuralNetworkMetric(
                        0, 0, net, sub_g_b, self.features, self.test_mask,
                        all_query_labels, self.labels
                    )
                    focus_gnn_metrics.evaluate()

                    best_performance_metrics.fidelity = max(
                        best_performance_metrics.fidelity, focus_gnn_metrics.fidelity)
                    best_performance_metrics.accuracy = max(
                        best_performance_metrics.accuracy, focus_gnn_metrics.accuracy)

                if epoch % 10 == 0:
                    torch.cuda.empty_cache()

            print(best_performance_metrics)

        except RuntimeError as e:
            print(f"Runtime error: {e}")
            torch.cuda.empty_cache()
            raise


class ModelExtractionAttack2(ModelExtractionAttack):
    """
    ModelExtractionAttack2.

    A strategy that randomly samples a fraction of nodes as attack nodes,
    synthesizes identity features for all nodes, then trains an extraction
    model. The leftover nodes become test nodes.

    Inherits
    --------
    ModelExtractionAttack
    """

    def __init__(self, dataset, attack_node_fraction, model_path=None):
        super().__init__(dataset, attack_node_fraction, model_path)

    def attack(self):
        """
        Main attack procedure.

        1. Randomly select `attack_node_number` nodes as training nodes.
        2. Set up synthetic features as identity vectors for all nodes.
        3. Train a `Net_attack` model on these nodes with the queried labels.
        4. Evaluate fidelity & accuracy on a subset of leftover nodes.
        """
        try:
            torch.cuda.empty_cache()

            attack_nodes = []
            for i in range(self.attack_node_number):
                candidate_node = random.randint(0, self.node_number - 1)
                if candidate_node not in attack_nodes:
                    attack_nodes.append(candidate_node)

            test_num = 0
            for i in range(self.node_number):
                if i in attack_nodes:
                    self.test_mask[i] = 0
                    self.train_mask[i] = 1
                else:
                    if test_num < 1000:
                        self.test_mask[i] = 1
                        self.train_mask[i] = 0
                        test_num += 1
                    else:
                        self.test_mask[i] = 0
                        self.train_mask[i] = 0

            self.net1.eval()
            with torch.no_grad():
                logits_query = self.net1(self.graph, self.features)
                _, labels_query = torch.max(logits_query, dim=1)

            syn_features_np = np.eye(self.node_number)
            syn_features = torch.FloatTensor(syn_features_np).to(device)
            g = self.graph.to(device)

            degs = g.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.to(device)
            g.ndata['norm'] = norm.unsqueeze(1)

            net_attack = Net_attack(self.node_number, self.label_number).to(device)
            optimizer_original = torch.optim.Adam(net_attack.parameters(), lr=5e-2, weight_decay=5e-4)

            dur = []
            best_performance_metrics = GraphNeuralNetworkMetric()

            print("=========Model Extracting==========================")
            for epoch in tqdm(range(200)):
                if epoch >= 3:
                    t0 = time.time()

                net_attack.train()
                logits = net_attack(g, syn_features)
                logp = F.log_softmax(logits, 1)
                loss = F.nll_loss(logp[self.train_mask.to(device)], labels_query[self.train_mask].to(device))

                optimizer_original.zero_grad()
                loss.backward()
                optimizer_original.step()

                if epoch >= 3:
                    dur.append(time.time() - t0)

                with torch.no_grad():
                    focus_gnn_metrics = GraphNeuralNetworkMetric(
                        0, 0, net_attack, g, syn_features,
                        self.test_mask.to(device),
                        self.labels.to(device),
                        labels_query.to(device)
                    )
                    focus_gnn_metrics.evaluate()

                    best_performance_metrics.fidelity = max(
                        best_performance_metrics.fidelity, focus_gnn_metrics.fidelity)
                    best_performance_metrics.accuracy = max(
                        best_performance_metrics.accuracy, focus_gnn_metrics.accuracy)

                if epoch % 10 == 0:
                    torch.cuda.empty_cache()

            print("========================Final results:=========================================")
            print(best_performance_metrics)

        except RuntimeError as e:
            print(f"Runtime error: {e}")
            torch.cuda.empty_cache()
            raise


class ModelExtractionAttack3(ModelExtractionAttack):
    """
    ModelExtractionAttack3.

    A more complex extraction strategy that uses a "shadow graph index"
    file to build partial subgraphs and merges them. It queries selected
    nodes from a potential set and forms a combined adjacency matrix.

    Inherits
    --------
    ModelExtractionAttack
    """

    def __init__(self, dataset, attack_node_fraction, model_path=None):
        super().__init__(dataset, attack_node_fraction, model_path)

    def attack(self):
        """
        Main attack procedure.

        Steps:
        1. Loads indices for two subgraphs from text files.
        2. Selects `attack_node_number` nodes from the first subgraph index.
        3. Merges subgraph adjacency matrices and constructs a new graph
           with combined features.
        4. Trains a new GCN and evaluates fidelity & accuracy w.r.t. the
           original target.
        """
        try:
            torch.cuda.empty_cache()
            g_numpy = self.graph.adjacency_matrix().to_dense().cpu().numpy()

            defense_path = inspect.getfile(gnn_mea)

            sub_graph_index_b = []
            with open(os.path.abspath(os.path.join(
                defense_path,
                '../../../pygip/data/attack3_shadow_graph/' + self.dataset.dataset_name +
                '/attack_6_sub_shadow_graph_index_attack_2.txt')), 'r') as fileObject:
                for ip in fileObject:
                    sub_graph_index_b.append(int(ip))

            sub_graph_index_a = []
            with open(os.path.abspath(os.path.join(
                defense_path,
                '../../../pygip/data/attack3_shadow_graph/' + self.dataset.dataset_name +
                '/protential_1300_shadow_graph_index.txt')), 'r') as fileObject:
                for ip in fileObject:
                    sub_graph_index_a.append(int(ip))

            attack_node = []
            while len(attack_node) < self.attack_node_number:
                protential_node_index = random.randint(0, len(sub_graph_index_b) - 1)
                protential_node = sub_graph_index_b[protential_node_index]
                if protential_node not in attack_node:
                    attack_node.append(int(protential_node))

            attack_features = self.features[attack_node].to(device)
            attack_labels = self.labels[attack_node].to(device)
            shadow_features = self.features[sub_graph_index_a].to(device)
            shadow_labels = self.labels[sub_graph_index_a].to(device)

            sub_graph_g_A = g_numpy[sub_graph_index_a]
            sub_graph_g_a = sub_graph_g_A[:, sub_graph_index_a]

            sub_graph_attack = g_numpy[attack_node]
            sub_graph_Attack = sub_graph_attack[:, attack_node]

            zeros_1 = np.zeros((len(attack_node), len(sub_graph_index_a)))
            zeros_2 = np.zeros((len(sub_graph_g_a), len(attack_node)))

            sub_graph_Attack = np.array(sub_graph_Attack)
            sub_graph_g_a = np.array(sub_graph_g_a)

            generated_graph_1 = np.concatenate((sub_graph_Attack, zeros_1), axis=1)
            generated_graph_2 = np.concatenate((zeros_2, sub_graph_g_a), axis=1)
            generated_graph = np.concatenate((generated_graph_1, generated_graph_2), axis=0)

            generated_features = torch.cat((attack_features, shadow_features), dim=0).to(device)
            generated_labels = torch.cat((attack_labels, shadow_labels), dim=0).to(device)

            generated_train_mask = torch.ones(len(generated_features), dtype=torch.bool, device=device)
            generated_test_mask = torch.ones(len(generated_features), dtype=torch.bool, device=device)

            generated_g = nx.from_numpy_array(generated_graph)
            generated_g.remove_edges_from(nx.selfloop_edges(generated_g))
            generated_g.add_edges_from(zip(generated_g.nodes(), generated_g.nodes()))
            generated_g = DGLGraph(generated_g)
            generated_g = generated_g.to(device)

            degs = generated_g.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.to(device)
            generated_g.ndata['norm'] = norm.unsqueeze(1)

            sub_graph_g_B = g_numpy[sub_graph_index_b]
            sub_graph_g_b = sub_graph_g_B[:, sub_graph_index_b]
            sub_graph_features_b = self.features[sub_graph_index_b].to(device)
            sub_graph_labels_b = self.labels[sub_graph_index_b].to(device)
            sub_graph_train_mask_b = self.train_mask[sub_graph_index_b].to(device)
            sub_graph_test_mask_b = self.test_mask[sub_graph_index_b].to(device)

            test_mask_length = min(len(sub_graph_test_mask_b), len(generated_train_mask))
            for i in range(test_mask_length):
                if i >= 140:
                    generated_train_mask[i] = 0
                    sub_graph_test_mask_b[i] = 1
                else:
                    generated_train_mask[i] = 1
                    sub_graph_test_mask_b[i] = 0

            if len(sub_graph_test_mask_b) > test_mask_length:
                sub_graph_test_mask_b[test_mask_length:] = 1

            sub_g_b = nx.from_numpy_array(sub_graph_g_b)
            sub_g_b.remove_edges_from(nx.selfloop_edges(sub_g_b))
            sub_g_b.add_edges_from(zip(sub_g_b.nodes(), sub_g_b.nodes()))
            sub_g_b = DGLGraph(sub_g_b)
            sub_g_b = sub_g_b.to(device)

            degs = sub_g_b.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.to(device)
            sub_g_b.ndata['norm'] = norm.unsqueeze(1)

            self.net1.eval()
            with torch.no_grad():
                logits_b = self.net1(sub_g_b, sub_graph_features_b)
                _, query_b = torch.max(logits_b, dim=1)

            net2 = Gcn_Net(self.feature_number, self.label_number).to(device)
            optimizer_a = torch.optim.Adam(net2.parameters(), lr=1e-2, weight_decay=5e-4)
            dur = []
            best_performance_metrics = GraphNeuralNetworkMetric()

            print("=========Model Extracting==========================")
            for epoch in tqdm(range(300)):
                if epoch >= 3:
                    t0 = time.time()

                net2.train()
                logits_a = net2(generated_g, generated_features)
                logp_a = F.log_softmax(logits_a, 1)
                loss_a = F.nll_loss(logp_a[generated_train_mask], generated_labels[generated_train_mask])

                optimizer_a.zero_grad()
                loss_a.backward()
                optimizer_a.step()

                if epoch >= 3:
                    dur.append(time.time() - t0)

                with torch.no_grad():
                    focus_gnn_metrics = GraphNeuralNetworkMetric(
                        0, 0, net2, sub_g_b, sub_graph_features_b,
                        sub_graph_test_mask_b, sub_graph_labels_b, query_b
                    )
                    focus_gnn_metrics.evaluate()

                    best_performance_metrics.fidelity = max(
                        best_performance_metrics.fidelity, focus_gnn_metrics.fidelity)
                    best_performance_metrics.accuracy = max(
                        best_performance_metrics.accuracy, focus_gnn_metrics.accuracy)

                if epoch % 10 == 0:
                    torch.cuda.empty_cache()

            print("========================Final results:=========================================")
            print(best_performance_metrics)

        except RuntimeError as e:
            print(f"Runtime error: {e}")
            torch.cuda.empty_cache()
            raise

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print(f"Error type: {type(e)}")
            torch.cuda.empty_cache()
            raise


class ModelExtractionAttack4(ModelExtractionAttack):
    """
    ModelExtractionAttack4.

    Another graph-based strategy that reads node indices from files,
    merges adjacency matrices, and links new edges based on feature similarity.

    Inherits
    --------
    ModelExtractionAttack
    """

    def __init__(self, dataset, attack_node_fraction, model_path=None):
        super().__init__(dataset, attack_node_fraction, model_path)
        self.model_path = model_path

    def attack(self):
        """
        Main attack procedure.

        1. Reads two sets of node indices from text files.
        2. Selects a fixed number of nodes from the target set for attack.
        3. Builds a combined adjacency matrix with zero blocks, then populates
           edges between shadow and attack nodes based on a distance threshold.
        4. Trains a new GCN on this combined graph and evaluates fidelity & accuracy.
        """
        try:
            torch.cuda.empty_cache()

            g_numpy = self.graph.adjacency_matrix().to_dense().cpu().numpy()
            defense_path = inspect.getfile(gnn_mea)

            sub_graph_index_b = []
            with open(os.path.abspath(os.path.join(
                defense_path, '../../../pygip/data/' + self.dataset.dataset_name +
                '/target_graph_index.txt')), 'r') as fileObject:
                for ip in fileObject:
                    sub_graph_index_b.append(int(ip))

            sub_graph_index_a = []
            with open(os.path.abspath(os.path.join(
                defense_path, '../../../pygip/data/' + self.dataset.dataset_name +
                '/protential_1200_shadow_graph_index.txt')), 'r') as fileObject:
                for ip in fileObject:
                    sub_graph_index_a.append(int(ip))

            attack_node_arg = 60
            attack_node = []
            while len(attack_node) < attack_node_arg:
                protential_node_index = random.randint(0, len(sub_graph_index_b) - 1)
                protential_node = sub_graph_index_b[protential_node_index]
                if protential_node not in attack_node:
                    attack_node.append(int(protential_node))

            attack_features = self.features[attack_node].cpu()
            attack_labels = self.labels[attack_node].cpu()
            shadow_features = self.features[sub_graph_index_a].cpu()
            shadow_labels = self.labels[sub_graph_index_a].cpu()

            sub_graph_g_A = np.array(g_numpy[sub_graph_index_a])
            sub_graph_g_a = np.array(sub_graph_g_A[:, sub_graph_index_a])
            sub_graph_Attack = np.zeros((len(attack_node), len(attack_node)))

            zeros_1 = np.zeros((len(attack_node), len(sub_graph_index_a)))
            zeros_2 = np.zeros((len(sub_graph_g_a), len(attack_node)))

            generated_graph = np.block([
                [sub_graph_Attack, zeros_1],
                [zeros_2, sub_graph_g_a]
            ])

            distance = []
            for i in range(100):
                index1 = i
                index2_list = np.nonzero(sub_graph_g_a[i])[0].tolist()
                for index2 in index2_list:
                    distance.append(float(np.linalg.norm(
                        shadow_features[index1].cpu().numpy() -
                        shadow_features[int(index2)].cpu().numpy())))

            threshold = np.mean(distance)
            max_threshold = max(distance)

            generated_features = np.vstack((attack_features.cpu().numpy(), shadow_features.cpu().numpy()))
            generated_labels = np.concatenate([attack_labels.cpu().numpy(), shadow_labels.cpu().numpy()])

            for i in range(len(attack_features)):
                for loop in range(1000):
                    j = random.randint(0, len(shadow_features) - 1)
                    if np.linalg.norm(generated_features[i] - generated_features[len(attack_features) + j]) < threshold:
                        generated_graph[i][len(attack_features) + j] = 1
                        generated_graph[len(attack_features) + j][i] = 1
                        break
                    if loop > 500:
                        if np.linalg.norm(generated_features[i] - generated_features[len(attack_features) + j]) < max_threshold:
                            generated_graph[i][len(attack_features) + j] = 1
                            generated_graph[len(attack_features) + j][i] = 1
                            break
                    if loop == 999:
                        print("one isolated node!")

            generated_train_mask = torch.ones(len(generated_features), dtype=torch.bool)
            generated_test_mask = torch.ones(len(generated_features), dtype=torch.bool)

            generated_features = torch.FloatTensor(generated_features).to(device)
            generated_labels = torch.LongTensor(generated_labels).to(device)
            generated_train_mask = generated_train_mask.to(device)
            generated_test_mask = generated_test_mask.to(device)

            generated_g = nx.from_numpy_array(generated_graph)
            generated_g.remove_edges_from(nx.selfloop_edges(generated_g))
            generated_g.add_edges_from(zip(generated_g.nodes(), generated_g.nodes()))
            generated_g = DGLGraph(generated_g)
            generated_g = generated_g.to(device)

            degs = generated_g.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.to(device)
            generated_g.ndata['norm'] = norm.unsqueeze(1)

            sub_graph_g_B = np.array(g_numpy[sub_graph_index_b])
            sub_graph_g_b = np.array(sub_graph_g_B[:, sub_graph_index_b])
            sub_graph_features_b = self.features[sub_graph_index_b].to(device)
            sub_graph_labels_b = self.labels[sub_graph_index_b].to(device)
            sub_graph_train_mask_b = self.train_mask[sub_graph_index_b].to(device)
            sub_graph_test_mask_b = self.test_mask[sub_graph_index_b].to(device)

            for i in range(len(sub_graph_test_mask_b)):
                if i >= 300:
                    sub_graph_train_mask_b[i] = 0
                    sub_graph_test_mask_b[i] = 1
                else:
                    sub_graph_train_mask_b[i] = 1
                    sub_graph_test_mask_b[i] = 0

            sub_g_b = nx.from_numpy_array(sub_graph_g_b)
            sub_g_b.remove_edges_from(nx.selfloop_edges(sub_g_b))
            sub_g_b.add_edges_from(zip(sub_g_b.nodes(), sub_g_b.nodes()))
            sub_g_b = DGLGraph(sub_g_b)
            sub_g_b = sub_g_b.to(device)

            degs = sub_g_b.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.to(device)
            sub_g_b.ndata['norm'] = norm.unsqueeze(1)

            self.net1.eval()
            with torch.no_grad():
                logits_b = self.net1(sub_g_b, sub_graph_features_b)
                _, query_b = torch.max(logits_b, dim=1)

            net2 = Gcn_Net(self.feature_number, self.label_number).to(device)
            optimizer_a = torch.optim.Adam(net2.parameters(), lr=1e-2, weight_decay=5e-4)
            dur = []
            best_performance_metrics = GraphNeuralNetworkMetric()

            print("=========Model Extracting==========================")
            for epoch in tqdm(range(300)):
                if epoch >= 3:
                    t0 = time.time()

                net2.train()
                logits_a = net2(generated_g, generated_features)
                logp_a = F.log_softmax(logits_a, 1)
                loss_a = F.nll_loss(logp_a[generated_train_mask], generated_labels[generated_train_mask])

                optimizer_a.zero_grad()
                loss_a.backward()
                optimizer_a.step()

                if epoch >= 3:
                    dur.append(time.time() - t0)

                with torch.no_grad():
                    focus_gnn_metrics = GraphNeuralNetworkMetric(
                        0, 0, net2, sub_g_b, sub_graph_features_b,
                        sub_graph_test_mask_b, sub_graph_labels_b, query_b
                    )
                    focus_gnn_metrics.evaluate()

                    best_performance_metrics.fidelity = max(
                        best_performance_metrics.fidelity, focus_gnn_metrics.fidelity)
                    best_performance_metrics.accuracy = max(
                        best_performance_metrics.accuracy, focus_gnn_metrics.accuracy)

                if epoch % 10 == 0:
                    torch.cuda.empty_cache()

            print("========================Final results:=========================================")
            print(best_performance_metrics)

        except RuntimeError as e:
            print(f"Runtime error: {e}")
            torch.cuda.empty_cache()
            raise

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print(f"Error type: {type(e)}")
            torch.cuda.empty_cache()
            raise


class ModelExtractionAttack5(ModelExtractionAttack):
    """
    ModelExtractionAttack5.

    Similar to ModelExtractionAttack4, but uses a slightly different
    strategy to link edges between nodes based on a threshold distance.

    Inherits
    --------
    ModelExtractionAttack
    """

    def __init__(self, dataset, attack_node_fraction, model_path=None):
        super().__init__(dataset, attack_node_fraction, model_path)
        self.model_path = model_path

    def attack(self):
        """
        Main attack procedure.

        1. Reads two sets of node indices (for target and shadow nodes).
        2. Builds a block adjacency matrix with all zero blocks, then links
           edges between attack nodes and shadow nodes if the feature distance
           is less than a threshold.
        3. Trains a new GCN on this combined graph and evaluates fidelity & accuracy.
        """
        try:
            torch.cuda.empty_cache()

            g_numpy = self.graph.adjacency_matrix().to_dense().cpu().numpy()
            defense_path = inspect.getfile(gnn_mea)

            sub_graph_index_b = []
            with open(os.path.abspath(os.path.join(
                defense_path,
                '../../../pygip/data/' + self.dataset.dataset_name +
                '/target_graph_index.txt')), 'r') as fileObject:
                for ip in fileObject:
                    sub_graph_index_b.append(int(ip))

            sub_graph_index_a = []
            with open(os.path.abspath(os.path.join(
                defense_path,
                '../../../pygip/data/' + self.dataset.dataset_name +
                '/protential_1200_shadow_graph_index.txt')), 'r') as fileObject:
                for ip in fileObject:
                    sub_graph_index_a.append(int(ip))

            attack_node = []
            while len(attack_node) < 60:
                protential_node_index = random.randint(0, len(sub_graph_index_b) - 1)
                protential_node = sub_graph_index_b[protential_node_index]
                if protential_node not in attack_node:
                    attack_node.append(int(protential_node))

            attack_features = self.features[attack_node].cpu()
            attack_labels = self.labels[attack_node].cpu()
            shadow_features = self.features[sub_graph_index_a].cpu()
            shadow_labels = self.labels[sub_graph_index_a].cpu()

            sub_graph_g_A = np.array(g_numpy[sub_graph_index_a])
            sub_graph_g_a = np.array(sub_graph_g_A[:, sub_graph_index_a])
            sub_graph_Attack = np.zeros((len(attack_node), len(attack_node)))

            zeros_1 = np.zeros((len(attack_node), len(sub_graph_index_a)))
            zeros_2 = np.zeros((len(sub_graph_g_a), len(attack_node)))

            generated_graph = np.block([
                [sub_graph_Attack, zeros_1],
                [zeros_2, sub_graph_g_a]
            ])

            distance = []
            for i in range(100):
                index1 = i
                index2_list = np.nonzero(sub_graph_g_a[i])[0].tolist()
                for index2 in index2_list:
                    distance.append(float(np.linalg.norm(
                        shadow_features[index1].cpu().numpy() -
                        shadow_features[int(index2)].cpu().numpy())))

            threshold = np.mean(distance)
            max_threshold = max(distance)

            generated_features = np.vstack((attack_features.cpu().numpy(),
                                            shadow_features.cpu().numpy()))
            generated_labels = np.concatenate([attack_labels.cpu().numpy(),
                                               shadow_labels.cpu().numpy()])

            for i in range(len(attack_features)):
                for loop in range(1000):
                    j = random.randint(0, len(shadow_features) - 1)
                    feat_diff = generated_features[i] - generated_features[len(attack_features) + j]
                    dist = np.linalg.norm(feat_diff)

                    if dist < threshold:
                        generated_graph[i][len(attack_features) + j] = 1
                        generated_graph[len(attack_features) + j][i] = 1
                        break
                    if loop > 500 and dist < max_threshold:
                        generated_graph[i][len(attack_features) + j] = 1
                        generated_graph[len(attack_features) + j][i] = 1
                        break
                    if loop == 999:
                        print("one isolated node!")

            generated_features = torch.FloatTensor(generated_features).to(device)
            generated_labels = torch.LongTensor(generated_labels).to(device)
            generated_train_mask = torch.ones(len(generated_features), dtype=torch.bool, device=device)
            generated_test_mask = torch.ones(len(generated_features), dtype=torch.bool, device=device)

            generated_g = nx.from_numpy_array(generated_graph)
            generated_g.remove_edges_from(nx.selfloop_edges(generated_g))
            generated_g.add_edges_from(zip(generated_g.nodes(), generated_g.nodes()))
            generated_g = DGLGraph(generated_g)
            generated_g = generated_g.to(device)

            degs = generated_g.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.to(device)
            generated_g.ndata['norm'] = norm.unsqueeze(1)

            sub_graph_g_B = np.array(g_numpy[sub_graph_index_b])
            sub_graph_g_b = np.array(sub_graph_g_B[:, sub_graph_index_b])
            sub_graph_features_b = self.features[sub_graph_index_b].to(device)
            sub_graph_labels_b = self.labels[sub_graph_index_b].to(device)
            sub_graph_train_mask_b = self.train_mask[sub_graph_index_b].to(device)
            sub_graph_test_mask_b = self.test_mask[sub_graph_index_b].to(device)

            for i in range(len(sub_graph_test_mask_b)):
                if i >= 300:
                    sub_graph_train_mask_b[i] = 0
                    sub_graph_test_mask_b[i] = 1
                else:
                    sub_graph_train_mask_b[i] = 1
                    sub_graph_test_mask_b[i] = 0

            sub_g_b = nx.from_numpy_array(sub_graph_g_b)
            sub_g_b.remove_edges_from(nx.selfloop_edges(sub_g_b))
            sub_g_b.add_edges_from(zip(sub_g_b.nodes(), sub_g_b.nodes()))
            sub_g_b = DGLGraph(sub_g_b)
            sub_g_b = sub_g_b.to(device)

            degs = sub_g_b.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.to(device)
            sub_g_b.ndata['norm'] = norm.unsqueeze(1)

            self.net1.eval()
            with torch.no_grad():
                logits_b = self.net1(sub_g_b, sub_graph_features_b)
                _, query_b = torch.max(logits_b, dim=1)

            net2 = Gcn_Net(self.feature_number, self.label_number).to(device)
            optimizer_a = torch.optim.Adam(net2.parameters(), lr=1e-2, weight_decay=5e-4)
            dur = []
            best_performance_metrics = GraphNeuralNetworkMetric()

            print("=========Model Extracting==========================")
            for epoch in tqdm(range(300)):
                if epoch >= 3:
                    t0 = time.time()

                net2.train()
                logits_a = net2(generated_g, generated_features)
                logp_a = F.log_softmax(logits_a, 1)
                loss_a = F.nll_loss(logp_a[generated_train_mask],
                                    generated_labels[generated_train_mask])

                optimizer_a.zero_grad()
                loss_a.backward()
                optimizer_a.step()

                if epoch >= 3:
                    dur.append(time.time() - t0)

                with torch.no_grad():
                    focus_gnn_metrics = GraphNeuralNetworkMetric(
                        0, 0, net2, sub_g_b, sub_graph_features_b,
                        sub_graph_test_mask_b, sub_graph_labels_b, query_b
                    )
                    focus_gnn_metrics.evaluate()

                    best_performance_metrics.fidelity = max(
                        best_performance_metrics.fidelity, focus_gnn_metrics.fidelity)
                    best_performance_metrics.accuracy = max(
                        best_performance_metrics.accuracy, focus_gnn_metrics.accuracy)

                if epoch % 10 == 0:
                    torch.cuda.empty_cache()

            print("========================Final results:=========================================")
            print(best_performance_metrics)

        except RuntimeError as e:
            print(f"Runtime error: {e}")
            torch.cuda.empty_cache()
            raise

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print(f"Error type: {type(e)}")
            torch.cuda.empty_cache()
            raise
