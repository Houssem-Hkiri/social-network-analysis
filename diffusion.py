"""
diffusion.py

This module implements diffusion models for simulating information
propagation in social networks using graph theory. The two main models
provided are the Independent Cascade (IC) Model and the Linear Threshold
(LT) Model. Additionally, a CompositeInfluenceAnalyzer class is implemented
to estimate the diffusion-based influence score of each node using Monte Carlo
simulations.

References:
  - Kempe, Kleinberg, and Tardos (2003), “Maximizing the spread of influence
    through a social network.”
  - Granovetter (1978) on threshold models.
"""

import random
import networkx as nx
import numpy as np


# =============================================================
# DIFFUSION MODELS
# =============================================================

class DiffusionModel:
    """
    Abstract base class for diffusion models on graphs.

    Diffusion models simulate the spread of "activation" (for example,
    information or contagion) throughout a network. They are often used
    to assess the dynamic influence of nodes by tracking which nodes become
    activated when diffusion starts from a set of seed nodes.
    """
    def __init__(self, G):
        """
        Initialize the diffusion model with graph G.

        Args:
            G (networkx.Graph): The network graph.
        """
        self.G = G
        self.active_set = set()

    def reset(self):
        """
        Reset the active set before starting a new diffusion simulation.
        """
        self.active_set = set()

    def simulate(self, initial_active):
        """
        Run a diffusion simulation starting from the 'initial_active' seed nodes.
        Subclasses must implement this method.

        Args:
            initial_active (list): List of seed node IDs.

        Returns:
            set: The set of nodes activated at the end of the simulation.
        """
        raise NotImplementedError("simulate() must be implemented by subclasses.")


class IndependentCascadeModel(DiffusionModel):
    """
    Independent Cascade (IC) Model:

    Once a node becomes activated, it gets a one-time chance to "activate"
    each of its inactive neighbors with probability p. If an edge has a weight
    attribute (interpreted as the activation probability), that is used.
    Otherwise, a default probability p is applied.

    This model captures a one-shot spread of influence, such as viral tweeting.
    """
    def __init__(self, G, p=0.1):
        """
        Args:
            G (networkx.Graph): The network graph.
            p (float): Default probability of activation if an edge weight is not specified.
        """
        super().__init__(G)
        self.p = p

    def simulate(self, initial_active):
        """
        Simulate the Independent Cascade process.

        Args:
            initial_active (list): List of initial seed node IDs.

        Returns:
            set: The final set of activated nodes.
        """
        self.reset()
        self.active_set = set(initial_active)
        # Record cascade steps for potential analysis (optional)
        activated_over_time = [set(initial_active)]
        frontier = list(initial_active)

        while frontier:
            new_active = []
            for u in frontier:
                for v in self.G.neighbors(u):
                    if v not in self.active_set:
                        # Use edge weight if available; otherwise, use default self.p.
                        edge_data = self.G.get_edge_data(u, v, default={})
                        p_edge = edge_data.get("weight", self.p)
                        if random.random() < p_edge:
                            new_active.append(v)
                            self.active_set.add(v)
            if not new_active:
                break
            activated_over_time.append(set(new_active))
            frontier = new_active
        return self.active_set


class LinearThresholdModel(DiffusionModel):
    """
    Linear Threshold (LT) Model:

    In the LT model, each node \(v\) is assigned a threshold \(\theta_v\)
    (typically drawn uniformly from [0,1]). Each incoming edge \((u,v)\)
    is associated with an influence weight. Node \(v\) becomes activated when the sum
    of influence weights from its active neighbors meets or exceeds \(\theta_v\).

    This model captures peer pressure and cumulative influence effects.
    """

    def __init__(self, G, weight='weight'):
        """
        Args:
            G (networkx.Graph): The network graph.
            weight (str): The key for edge weight in the graph data.
        """
        # Ensure that we work with a directed graph for proper threshold evaluation.
        if not G.is_directed():
            G = G.to_directed()
        super().__init__(G)
        self.weight = weight
        # Assign a random threshold from [0,1] to each node.
        self.threshold = {node: random.random() for node in self.G.nodes()}
        # Normalize the incoming edge weights for each node so that their sum equals 1.
        for node in self.G.nodes():
            in_edges = list(self.G.in_edges(node, data=True))
            total_w = sum(data.get(self.weight, 1.0) for _, _, data in in_edges)
            # If there is no incoming weight, skip normalization.
            if total_w == 0:
                continue
            for u, v, data in in_edges:
                data[self.weight] = data.get(self.weight, 1.0) / total_w

    def simulate(self, initial_active):
        """
        Simulate the Linear Threshold process.

        Args:
            initial_active (list): List of initial seed node IDs.

        Returns:
            set: The final set of activated nodes.
        """
        self.reset()
        self.active_set = set(initial_active)
        activated_over_time = [set(initial_active)]

        while True:
            new_active = []
            # Consider nodes that are not yet activated.
            for node in set(self.G.nodes()) - self.active_set:
                total_influence = 0.0
                # In a directed graph, consider all in-neighbors.
                for nbr in self.G.predecessors(node):
                    if nbr in self.active_set:
                        edge_data = self.G.get_edge_data(nbr, node, default={})
                        total_influence += edge_data.get(self.weight, 1.0)
                if total_influence >= self.threshold[node]:
                    new_active.append(node)
            if not new_active:
                break
            # Update activated nodes.
            for v in new_active:
                self.active_set.add(v)
            activated_over_time.append(set(new_active))

        return self.active_set


# =============================================================
# COMPOSITE INFLUENCE ANALYZER (Diffusion-based Influence Score)
# =============================================================

class CompositeInfluenceAnalyzer:
    """
    The CompositeInfluenceAnalyzer estimates a diffusion-based influence score for each node
    by running multiple Monte Carlo simulations with each node as the sole seed. The average
    cascade size over several runs indicates the node's dynamic influence.

    This approach helps complement static centrality measures by providing an empirical, simulation-based
    evaluation of a node's influence.

    To reduce computation time on large networks, users can restrict the evaluation to a subset
    of nodes and adjust the number of simulation runs.
    """
    def __init__(self, G, diffusion_model_class=IndependentCascadeModel, model_params=None, runs=50):
        """
        Args:
            G (networkx.Graph): The network graph.
            diffusion_model_class (class): The diffusion model to use (e.g., IndependentCascadeModel).
            model_params (dict): Parameters for the diffusion model, e.g., {'p': 0.1}.
            runs (int): Number of simulation runs per node.
        """
        self.G = G
        self.diffusion_model_class = diffusion_model_class
        self.model_params = model_params if model_params is not None else {}
        self.runs = runs

    def simulate_node_spread(self, node):
        """
        Run repeated diffusion simulations starting from a single node to compute the average cascade spread.

        Args:
            node: The node ID to test as the sole seed.

        Returns:
            float: The average number of nodes activated over self.runs simulations.
        """
        cascade_sizes = []
        for _ in range(self.runs):
            model = self.diffusion_model_class(self.G, **self.model_params)
            activated = model.simulate([node])
            cascade_sizes.append(len(activated))
        return np.mean(cascade_sizes)

    def compute_influence_scores(self, node_subset=None, progress_callback=None):
        """
        Compute the diffusion-based influence score for nodes in the specified subset.
        If node_subset is None, computes for all nodes.

        Args:
            node_subset (iterable): Subset of nodes to evaluate.
            progress_callback (function): Function to call with (current, total) progress.

        Returns:
            dict: A dictionary mapping node IDs to their average cascade spread.
        """
        if node_subset is None:
            node_subset = self.G.nodes()

        total_nodes = len(node_subset)
        influence_scores = {}

        for i, node in enumerate(node_subset, start=1):
            influence_scores[node] = self.simulate_node_spread(node)
            if progress_callback:
                progress_callback(i, total_nodes)

        return influence_scores

    def get_top_influencers(self, top_k=10, node_subset=None, progress_callback=None):
        """
        Rank nodes in the subset (or all nodes) by their diffusion-based influence score and return the top_k.

        Args:
            top_k (int): Number of top nodes to return.
            node_subset (iterable): Subset of nodes to evaluate.
            progress_callback (function): Progress update callback.

        Returns:
            list: List of tuples (node, influence_score) sorted in descending order.
        """
        scores = self.compute_influence_scores(node_subset=node_subset, progress_callback=progress_callback)
        top_influencers = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return top_influencers


# =============================================================
# EXAMPLE USAGE (for testing or debugging)
# =============================================================
if __name__ == '__main__':
    # Create a sample directed Erdős–Rényi graph for demonstration
    G = nx.erdos_renyi_graph(100, 0.05, seed=42, directed=True)
    # Assign random weights for LT model demonstration
    for u, v in G.edges():
        G[u][v]['weight'] = random.random()

    # Test Independent Cascade Model
    ic_model = IndependentCascadeModel(G, p=0.1)
    seeds = [0]  # Demo seed node
    final_active_ic = ic_model.simulate(seeds)
    print("Independent Cascade activated nodes:", final_active_ic)

    # Test Linear Threshold Model
    lt_model = LinearThresholdModel(G, weight='weight')
    final_active_lt = lt_model.simulate(seeds)
    print("Linear Threshold activated nodes:", final_active_lt)

    # Test Composite Influence Analyzer
    analyzer = CompositeInfluenceAnalyzer(
        G, diffusion_model_class=IndependentCascadeModel, model_params={'p': 0.1}, runs=50
    )
    scores = analyzer.compute_influence_scores()
    top_influencers = analyzer.get_top_influencers(top_k=5)
    print("Top influencers based on diffusion-based spread:", top_influencers)
