"""
metrics.py

This module provides functions to compute basic network statistics,
correlations between centrality measures, and to evaluate diffusion spread via repeated simulations.
"""

import numpy as np
import networkx as nx
from scipy import stats
import random
import diffusion  # Ensure the diffusion module is accessible

def basic_network_stats(G):
    """Return statistics of graph G."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)
    avg_deg = (sum(dict(G.degree()).values()) / n) if n > 0 else 0
    clustering = nx.average_clustering(G) if not G.is_directed() else None
    return {"nodes": n, "edges": m, "density": density, "average_degree": avg_deg, "avg_clustering_coef": clustering}

def centrality_rank_correlation(G, cent_func1, cent_func2):
    """
    Compute the Spearman rank correlation between two centrality measures.
    """
    vals1 = cent_func1(G)
    vals2 = cent_func2(G)
    ranking1 = sorted(vals1.items(), key=lambda x: x[1], reverse=True)
    ranking2 = sorted(vals2.items(), key=lambda x: x[1], reverse=True)
    rank_pos1 = {node: i for i, (node, _) in enumerate(ranking1)}
    rank_pos2 = {node: i for i, (node, _) in enumerate(ranking2)}
    common_nodes = set(rank_pos1.keys())
    ranks1 = [rank_pos1[node] for node in common_nodes]
    ranks2 = [rank_pos2[node] for node in common_nodes]
    corr, _ = stats.spearmanr(ranks1, ranks2)
    return corr

def diffusion_spread(model, seeds, runs=100):
    """
    Evaluate the average cascade spread using a diffusion model.
    """
    total_activated_counts = []
    for _ in range(runs):
        activated = model.simulate(seeds)
        total_activated_counts.append(len(activated))
        model.reset()
    mean_spread = float(np.mean(total_activated_counts))
    std_spread = float(np.std(total_activated_counts))
    return {"mean_spread": mean_spread, "std_spread": std_spread, "runs": runs, "seed_set_size": len(seeds)}

def cascade_steps(model, seeds):
    """
    Return the activation history (list of nodes activated per step) for the diffusion simulation.
    """
    model.reset()
    active = list(seeds)
    history = [list(seeds)]
    model.active_set = set(seeds)
    if hasattr(model, 'simulate'):
        if isinstance(model, diffusion.IndependentCascadeModel):
            frontier = active
            while frontier:
                new_active = []
                for u in frontier:
                    for v in model.G.neighbors(u):
                        if v not in model.active_set:
                            if random.random() < model.p:
                                new_active.append(v)
                                model.active_set.add(v)
                if not new_active:
                    break
                history.append(new_active)
                frontier = new_active
        elif isinstance(model, diffusion.LinearThresholdModel):
            while True:
                new_active = []
                for node in set(model.G.nodes()) - model.active_set:
                    total_influence = 0.0
                    neighbors = model.G.predecessors(node) if model.G.is_directed() else model.G.neighbors(node)
                    for nbr in neighbors:
                        if nbr in model.active_set:
                            edge_data = model.G.get_edge_data(nbr, node, default={})
                            total_influence += edge_data.get(model.weight, 1.0)
                    if total_influence >= model.threshold[node]:
                        new_active.append(node)
                if not new_active:
                    break
                for v in new_active:
                    model.active_set.add(v)
                history.append(new_active)
    return history
