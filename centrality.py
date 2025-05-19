"""
centrality.py

This module implements various centrality measures used in Social Network Analysis.
The measures include:
    - Degree Centrality
    - Closeness Centrality
    - Betweenness Centrality (with optional approximation)
    - PageRank Centrality
    - Composite Centrality Score (an average of the above)
"""

import networkx as nx

def degree_centrality(G):
    """Compute degree centrality for each node in graph G."""
    return nx.degree_centrality(G)

def closeness_centrality(G):
    """Compute closeness centrality for each node in graph G."""
    return nx.closeness_centrality(G)

def betweenness_centrality(G, k=None):
    """Compute betweenness centrality for each node in graph G."""
    return nx.betweenness_centrality(G, k=k, normalized=True)

def pagerank_centrality(G, alpha=0.85):
    """Compute PageRank centrality for each node in graph G."""
    return nx.pagerank(G, alpha=alpha)

def composite_centrality(G):
    """Compute a composite score by averaging normalized centrality measures."""
    deg = degree_centrality(G)
    clos = closeness_centrality(G)
    between = betweenness_centrality(G)
    pr = pagerank_centrality(G)
    composite = {}
    for node in G.nodes():
        composite[node] = (deg[node] + clos[node] + between[node] + pr[node]) / 4.0
    return composite
